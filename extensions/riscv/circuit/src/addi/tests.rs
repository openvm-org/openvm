use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, PreflightExecutor, BLOCK_FE_WIDTH,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::BaseAluImmOpcode;
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use super::{core::run_addi, AddICoreAir, Rv64AddIChip, Rv64AddIExecutor};
use crate::{
    adapters::{
        rv64_bytes_to_u16_block, rv64_u16_block_to_bytes, Rv64BaseAluImmU16AdapterAir,
        Rv64BaseAluImmU16AdapterExecutor, Rv64BaseAluImmU16AdapterFiller, RV64_REGISTER_NUM_LIMBS,
        U16_BITS,
    },
    addi::AddICoreCols,
    test_utils::{generate_rv64_is_type_immediate, rv64_rand_write_register_or_imm},
    AddIFiller, Rv64AddIAir,
};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness = TestChipHarness<F, Rv64AddIExecutor, Rv64AddIAir, Rv64AddIChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64AddIAir, Rv64AddIExecutor, Rv64AddIChip<F>) {
    let air = Rv64AddIAir::new(
        Rv64BaseAluImmU16AdapterAir::new(execution_bridge, memory_bridge, range_checker_chip.bus()),
        AddICoreAir::new(range_checker_chip.bus(), BaseAluImmOpcode::CLASS_OFFSET),
    );
    let executor = Rv64AddIExecutor::new(
        Rv64BaseAluImmU16AdapterExecutor,
        BaseAluImmOpcode::CLASS_OFFSET,
    );
    let chip = Rv64AddIChip::new(
        AddIFiller::new(
            Rv64BaseAluImmU16AdapterFiller::new(range_checker_chip.clone()),
            range_checker_chip,
            BaseAluImmOpcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        range_checker,
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

// ADDI is always immediate — no opcode or is_imm parameter needed.
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    b: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    c: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let (imm, c) = if let Some(c) = c {
        ((u64::from_le_bytes(c) & 0xFFFFFF) as usize, c)
    } else {
        generate_rv64_is_type_immediate(rng)
    };

    let (instruction, rd) = rv64_rand_write_register_or_imm(
        tester,
        b,
        c,
        Some(imm),
        BaseAluImmOpcode::ADDI.global_opcode().as_usize(),
        rng,
    );
    tester.execute(executor, arena, &instruction);

    let b_u16 = rv64_bytes_to_u16_block(b);
    let imm_low11 = (imm as u32 & 0x7FF) as u16;
    let imm_sign = ((imm as u32 >> 11) & 1) as u16;
    let a = run_addi::<BLOCK_FE_WIDTH, U16_BITS>(&b_u16, imm_low11, imm_sign);
    let a_bytes = rv64_u16_block_to_bytes(a).map(F::from_u8);
    assert_eq!(a_bytes, tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd))
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//////////////////////////////////////////////////////////////////////////////////////

#[test]
fn rand_rv64_addi_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    for _ in 0..100 {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            None,
            None,
        );
    }

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//////////////////////////////////////////////////////////////////////////////////////

fn run_negative_addi_test(
    prank_rd: [u32; BLOCK_FE_WIDTH],
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_rs1: Option<[u32; BLOCK_FE_WIDTH]>,
    prank_imm_sign: Option<u32>,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        Some(b),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut AddICoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.rd = prank_rd.map(F::from_u32);
        if let Some(prank_rs1) = prank_rs1 {
            cols.rs1 = prank_rs1.map(F::from_u32);
        }
        if let Some(prank_imm_sign) = prank_imm_sign {
            cols.imm_sign = F::from_u32(prank_imm_sign);
        }
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}

#[test]
fn rv64_addi_wrong_negative_test() {
    run_negative_addi_test(
        [499, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
    );
}

#[test]
fn rv64_addi_out_of_range_negative_test() {
    // b[0] = c[0] = 65535; the correct result is [65534, 1, 0, 0]. Pranking
    // a = [131070, 0, 0, 0] satisfies every carry constraint (carry[0] = 0),
    // so only the 16-bit range check on a[0] can catch it.
    run_negative_addi_test(
        [131070, 0, 0, 0],
        [255, 255, 0, 0, 0, 0, 0, 0],
        [255, 255, 0, 0, 0, 0, 0, 0],
        None,
        None,
    );
}

#[test]
fn rv64_addi_imm_sign_extension_negative_test() {
    // imm = 5 (positive, so imm_sign should be 0). Prank imm_sign = 1:
    // instr_c reconstructs to 5 + 0xFFF800 which doesn't match instruction.c = 5,
    // so the execution bridge interaction fails.
    run_negative_addi_test(
        [5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0],
        None,
        Some(1),
    );
}

#[test]
fn rv64_addi_noncanonical_b_negative_test() {
    // Prank core b[0] = 2^16 with a = [0, 1, 0, 0] as compensation: every carry
    // constraint holds (carry[0] = 1) and all a cells are canonical, so only the
    // memory bus read interaction can catch that b doesn't match rs1 in memory.
    run_negative_addi_test(
        [0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        Some([1 << U16_BITS, 0, 0, 0]), // prank rs1[0] out of range
        None,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
// SANITY TESTS
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_addi_sanity_test() {
    // rs1 = 500, imm = -3 (imm_low11 = 0x7FD, imm_sign = 1) → rd = 497
    let rs1 = rv64_bytes_to_u16_block([0xF4, 0x01, 0, 0, 0, 0, 0, 0]);
    let expected = rv64_bytes_to_u16_block([0xF1, 0x01, 0, 0, 0, 0, 0, 0]);
    let result = run_addi::<BLOCK_FE_WIDTH, U16_BITS>(&rs1, 0x7FD, 1);
    assert_eq!(expected, result);
}
