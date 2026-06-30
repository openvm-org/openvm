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
use openvm_riscv_transpiler::AddIOpcode;
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

use super::{core::run_add, AddICoreAir, Rv64AddIChip, Rv64AddIExecutor};
use crate::{
    adapters::{
        rv64_bytes_to_u16_block, rv64_u16_block_to_bytes, Rv64AddIAdapterAir,
        Rv64AddIAdapterExecutor, Rv64AddIAdapterFiller, RV64_REGISTER_NUM_LIMBS, U16_BITS,
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
        Rv64AddIAdapterAir::new(execution_bridge, memory_bridge, range_checker_chip.bus()),
        AddICoreAir::new(range_checker_chip.bus(), AddIOpcode::CLASS_OFFSET),
    );
    let executor = Rv64AddIExecutor::new(Rv64AddIAdapterExecutor, AddIOpcode::CLASS_OFFSET);
    let chip = Rv64AddIChip::new(
        AddIFiller::new(
            Rv64AddIAdapterFiller::new(range_checker_chip.clone()),
            range_checker_chip,
            AddIOpcode::CLASS_OFFSET,
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
        AddIOpcode::ADDI.global_opcode().as_usize(),
        rng,
    );
    tester.execute(executor, arena, &instruction);

    let b_u16 = rv64_bytes_to_u16_block(b);
    let c_u16 = rv64_bytes_to_u16_block(c);
    let a = run_add::<BLOCK_FE_WIDTH, U16_BITS>(&b_u16, &c_u16);
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
    prank_a: [u32; BLOCK_FE_WIDTH],
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_b: Option<[u32; BLOCK_FE_WIDTH]>,
    prank_c: Option<[u32; BLOCK_FE_WIDTH]>,
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
        cols.a = prank_a.map(F::from_u32);
        if let Some(prank_b) = prank_b {
            cols.b = prank_b.map(F::from_u32);
        }
        if let Some(prank_c) = prank_c {
            cols.c = prank_c.map(F::from_u32);
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
    // Prank core c[2] = 1 while the adapter's rs2_imm_sign cell is 0. The adapter
    // must catch that the high u16 limbs don't match the sign cell. Also prank
    // a[2] = 1 so the ADD carry constraint (a = b + c) still holds.
    run_negative_addi_test(
        [5, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0],
        None,
        Some([5, 0, 1, 0]),
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
        Some([1 << U16_BITS, 0, 0, 0]),
        None,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
// SANITY TESTS
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_addi_sanity_test() {
    let x = rv64_bytes_to_u16_block([229, 33, 29, 111, 145, 34, 25, 205]);
    let y = rv64_bytes_to_u16_block([50, 171, 44, 194, 73, 35, 25, 206]);
    let z = rv64_bytes_to_u16_block([23, 205, 73, 49, 219, 69, 50, 155]);
    let result = run_add::<BLOCK_FE_WIDTH, U16_BITS>(&x, &y);
    for i in 0..BLOCK_FE_WIDTH {
        assert_eq!(z[i], result[i])
    }
}
