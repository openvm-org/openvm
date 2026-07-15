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

use super::{AddICoreAir, Rv64AddIChip, Rv64AddIExecutor};
use crate::{
    adapters::{
        Rv64BaseAluImmU16AdapterAir, Rv64BaseAluImmU16AdapterExecutor,
        Rv64BaseAluImmU16AdapterFiller, RV64_REGISTER_NUM_LIMBS, U16_BITS,
    },
    addi::AddICoreCols,
    test_utils::{generate_rv64_is_type_immediate, rv64_rand_write_register_or_imm},
    AddIFiller, Rv64AddIAir,
};

const MAX_INS_CAPACITY: usize = 128;
const NONCANONICAL_ZERO: [u32; BLOCK_FE_WIDTH] = [
    1 << U16_BITS,
    (1 << U16_BITS) - 1,
    (1 << U16_BITS) - 1,
    (1 << U16_BITS) - 1,
];
type F = BabyBear;
type Harness = TestChipHarness<F, Rv64AddIExecutor, Rv64AddIAir, Rv64AddIChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64AddIAir, Rv64AddIExecutor, Rv64AddIChip<F>) {
    let air = Rv64AddIAir::new(
        Rv64BaseAluImmU16AdapterAir::new(execution_bridge, memory_bridge),
        AddICoreAir::new(
            range_checker_chip.bus(),
            BaseAluImmOpcode::CLASS_OFFSET,
            BaseAluImmOpcode::ADDI as usize,
        ),
    );
    let executor = Rv64AddIExecutor::new(
        Rv64BaseAluImmU16AdapterExecutor,
        BaseAluImmOpcode::CLASS_OFFSET,
        BaseAluImmOpcode::ADDI as usize,
    );
    let chip = Rv64AddIChip::new(
        AddIFiller::new(Rv64BaseAluImmU16AdapterFiller::new(), range_checker_chip),
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

    let rs1 = u64::from_le_bytes(b);
    let signed_imm = ((imm as u32) << 20) as i32 >> 20;
    let expected = rs1.wrapping_add(signed_imm as i64 as u64);
    assert_eq!(
        expected.to_le_bytes().map(F::from_u8),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    )
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

#[test]
fn rv64_addi_rs1_memory_binding_negative_test() {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        Some([0; RV64_REGISTER_NUM_LIMBS]),
        Some([0; RV64_REGISTER_NUM_LIMBS]),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut AddICoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        // These limbs represent 2^64, so ADDI 0 still produces zero and all core
        // constraints hold. Only the rs1 memory-read interaction rejects the row.
        cols.rs1 = NONCANONICAL_ZERO.map(F::from_u32);
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
