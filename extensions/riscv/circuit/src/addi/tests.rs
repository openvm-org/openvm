use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
        ExecutionBridge, BLOCK_FE_WIDTH,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::{BaseAluImmOpcode, BaseAluWImmOpcode};
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
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::AddIWChipGpu,
    openvm_circuit::arch::testing::{
        default_var_range_checker_bus, GpuChipTestBuilder, GpuTestChipHarness,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    std::sync::Arc,
};

use super::{
    trace::{generate_trace_from_postflight, generate_w_trace_from_postflight},
    AddIChip, AddICoreAir, AddIExecutor, AddIWChip, AddIWExecutor,
};
use crate::{
    adapters::{BaseAluImmU16AdapterAir, BaseAluWImmU16AdapterAir, REGISTER_NUM_LIMBS, U16_BITS},
    addi::AddICoreCols,
    test_utils::{generate_is_type_immediate, rand_write_register_or_imm},
    AddIAir, AddIFiller, AddIWAir,
};

const MAX_INS_CAPACITY: usize = 128;
const NONCANONICAL_ZERO: [u32; BLOCK_FE_WIDTH] = [
    1 << U16_BITS,
    (1 << U16_BITS) - 1,
    (1 << U16_BITS) - 1,
    (1 << U16_BITS) - 1,
];
type F = BabyBear;
type Harness = TestChipHarness<F, AddIExecutor, AddIAir, AddIChip<F>>;
type WHarness = TestChipHarness<F, AddIWExecutor, AddIWAir, AddIWChip<F>>;
#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuWHarness = GpuTestChipHarness<F, AddIWExecutor, AddIWAir, AddIWChipGpu, AddIWChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (AddIAir, AddIExecutor, AddIChip<F>) {
    let air = AddIAir::new(
        BaseAluImmU16AdapterAir::new(execution_bridge, memory_bridge),
        AddICoreAir::new(
            range_checker_chip.bus(),
            BaseAluImmOpcode::CLASS_OFFSET,
            BaseAluImmOpcode::ADDI as usize,
        ),
    );
    let executor = AddIExecutor::new(BaseAluImmOpcode::CLASS_OFFSET);
    let chip = AddIChip::new(AddIFiller::new(range_checker_chip), memory_helper);
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
    Harness::with_capacity(
        executor,
        air,
        chip,
        MAX_INS_CAPACITY,
        generate_trace_from_postflight,
    )
}

fn create_w_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (AddIWAir, AddIWExecutor, AddIWChip<F>) {
    let air = AddIWAir::new(
        BaseAluWImmU16AdapterAir::new(execution_bridge, memory_bridge, range_checker.bus()),
        AddICoreAir::new(
            range_checker.bus(),
            BaseAluWImmOpcode::CLASS_OFFSET,
            BaseAluWImmOpcode::ADDIW as usize,
        ),
    );
    let executor = AddIWExecutor::new(BaseAluWImmOpcode::CLASS_OFFSET);
    let chip = AddIWChip::new(AddIFiller::new(range_checker), memory_helper);
    (air, executor, chip)
}

fn create_w_harness(tester: &VmChipTestBuilder<F>) -> WHarness {
    let (air, executor, chip) = create_w_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker(),
        tester.memory_helper(),
    );
    WHarness::with_capacity(
        executor,
        air,
        chip,
        MAX_INS_CAPACITY,
        generate_w_trace_from_postflight,
    )
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_cuda_w_harness(tester: &GpuChipTestBuilder) -> GpuWHarness {
    let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));
    let (air, executor, cpu_chip) = create_w_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = AddIWChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 8).with_trace_generators(
        generate_w_trace_from_postflight,
        |chip, program, transcript, plan| {
            chip.generate_proving_ctx_from_postflight(program, transcript, plan)
        },
    )
}

fn set_and_execute<E: openvm_circuit::arch::Executor<F> + Clone>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    preflight: &mut openvm_circuit::arch::testing::TestPreflight<F>,
    rng: &mut StdRng,
    b: Option<[u8; REGISTER_NUM_LIMBS]>,
    c: Option<[u8; REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let (imm, c) = if let Some(c) = c {
        ((u64::from_le_bytes(c) & 0xFFFFFF) as usize, c)
    } else {
        generate_is_type_immediate(rng)
    };

    let (instruction, rd) = rand_write_register_or_imm(
        tester,
        b,
        c,
        Some(imm),
        BaseAluImmOpcode::ADDI.global_opcode().as_usize(),
        rng,
    );
    tester.execute(executor, preflight, &instruction);

    let rs1 = u64::from_le_bytes(b);
    let signed_imm = ((imm as u32) << 20) as i32 >> 20;
    let expected = rs1.wrapping_add(signed_imm as i64 as u64);
    assert_eq!(
        expected.to_le_bytes().map(F::from_u8),
        tester.read_bytes::<REGISTER_NUM_LIMBS>(1, rd)
    )
}

fn set_and_execute_w<E: openvm_circuit::arch::Executor<F> + Clone>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    preflight: &mut openvm_circuit::arch::testing::TestPreflight<F>,
    rng: &mut StdRng,
    rs1: u64,
    imm: usize,
) {
    let (instruction, rd) = rand_write_register_or_imm(
        tester,
        rs1.to_le_bytes(),
        (imm as u64).to_le_bytes(),
        Some(imm),
        BaseAluWImmOpcode::ADDIW.global_opcode().as_usize(),
        rng,
    );
    tester.execute(executor, preflight, &instruction);

    let signed_imm = ((imm as u32) << 20) as i32 >> 20;
    let expected = (rs1 as u32).wrapping_add(signed_imm as u32) as i32 as i64 as u64;
    assert_eq!(
        expected.to_le_bytes().map(F::from_u8),
        tester.read_bytes::<REGISTER_NUM_LIMBS>(1, rd)
    );
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//////////////////////////////////////////////////////////////////////////////////////

#[test]
fn rand_addi_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    for _ in 0..100 {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            None,
            None,
        );
    }

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn addiw_boundaries_and_sign_extension() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_w_harness(&tester);

    for (rs1, imm) in [
        (0x0000_0000_0000_0000u64, 0x00_0000usize),
        (0x0000_0000_0000_0000, 0x00_07ff),
        (0x0000_0000_0000_0000, 0x00_fff800),
        (0x0000_0000_0000_0000, 0x00_ffffff),
        (0x0000_0000_7fff_ffff, 0x00_000001),
    ] {
        set_and_execute_w(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            rs1,
            imm,
        );
    }

    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .expect("verification failed");
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_addiw_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_w_harness(&tester);

    for (rs1, imm) in [
        (0x0000_0000_0000_0000u64, 0x00_0000usize),
        (0x0000_0000_0000_0000, 0x00_07ff),
        (0x0000_0000_0000_0000, 0x00_fff800),
        (0x0000_0000_0000_0000, 0x00_ffffff),
        (0x0000_0000_7fff_ffff, 0x00_000001),
    ] {
        set_and_execute_w(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            rs1,
            imm,
        );
    }

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .expect("verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//////////////////////////////////////////////////////////////////////////////////////

#[test]
fn addi_rs1_memory_binding_negative_test() {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        Some([0; REGISTER_NUM_LIMBS]),
        Some([0; REGISTER_NUM_LIMBS]),
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
