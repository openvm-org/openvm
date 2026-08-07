use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        testing::{
            memory::gen_distinct_register_pointers, TestBuilder, TestChipHarness, VmChipTestBuilder,
        },
        ExecutionBridge, BLOCK_FE_WIDTH,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, MAX_ALLOWED_PC},
    LocalOpcode,
};
use openvm_riscv_transpiler::BranchEqualOpcode;
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
use test_case::test_case;
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::BranchEqualChipGpu,
    openvm_circuit::arch::testing::{GpuChipTestBuilder, GpuTestChipHarness},
};

use super::{
    core::run_eq, trace::generate_trace_from_postflight, BranchEqualChip, BranchEqualCoreCols,
};
use crate::{
    adapters::{bytes_to_u16_block, BranchAdapterAir, REGISTER_NUM_LIMBS, RV_B_TYPE_IMM_BITS},
    branch_eq::fast_run_eq,
    test_utils::marker_bytes_to_u16_marker,
    BranchEqualAir, BranchEqualCoreAir, BranchEqualExecutor, BranchEqualFiller,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
const ABS_MAX_IMM: i32 = 1 << (RV_B_TYPE_IMM_BITS - 1);
type Harness = TestChipHarness<F, BranchEqualExecutor, BranchEqualAir, BranchEqualChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    memory_helper: SharedMemoryHelper<F>,
) -> (BranchEqualAir, BranchEqualExecutor, BranchEqualChip<F>) {
    let air = BranchEqualAir::new(
        BranchAdapterAir::new(execution_bridge, memory_bridge),
        BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
    );
    let executor = BranchEqualExecutor::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP);
    let chip = BranchEqualChip::new(BranchEqualFiller::new(DEFAULT_PC_STEP), memory_helper);
    (air, executor, chip)
}

fn create_harness(tester: &mut VmChipTestBuilder<F>) -> Harness {
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
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

#[allow(clippy::too_many_arguments)]
fn set_and_execute<E: openvm_circuit::arch::Executor<F> + Clone>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    preflight: &mut openvm_circuit::arch::testing::TestPreflight<F>,
    rng: &mut StdRng,
    opcode: BranchEqualOpcode,
    a: Option<[u8; REGISTER_NUM_LIMBS]>,
    b: Option<[u8; REGISTER_NUM_LIMBS]>,
    imm: Option<i32>,
) {
    let a = a.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let b = b.unwrap_or(if rng.random_bool(0.5) {
        a
    } else {
        array::from_fn(|_| rng.random_range(0..=u8::MAX))
    });

    // Branch offsets are DEFAULT_PC_STEP-aligned byte offsets.
    let imm = imm.unwrap_or(
        rng.random_range(
            (-ABS_MAX_IMM / DEFAULT_PC_STEP as i32)..(ABS_MAX_IMM / DEFAULT_PC_STEP as i32),
        ) * DEFAULT_PC_STEP as i32,
    );
    let [rs1, rs2] = gen_distinct_register_pointers(rng, REGISTER_NUM_LIMBS);
    tester.write_bytes::<REGISTER_NUM_LIMBS>(1, rs1, a.map(F::from_u8));
    tester.write_bytes::<REGISTER_NUM_LIMBS>(1, rs2, b.map(F::from_u8));

    // An aligned byte pc over the full 32-bit range, keeping the taken target in bounds.
    let lo = (-imm).max(0) as u32 / DEFAULT_PC_STEP;
    let hi = (MAX_ALLOWED_PC - DEFAULT_PC_STEP - imm.max(0) as u32) / DEFAULT_PC_STEP;
    let initial_pc = rng.random_range(lo..=hi) * DEFAULT_PC_STEP;
    tester.execute_with_pc(
        executor,
        preflight,
        &Instruction::from_isize(
            opcode.global_opcode(),
            rs1 as isize,
            rs2 as isize,
            imm as isize,
            1,
            1,
        ),
        initial_pc,
    );

    let cmp_result = fast_run_eq(opcode, &bytes_to_u16_block(a), &bytes_to_u16_block(b));
    let from_pc = tester.last_from_pc() as i64;
    let to_pc = tester.last_to_pc() as i64;
    let pc_inc = if cmp_result { imm as i64 } else { 4 };

    assert_eq!(to_pc, from_pc + pc_inc);
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(BranchEqualOpcode::BEQ, 100)]
#[test_case(BranchEqualOpcode::BNE, 100)]
fn rand_branch_eq_test(opcode: BranchEqualOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&mut tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            opcode,
            None,
            None,
            None,
        );
    }

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[allow(clippy::too_many_arguments)]
fn run_negative_branch_eq_test(
    opcode: BranchEqualOpcode,
    a: [u8; REGISTER_NUM_LIMBS],
    b: [u8; REGISTER_NUM_LIMBS],
    prank_cmp_result: Option<bool>,
    prank_diff_inv_marker: Option<[u32; BLOCK_FE_WIDTH]>,
    _interaction_error: bool,
) {
    let imm = 16i32;
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        opcode,
        Some(a),
        Some(b),
        Some(imm),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut BranchEqualCoreCols<F, BLOCK_FE_WIDTH> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        if let Some(cmp_result) = prank_cmp_result {
            cols.cmp_result = F::from_bool(cmp_result);
        }
        if let Some(diff_inv_marker) = prank_diff_inv_marker {
            cols.diff_inv_marker = diff_inv_marker.map(F::from_u32);
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
fn beq_wrong_cmp_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BEQ,
        [0, 0, 7, 0, 0, 0, 0, 0],
        [0, 0, 0, 7, 0, 0, 0, 0],
        Some(true),
        None,
        false,
    );

    run_negative_branch_eq_test(
        BranchEqualOpcode::BEQ,
        [0, 0, 7, 0, 0, 0, 0, 0],
        [0, 0, 7, 0, 0, 0, 0, 0],
        Some(false),
        None,
        false,
    );
}

#[test]
fn beq_zero_inv_marker_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BEQ,
        [0, 0, 7, 0, 0, 0, 0, 0],
        [0, 0, 0, 7, 0, 0, 0, 0],
        Some(true),
        Some(marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 0])),
        false,
    );
}

#[test]
fn beq_invalid_inv_marker_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BEQ,
        [0, 0, 7, 0, 0, 0, 0, 0],
        [0, 0, 7, 0, 0, 0, 0, 0],
        Some(false),
        Some(marker_bytes_to_u16_marker([0, 0, 1, 0, 0, 0, 0, 0])),
        false,
    );
}

#[test]
fn bne_wrong_cmp_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BNE,
        [0, 0, 7, 0, 0, 0, 0, 0],
        [0, 0, 0, 7, 0, 0, 0, 0],
        Some(false),
        None,
        false,
    );

    run_negative_branch_eq_test(
        BranchEqualOpcode::BNE,
        [0, 0, 7, 0, 0, 0, 0, 0],
        [0, 0, 7, 0, 0, 0, 0, 0],
        Some(true),
        None,
        false,
    );
}

#[test]
fn bne_zero_inv_marker_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BNE,
        [0, 0, 7, 0, 0, 0, 0, 0],
        [0, 0, 0, 7, 0, 0, 0, 0],
        Some(false),
        Some(marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 0])),
        false,
    );
}

#[test]
fn bne_invalid_inv_marker_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BNE,
        [0, 0, 7, 0, 0, 0, 0, 0],
        [0, 0, 7, 0, 0, 0, 0, 0],
        Some(true),
        Some(marker_bytes_to_u16_marker([0, 0, 1, 0, 0, 0, 0, 0])),
        false,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn execute_roundtrip_sanity_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&mut tester);

    let x = [19, 4, 179, 60, 201, 77, 1, 240];
    let y = [19, 32, 180, 60, 201, 77, 1, 240];
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        BranchEqualOpcode::BEQ,
        Some(x),
        Some(y),
        Some(8),
    );

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        BranchEqualOpcode::BNE,
        Some(x),
        Some(y),
        Some(8),
    );
}

#[test]
fn run_eq_sanity_test() {
    let x = bytes_to_u16_block([19, 4, 17, 60, 201, 77, 1, 240]);
    let (cmp_result, _, diff_val) = run_eq::<F, BLOCK_FE_WIDTH>(true, &x, &x);
    assert!(cmp_result);
    assert_eq!(diff_val, F::ZERO);

    let (cmp_result, _, diff_val) = run_eq::<F, BLOCK_FE_WIDTH>(false, &x, &x);
    assert!(!cmp_result);
    assert_eq!(diff_val, F::ZERO);
}

#[test]
fn run_ne_sanity_test() {
    let x = bytes_to_u16_block([19, 4, 17, 60, 201, 77, 1, 240]);
    let y = bytes_to_u16_block([19, 32, 18, 60, 201, 77, 1, 240]);
    let (cmp_result, diff_idx, diff_val) = run_eq::<F, BLOCK_FE_WIDTH>(true, &x, &y);
    assert!(!cmp_result);
    assert_eq!(
        diff_val * (F::from_u16(x[diff_idx]) - F::from_u16(y[diff_idx])),
        F::ONE
    );

    let (cmp_result, diff_idx, diff_val) = run_eq::<F, BLOCK_FE_WIDTH>(false, &x, &y);
    assert!(cmp_result);
    assert_eq!(
        diff_val * (F::from_u16(x[diff_idx]) - F::from_u16(y[diff_idx])),
        F::ONE
    );
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuHarness = GpuTestChipHarness<
    F,
    BranchEqualExecutor,
    BranchEqualAir,
    BranchEqualChipGpu,
    BranchEqualChip<F>,
>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = BranchEqualChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
        .with_trace_generators(
            generate_trace_from_postflight,
            |chip, program, transcript, plan| {
                chip.generate_proving_ctx_from_postflight(program, transcript, plan)
            },
        )
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test_case(BranchEqualOpcode::BEQ, 100)]
#[test_case(BranchEqualOpcode::BNE, 100)]
fn test_cuda_rand_beq_tracegen(opcode: BranchEqualOpcode, num_ops: usize) {
    let mut tester = GpuChipTestBuilder::default();
    let mut rng = create_seeded_rng();

    let mut harness = create_cuda_harness(&tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            opcode,
            None,
            None,
            None,
        );
    }

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
