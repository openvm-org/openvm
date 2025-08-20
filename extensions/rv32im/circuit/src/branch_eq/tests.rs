use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        testing::{memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::BranchEqualOpcode;
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv32BranchAdapterRecord, BranchEqualCoreRecord, Rv32BranchEqualChipGpu},
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::{core::run_eq, BranchEqualCoreCols, Rv32BranchEqualChip};
use crate::{
    adapters::{
        Rv32BranchAdapterAir, Rv32BranchAdapterExecutor, Rv32BranchAdapterFiller,
        RV32_REGISTER_NUM_LIMBS, RV_B_TYPE_IMM_BITS,
    },
    branch_eq::fast_run_eq,
    test_utils::get_verification_error,
    BranchEqualCoreAir, BranchEqualFiller, Rv32BranchEqualAir, Rv32BranchEqualExecutor,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
const ABS_MAX_IMM: i32 = 1 << (RV_B_TYPE_IMM_BITS - 1);
type Harness =
    TestChipHarness<F, Rv32BranchEqualExecutor, Rv32BranchEqualAir, Rv32BranchEqualChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv32BranchEqualAir,
    Rv32BranchEqualExecutor,
    Rv32BranchEqualChip<F>,
) {
    let air = Rv32BranchEqualAir::new(
        Rv32BranchAdapterAir::new(execution_bridge, memory_bridge),
        BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
    );
    let executor = Rv32BranchEqualExecutor::new(
        Rv32BranchAdapterExecutor,
        BranchEqualOpcode::CLASS_OFFSET,
        DEFAULT_PC_STEP,
    );
    let chip = Rv32BranchEqualChip::new(
        BranchEqualFiller::new(
            Rv32BranchAdapterFiller,
            BranchEqualOpcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(tester: &mut VmChipTestBuilder<F>) -> Harness {
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: BranchEqualOpcode,
    a: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    b: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    imm: Option<i32>,
) {
    let a = a.unwrap_or(array::from_fn(|_| rng.gen_range(0..=u8::MAX)));
    let b = b.unwrap_or(if rng.gen_bool(0.5) {
        a
    } else {
        array::from_fn(|_| rng.gen_range(0..=u8::MAX))
    });

    let imm = imm.unwrap_or(rng.gen_range((-ABS_MAX_IMM)..ABS_MAX_IMM));
    let rs1 = gen_pointer(rng, 4);
    let rs2 = gen_pointer(rng, 4);
    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, a.map(F::from_canonical_u8));
    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, b.map(F::from_canonical_u8));

    let initial_pc = rng.gen_range(imm.unsigned_abs()..(1 << (PC_BITS - 1)));
    tester.execute_with_pc(
        executor,
        arena,
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

    let cmp_result = fast_run_eq(opcode, &a, &b);
    let from_pc = tester.last_from_pc().as_canonical_u32() as i32;
    let to_pc = tester.last_to_pc().as_canonical_u32() as i32;
    let pc_inc = if cmp_result { imm } else { 4 };

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
fn rand_rv32_branch_eq_test(opcode: BranchEqualOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&mut tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
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
    a: [u8; RV32_REGISTER_NUM_LIMBS],
    b: [u8; RV32_REGISTER_NUM_LIMBS],
    prank_cmp_result: Option<bool>,
    prank_diff_inv_marker: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    interaction_error: bool,
) {
    let imm = 16i32;
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(a),
        Some(b),
        Some(imm),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut BranchEqualCoreCols<F, RV32_REGISTER_NUM_LIMBS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        if let Some(cmp_result) = prank_cmp_result {
            cols.cmp_result = F::from_bool(cmp_result);
        }
        if let Some(diff_inv_marker) = prank_diff_inv_marker {
            cols.diff_inv_marker = diff_inv_marker.map(F::from_canonical_u32);
        }
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
}

#[test]
fn rv32_beq_wrong_cmp_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BEQ,
        [0, 0, 7, 0],
        [0, 0, 0, 7],
        Some(true),
        None,
        false,
    );

    run_negative_branch_eq_test(
        BranchEqualOpcode::BEQ,
        [0, 0, 7, 0],
        [0, 0, 7, 0],
        Some(false),
        None,
        false,
    );
}

#[test]
fn rv32_beq_zero_inv_marker_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BEQ,
        [0, 0, 7, 0],
        [0, 0, 0, 7],
        Some(true),
        Some([0, 0, 0, 0]),
        false,
    );
}

#[test]
fn rv32_beq_invalid_inv_marker_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BEQ,
        [0, 0, 7, 0],
        [0, 0, 7, 0],
        Some(false),
        Some([0, 0, 1, 0]),
        false,
    );
}

#[test]
fn rv32_bne_wrong_cmp_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BNE,
        [0, 0, 7, 0],
        [0, 0, 0, 7],
        Some(false),
        None,
        false,
    );

    run_negative_branch_eq_test(
        BranchEqualOpcode::BNE,
        [0, 0, 7, 0],
        [0, 0, 7, 0],
        Some(true),
        None,
        false,
    );
}

#[test]
fn rv32_bne_zero_inv_marker_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BNE,
        [0, 0, 7, 0],
        [0, 0, 0, 7],
        Some(false),
        Some([0, 0, 0, 0]),
        false,
    );
}

#[test]
fn rv32_bne_invalid_inv_marker_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BNE,
        [0, 0, 7, 0],
        [0, 0, 7, 0],
        Some(true),
        Some([0, 0, 1, 0]),
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

    let x = [19, 4, 179, 60];
    let y = [19, 32, 180, 60];
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        BranchEqualOpcode::BEQ,
        Some(x),
        Some(y),
        Some(8),
    );

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        BranchEqualOpcode::BNE,
        Some(x),
        Some(y),
        Some(8),
    );
}

#[test]
fn run_eq_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [19, 4, 17, 60];
    let (cmp_result, _, diff_val) = run_eq::<F, RV32_REGISTER_NUM_LIMBS>(true, &x, &x);
    assert!(cmp_result);
    assert_eq!(diff_val, F::ZERO);

    let (cmp_result, _, diff_val) = run_eq::<F, RV32_REGISTER_NUM_LIMBS>(false, &x, &x);
    assert!(!cmp_result);
    assert_eq!(diff_val, F::ZERO);
}

#[test]
fn run_ne_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [19, 4, 17, 60];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [19, 32, 18, 60];
    let (cmp_result, diff_idx, diff_val) = run_eq::<F, RV32_REGISTER_NUM_LIMBS>(true, &x, &y);
    assert!(!cmp_result);
    assert_eq!(
        diff_val * (F::from_canonical_u8(x[diff_idx]) - F::from_canonical_u8(y[diff_idx])),
        F::ONE
    );

    let (cmp_result, diff_idx, diff_val) = run_eq::<F, RV32_REGISTER_NUM_LIMBS>(false, &x, &y);
    assert!(cmp_result);
    assert_eq!(
        diff_val * (F::from_canonical_u8(x[diff_idx]) - F::from_canonical_u8(y[diff_idx])),
        F::ONE
    );
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv32BranchEqualExecutor,
    Rv32BranchEqualAir,
    Rv32BranchEqualChipGpu,
    Rv32BranchEqualChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv32BranchEqualChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
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
            &mut harness.dense_arena,
            &mut rng,
            opcode,
            None,
            None,
            None,
        );
    }

    type Record<'a> = (
        &'a mut Rv32BranchAdapterRecord,
        &'a mut BranchEqualCoreRecord<RV32_REGISTER_NUM_LIMBS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32BranchAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
