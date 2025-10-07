use std::borrow::BorrowMut;

#[cfg(feature = "cuda")]
use openvm_circuit::arch::{
    testing::{GpuChipTestBuilder, GpuTestChipHarness},
    EmptyAdapterCoreLayout,
};
use openvm_circuit::arch::{
    testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
    Arena, PreflightExecutor,
};
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    utils::isize_to_field,
    LocalOpcode,
};
use openvm_native_compiler::NativeBranchEqualOpcode;
use openvm_rv32im_circuit::{
    adapters::RV_B_TYPE_IMM_BITS, BranchEqualCoreAir, BranchEqualCoreCols,
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
    verifier::VerificationError,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;

#[cfg(feature = "cuda")]
use super::NativeBranchEqChipGpu;
#[cfg(feature = "cuda")]
use crate::{adapters::BranchNativeAdapterRecord, branch_eq::NativeBranchEqualCoreRecord};
use crate::{
    adapters::{BranchNativeAdapterAir, BranchNativeAdapterExecutor, BranchNativeAdapterFiller},
    branch_eq::{
        run_eq, NativeBranchEqAir, NativeBranchEqChip, NativeBranchEqExecutor,
        NativeBranchEqualFiller,
    },
    test_utils::write_native_or_imm,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
const ABS_MAX_IMM: i32 = 1 << (RV_B_TYPE_IMM_BITS - 1);
type Harness = TestChipHarness<F, NativeBranchEqExecutor, NativeBranchEqAir, NativeBranchEqChip<F>>;

fn create_test_chip(tester: &mut VmChipTestBuilder<F>) -> Harness {
    let air = NativeBranchEqAir::new(
        BranchNativeAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        BranchEqualCoreAir::new(NativeBranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
    );
    let executor = NativeBranchEqExecutor::new(
        BranchNativeAdapterExecutor,
        NativeBranchEqualOpcode::CLASS_OFFSET,
        DEFAULT_PC_STEP,
    );
    let chip = NativeBranchEqChip::<F>::new(
        NativeBranchEqualFiller::new(BranchNativeAdapterFiller),
        tester.memory_helper(),
    );

    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
fn create_test_harness(
    tester: &GpuChipTestBuilder,
) -> GpuTestChipHarness<
    F,
    NativeBranchEqExecutor,
    NativeBranchEqAir,
    NativeBranchEqChipGpu,
    NativeBranchEqChip<F>,
> {
    let adapter_air =
        BranchNativeAdapterAir::new(tester.execution_bridge(), tester.memory_bridge());
    let core_air = BranchEqualCoreAir::new(NativeBranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP);
    let air = NativeBranchEqAir::new(adapter_air, core_air);

    let adapter_step = BranchNativeAdapterExecutor::new();
    let executor = NativeBranchEqExecutor::new(
        adapter_step,
        NativeBranchEqualOpcode::CLASS_OFFSET,
        DEFAULT_PC_STEP,
    );

    let core_filler = NativeBranchEqualFiller::new(BranchNativeAdapterFiller);

    let cpu_chip = NativeBranchEqChip::new(core_filler, tester.dummy_memory_helper());
    let gpu_chip = NativeBranchEqChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<E, RA>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: NativeBranchEqualOpcode,
    a: Option<F>,
    b: Option<F>,
    imm: Option<i32>,
) where
    E: PreflightExecutor<F, RA>,
    RA: Arena,
{
    let a_val = a.unwrap_or(rng.gen());
    let b_val = b.unwrap_or(if rng.gen_bool(0.5) { a_val } else { rng.gen() });
    let imm = imm.unwrap_or(rng.gen_range((-ABS_MAX_IMM)..ABS_MAX_IMM));
    let (a, a_as) = write_native_or_imm(tester, rng, a_val, None);
    let (b, b_as) = write_native_or_imm(tester, rng, b_val, None);
    let initial_pc = rng.gen_range(imm.unsigned_abs()..(1 << (PC_BITS - 1)) - imm.unsigned_abs());

    tester.execute_with_pc(
        executor,
        arena,
        &Instruction::new(
            opcode.global_opcode(),
            a,
            b,
            isize_to_field::<F>(imm as isize),
            F::from_canonical_usize(a_as),
            F::from_canonical_usize(b_as),
            F::ZERO,
            F::ZERO,
        ),
        initial_pc,
    );

    let cmp_result = run_eq(opcode.0 == BranchEqualOpcode::BEQ, a_val, b_val).0;
    let to_pc = tester.execution_final_state().pc.as_canonical_u32() as i32;
    let pc_inc = if cmp_result {
        imm
    } else {
        DEFAULT_PC_STEP as i32
    };

    assert_eq!(to_pc, initial_pc as i32 + pc_inc);
}

fn rand_set_and_execute<E, RA>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    opcode: BranchEqualOpcode,
    num_ops: usize,
) where
    E: PreflightExecutor<F, RA>,
    RA: Arena,
{
    let opcode = NativeBranchEqualOpcode(opcode);
    let mut rng = create_seeded_rng();
    for _ in 0..num_ops {
        set_and_execute(tester, executor, arena, &mut rng, opcode, None, None, None);
    }
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
    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip(&mut tester);
    rand_set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        opcode,
        num_ops,
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test_case(BranchEqualOpcode::BEQ, 100)]
#[test_case(BranchEqualOpcode::BNE, 100)]
fn test_cuda_rand_rv32_branch_eq_test(opcode: BranchEqualOpcode, num_ops: usize) {
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_test_harness(&tester);
    rand_set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.dense_arena,
        opcode,
        num_ops,
    );

    type Record<'a> = (
        &'a mut BranchNativeAdapterRecord<F>,
        &'a mut NativeBranchEqualCoreRecord<F>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, BranchNativeAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
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
    a: F,
    b: F,
    prank_cmp_result: Option<bool>,
    prank_diff_inv_marker: Option<F>,
    error: VerificationError,
) {
    let imm = 16i32;
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        NativeBranchEqualOpcode(opcode),
        Some(a),
        Some(b),
        Some(imm),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut BranchEqualCoreCols<F, 1> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        if let Some(cmp_result) = prank_cmp_result {
            cols.cmp_result = F::from_bool(cmp_result);
        }
        if let Some(diff_inv_marker) = prank_diff_inv_marker {
            cols.diff_inv_marker = [diff_inv_marker];
        }
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester.simple_test_with_expected_error(error);
}

#[test]
fn rv32_beq_wrong_cmp_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BEQ,
        F::from_canonical_u32(7 << 16),
        F::from_canonical_u32(7 << 24),
        Some(true),
        None,
        VerificationError::OodEvaluationMismatch,
    );

    run_negative_branch_eq_test(
        BranchEqualOpcode::BEQ,
        F::from_canonical_u32(7 << 16),
        F::from_canonical_u32(7 << 16),
        Some(false),
        None,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn rv32_beq_zero_inv_marker_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BEQ,
        F::from_canonical_u32(7 << 16),
        F::from_canonical_u32(7 << 24),
        Some(true),
        Some(F::ZERO),
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn rv32_beq_invalid_inv_marker_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BEQ,
        F::from_canonical_u32(7 << 16),
        F::from_canonical_u32(7 << 24),
        Some(false),
        Some(F::from_canonical_u32(1 << 16)),
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn rv32_bne_wrong_cmp_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BNE,
        F::from_canonical_u32(7 << 16),
        F::from_canonical_u32(7 << 24),
        Some(false),
        None,
        VerificationError::OodEvaluationMismatch,
    );

    run_negative_branch_eq_test(
        BranchEqualOpcode::BNE,
        F::from_canonical_u32(7 << 16),
        F::from_canonical_u32(7 << 16),
        Some(true),
        None,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn rv32_bne_zero_inv_marker_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BNE,
        F::from_canonical_u32(7 << 16),
        F::from_canonical_u32(7 << 24),
        Some(false),
        Some(F::ZERO),
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn rv32_bne_invalid_inv_marker_negative_test() {
    run_negative_branch_eq_test(
        BranchEqualOpcode::BNE,
        F::from_canonical_u32(7 << 16),
        F::from_canonical_u32(7 << 24),
        Some(true),
        Some(F::from_canonical_u32(1 << 16)),
        VerificationError::OodEvaluationMismatch,
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
    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip(&mut tester);

    let x = F::from_canonical_u32(u32::from_le_bytes([19, 4, 179, 60]));
    let y = F::from_canonical_u32(u32::from_le_bytes([19, 32, 180, 60]));
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        NativeBranchEqualOpcode(BranchEqualOpcode::BEQ),
        Some(x),
        Some(y),
        Some(8),
    );

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        NativeBranchEqualOpcode(BranchEqualOpcode::BNE),
        Some(x),
        Some(y),
        Some(8),
    );
}

#[test]
fn run_eq_sanity_test() {
    let x = F::from_canonical_u32(u32::from_le_bytes([19, 4, 17, 60]));
    let (cmp_result, diff_val) = run_eq(true, x, x);
    assert!(cmp_result);
    assert_eq!(diff_val, F::ZERO);

    let (cmp_result, diff_val) = run_eq(false, x, x);
    assert!(!cmp_result);
    assert_eq!(diff_val, F::ZERO);
}

#[test]
fn run_ne_sanity_test() {
    let x = F::from_canonical_u32(u32::from_le_bytes([19, 4, 17, 60]));
    let y = F::from_canonical_u32(u32::from_le_bytes([19, 32, 18, 60]));
    let (cmp_result, diff_val) = run_eq(true, x, y);
    assert!(!cmp_result);
    assert_eq!(diff_val * (x - y), F::ONE);

    let (cmp_result, diff_val) = run_eq(false, x, y);
    assert!(cmp_result);
    assert_eq!(diff_val * (x - y), F::ONE);
}
