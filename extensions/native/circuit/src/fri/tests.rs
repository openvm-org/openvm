use std::borrow::BorrowMut;

use itertools::Itertools;
#[cfg(feature = "cuda")]
use openvm_circuit::arch::testing::{GpuChipTestBuilder, GpuTestChipHarness};
use openvm_circuit::arch::{
    testing::{memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder},
    Arena, PreflightExecutor,
};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_native_compiler::{conversion::AS, FriOpcode::FRI_REDUCED_OPENING};
use openvm_stark_backend::{
    p3_field::{Field, FieldAlgebra},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
#[cfg(feature = "cuda")]
use test_case::test_case;

use super::{
    super::field_extension::FieldExtension, elem_to_ext, FriReducedOpeningAir,
    FriReducedOpeningChip, FriReducedOpeningExecutor, EXT_DEG,
};
#[cfg(feature = "cuda")]
use crate::fri::{FriReducedOpeningChipGpu, FriReducedOpeningRecordMut};
use crate::{
    fri::{FriReducedOpeningFiller, WorkloadCols, OVERALL_WIDTH, WL_WIDTH},
    write_native_array,
};

const MAX_INS_CAPACITY: usize = 1024;
type F = BabyBear;
type Harness =
    TestChipHarness<F, FriReducedOpeningExecutor, FriReducedOpeningAir, FriReducedOpeningChip<F>>;

fn create_test_chip(tester: &VmChipTestBuilder<F>) -> Harness {
    let air = FriReducedOpeningAir::new(tester.execution_bridge(), tester.memory_bridge());
    let step = FriReducedOpeningExecutor::new();
    let chip = FriReducedOpeningChip::new(FriReducedOpeningFiller, tester.memory_helper());

    Harness::with_capacity(step, air, chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
fn create_test_harness(
    tester: &GpuChipTestBuilder,
) -> GpuTestChipHarness<
    F,
    FriReducedOpeningExecutor,
    FriReducedOpeningAir,
    FriReducedOpeningChipGpu,
    FriReducedOpeningChip<F>,
> {
    let air = FriReducedOpeningAir::new(tester.execution_bridge(), tester.memory_bridge());
    let executor = FriReducedOpeningExecutor;

    let cpu_chip =
        FriReducedOpeningChip::new(FriReducedOpeningFiller, tester.dummy_memory_helper());
    let gpu_chip =
        FriReducedOpeningChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

fn compute_fri_mat_opening<F: Field>(
    alpha: [F; EXT_DEG],
    a: &[F],
    b: &[[F; EXT_DEG]],
) -> [F; EXT_DEG] {
    let mut alpha_pow: [F; EXT_DEG] = elem_to_ext(F::ONE);
    let mut result = [F::ZERO; EXT_DEG];
    for (&a, &b) in a.iter().zip_eq(b) {
        result = FieldExtension::add(
            result,
            FieldExtension::multiply(FieldExtension::subtract(b, elem_to_ext(a)), alpha_pow),
        );
        alpha_pow = FieldExtension::multiply(alpha, alpha_pow);
    }
    result
}

fn set_and_execute<E, RA>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
) where
    E: PreflightExecutor<F, RA>,
    RA: Arena,
{
    let len = rng.gen_range(1..=28);
    let a_ptr = gen_pointer(rng, len);
    let b_ptr = gen_pointer(rng, len);
    let a_ptr_ptr =
        write_native_array::<F, 1>(tester, rng, Some([F::from_canonical_usize(a_ptr)])).1;
    let b_ptr_ptr =
        write_native_array::<F, 1>(tester, rng, Some([F::from_canonical_usize(b_ptr)])).1;

    let len_ptr = write_native_array::<F, 1>(tester, rng, Some([F::from_canonical_usize(len)])).1;
    let (alpha, alpha_ptr) = write_native_array::<F, EXT_DEG>(tester, rng, None);
    let out_ptr = gen_pointer(rng, EXT_DEG);
    let is_init = true;
    let is_init_ptr = write_native_array::<F, 1>(tester, rng, Some([F::from_bool(is_init)])).1;

    let mut vec_a = Vec::with_capacity(len);
    let mut vec_b = Vec::with_capacity(len);
    for i in 0..len {
        let a = rng.gen();
        let b: [F; EXT_DEG] = std::array::from_fn(|_| rng.gen());
        vec_a.push(a);
        vec_b.push(b);
        if !is_init {
            tester.streams_mut().hint_space[0].push(a);
        } else {
            tester.write(AS::Native as usize, a_ptr + i, [a]);
        }
        tester.write(AS::Native as usize, b_ptr + (EXT_DEG * i), b);
    }

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(
            FRI_REDUCED_OPENING.global_opcode(),
            [
                a_ptr_ptr,
                b_ptr_ptr,
                len_ptr,
                alpha_ptr,
                out_ptr,
                0, // hint id, will just use 0 for testing
                is_init_ptr,
            ],
        ),
    );

    let expected_result = compute_fri_mat_opening(alpha, &vec_a, &vec_b);
    assert_eq!(expected_result, tester.read(AS::Native as usize, out_ptr));

    for (i, ai) in vec_a.iter().enumerate() {
        let [found] = tester.read(AS::Native as usize, a_ptr + i);
        assert_eq!(*ai, found);
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn fri_mat_opening_air_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip(&tester);

    let num_ops = 28; // non-power-of-2 to also test padding
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
        );
    }

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test_case(28)]
fn test_fri_tracegen(num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_test_harness(&tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
        );
    }

    harness
        .dense_arena
        .get_record_seeker::<FriReducedOpeningRecordMut<F>, _>()
        .transfer_to_matrix_arena(&mut harness.matrix_arena);

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

#[test]
fn run_negative_fri_mat_opening_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
    );

    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut WorkloadCols<F> = values[..WL_WIDTH].borrow_mut();

        cols.prefix.a_or_is_first = F::from_canonical_u32(42);

        *trace = RowMajorMatrix::new(values, OVERALL_WIDTH);
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester.simple_test_with_expected_error(VerificationError::OodEvaluationMismatch);
}
