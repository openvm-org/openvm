use std::borrow::BorrowMut;

use itertools::Itertools;
use openvm_circuit::arch::testing::{memory::gen_pointer, TestChipHarness, VmChipTestBuilder};
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

use super::{
    super::field_extension::FieldExtension, compute_polynomial_evaluation, elem_to_ext,
    scalar_polynomial_evaluation, FriReducedOpeningAir, FriReducedOpeningChip,
    FriReducedOpeningExecutor, EXT_DEG,
};
use crate::{
    fri::{WorkloadCols, OVERALL_WIDTH, WL_WIDTH},
    write_native_array, FriReducedOpeningFiller,
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

fn set_and_execute(tester: &mut VmChipTestBuilder<F>, harness: &mut Harness, rng: &mut StdRng) {
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
            tester.streams.hint_space[0].push(a);
        } else {
            tester.write(AS::Native as usize, a_ptr + i, [a]);
        }
        tester.write(AS::Native as usize, b_ptr + (EXT_DEG * i), b);
    }

    tester.execute(
        harness,
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
        set_and_execute(&mut tester, &mut harness, &mut rng);
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

#[test]
fn run_negative_fri_mat_opening_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip(&tester);

    set_and_execute(&mut tester, &mut harness, &mut rng);

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

///////////////////////////////////////////////////////////////////////////////////////
/// VECTORIZED POLYNOMIAL EVALUATION TESTS
///
/// Test that vectorized and scalar polynomial evaluation produce identical results
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_scalar_vs_vectorized_polynomial_evaluation() {
    use openvm_stark_backend::p3_field::{
        extension::BinomialExtensionField, FieldExtensionAlgebra,
    };

    type EF = BinomialExtensionField<F, EXT_DEG>;
    let mut rng = create_seeded_rng();

    // Test different sizes to verify correctness
    let test_sizes = [1, 3, 4, 7, 8, 15, 16];

    for &size in &test_sizes {
        let alpha: EF = EF::from_base_fn(|_| rng.gen::<F>());

        // Generate random test data
        let mut as_and_bs = Vec::with_capacity(size);
        for _ in 0..size {
            let a: F = rng.gen();
            let b: EF = EF::from_base_fn(|_| rng.gen::<F>());
            as_and_bs.push((a, b));
        }

        // Compute using both methods
        let scalar_result = scalar_polynomial_evaluation(&as_and_bs, alpha);
        let vectorized_result = compute_polynomial_evaluation(&as_and_bs, alpha);

        assert_eq!(
            scalar_result, vectorized_result,
            "Scalar and vectorized results differ for size {}: scalar={:?}, vectorized={:?}",
            size, scalar_result, vectorized_result
        );
    }
}

#[test]
fn test_polynomial_evaluation_edge_cases() {
    use openvm_stark_backend::p3_field::{
        extension::BinomialExtensionField, FieldExtensionAlgebra,
    };

    type EF = BinomialExtensionField<F, EXT_DEG>;

    // Test with alpha = 0
    let alpha_zero = EF::ZERO;
    let as_and_bs = vec![(F::ONE, EF::from_base(F::TWO)); 5];
    let result_zero = compute_polynomial_evaluation(&as_and_bs, alpha_zero);
    let expected_zero = EF::from_base(F::ONE); // Should be b_0 - a_0 = 2 - 1 = 1
    assert_eq!(result_zero, expected_zero, "Failed for alpha = 0");

    // Test with alpha = 1
    let alpha_one = EF::ONE;
    let as_and_bs = vec![(F::ONE, EF::from_base(F::TWO)); 3];
    let result_one = compute_polynomial_evaluation(&as_and_bs, alpha_one);
    let expected_one = EF::from_base(F::from_canonical_u32(3)); // 1 + 1 + 1 = 3
    assert_eq!(result_one, expected_one, "Failed for alpha = 1");

    // Test empty vector should return 0
    let empty_vec: Vec<(F, EF)> = vec![];
    let result_empty = compute_polynomial_evaluation(&empty_vec, EF::from_base(F::TWO));
    assert_eq!(result_empty, EF::ZERO, "Failed for empty vector");

    // Test single element
    let single = vec![(
        F::from_canonical_u32(3),
        EF::from_base(F::from_canonical_u32(7)),
    )];
    let alpha: EF = EF::from_base(F::from_canonical_u32(5));
    let result_single = compute_polynomial_evaluation(&single, alpha);
    let expected_single = EF::from_base(F::from_canonical_u32(4)); // 7 - 3 = 4
    assert_eq!(result_single, expected_single, "Failed for single element");
}

#[test]
fn test_polynomial_evaluation_consistency_with_reference() {
    use openvm_stark_backend::p3_field::{
        extension::BinomialExtensionField, FieldExtensionAlgebra,
    };

    type EF = BinomialExtensionField<F, EXT_DEG>;
    let mut rng = create_seeded_rng();

    // Test against the reference implementation from the test file
    for size in [5, 10, 20, 50] {
        let alpha: [F; EXT_DEG] = std::array::from_fn(|_| rng.gen());
        let alpha_ef = EF::from_base_slice(&alpha);

        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut as_and_bs = Vec::new();

        for _ in 0..size {
            let a_val: F = rng.gen();
            let b_val: [F; EXT_DEG] = std::array::from_fn(|_| rng.gen());
            let b_ef = EF::from_base_slice(&b_val);

            a.push(a_val);
            b.push(b_val);
            as_and_bs.push((a_val, b_ef));
        }

        // Reference implementation from compute_fri_mat_opening
        let reference_result = compute_fri_mat_opening(alpha, &a, &b);
        let our_result = compute_polynomial_evaluation(&as_and_bs, alpha_ef);

        assert_eq!(
            reference_result,
            our_result.as_base_slice(),
            "Results differ from reference implementation for size {}",
            size
        );
    }
}
