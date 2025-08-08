use std::borrow::BorrowMut;

use openvm_circuit::arch::testing::{memory::gen_pointer, TestChipHarness, VmChipTestBuilder};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_native_compiler::{conversion::AS, FieldArithmeticOpcode};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
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

use super::{
    FieldArithmeticChip, FieldArithmeticCoreAir, FieldArithmeticCoreCols, FieldArithmeticExecutor,
};
use crate::{
    adapters::{AluNativeAdapterAir, AluNativeAdapterExecutor, AluNativeAdapterFiller},
    field_arithmetic::{run_field_arithmetic, FieldArithmeticAir},
    test_utils::write_native_or_imm,
    FieldArithmeticCoreFiller,
};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness =
    TestChipHarness<F, FieldArithmeticExecutor, FieldArithmeticAir, FieldArithmeticChip<F>>;

fn create_test_chip(tester: &VmChipTestBuilder<F>) -> Harness {
    let air = FieldArithmeticAir::new(
        AluNativeAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        FieldArithmeticCoreAir::new(),
    );
    let executor = FieldArithmeticExecutor::new(AluNativeAdapterExecutor::new());
    let chip = FieldArithmeticChip::<F>::new(
        FieldArithmeticCoreFiller::new(AluNativeAdapterFiller),
        tester.memory_helper(),
    );

    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    harness: &mut Harness,
    rng: &mut StdRng,
    opcode: FieldArithmeticOpcode,
    b: Option<F>,
    c: Option<F>,
) {
    let b_val = b.unwrap_or(rng.gen());
    let c_val = c.unwrap_or(if opcode == FieldArithmeticOpcode::DIV {
        // If division, make sure c is not zero
        F::from_canonical_u32(rng.gen_range(0..F::NEG_ONE.as_canonical_u32()) + 1)
    } else {
        rng.gen()
    });
    assert!(!c_val.is_zero(), "Division by zero");
    let (b, b_as) = write_native_or_imm(tester, rng, b_val, None);
    let (c, c_as) = write_native_or_imm(tester, rng, c_val, None);
    let a = gen_pointer(rng, 1);

    tester.execute(
        harness,
        &Instruction::new(
            opcode.global_opcode(),
            F::from_canonical_usize(a),
            b,
            c,
            F::from_canonical_usize(AS::Native as usize),
            F::from_canonical_usize(b_as),
            F::from_canonical_usize(c_as),
            F::ZERO,
        ),
    );

    let expected = run_field_arithmetic(opcode, b_val, c_val);
    let result = tester.read::<1>(AS::Native as usize, a)[0];
    assert_eq!(result, expected);
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////
#[test_case(FieldArithmeticOpcode::ADD, 100)]
#[test_case(FieldArithmeticOpcode::SUB, 100)]
#[test_case(FieldArithmeticOpcode::MUL, 100)]
#[test_case(FieldArithmeticOpcode::DIV, 100)]
fn new_field_arithmetic_air_test(opcode: FieldArithmeticOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip(&tester);

    for _ in 0..num_ops {
        set_and_execute(&mut tester, &mut harness, &mut rng, opcode, None, None);
    }

    set_and_execute(
        &mut tester,
        &mut harness,
        &mut rng,
        opcode,
        Some(F::ZERO),
        None,
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Default)]
struct FieldExpressionPrankVals {
    a: Option<F>,
    b: Option<F>,
    c: Option<F>,
    opcode_flags: Option<[bool; 4]>,
    divisor_inv: Option<F>,
}
#[allow(clippy::too_many_arguments)]
fn run_negative_field_arithmetic_test(
    opcode: FieldArithmeticOpcode,
    b: F,
    c: F,
    prank_vals: FieldExpressionPrankVals,
    error: VerificationError,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip(&tester);

    set_and_execute(
        &mut tester,
        &mut harness,
        &mut rng,
        opcode,
        Some(b),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut FieldArithmeticCoreCols<F> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        if let Some(a) = prank_vals.a {
            cols.a = a;
        }
        if let Some(b) = prank_vals.b {
            cols.b = b;
        }
        if let Some(c) = prank_vals.c {
            cols.c = c;
        }
        if let Some(opcode_flags) = prank_vals.opcode_flags {
            [cols.is_add, cols.is_sub, cols.is_mul, cols.is_div] = opcode_flags.map(F::from_bool);
        }
        if let Some(divisor_inv) = prank_vals.divisor_inv {
            cols.divisor_inv = divisor_inv;
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
fn field_arithmetic_negative_zero_div_test() {
    run_negative_field_arithmetic_test(
        FieldArithmeticOpcode::DIV,
        F::from_canonical_u32(111),
        F::from_canonical_u32(222),
        FieldExpressionPrankVals {
            b: Some(F::ZERO),
            ..Default::default()
        },
        VerificationError::OodEvaluationMismatch,
    );

    run_negative_field_arithmetic_test(
        FieldArithmeticOpcode::DIV,
        F::ZERO,
        F::TWO,
        FieldExpressionPrankVals {
            c: Some(F::ZERO),
            ..Default::default()
        },
        VerificationError::OodEvaluationMismatch,
    );

    run_negative_field_arithmetic_test(
        FieldArithmeticOpcode::DIV,
        F::ZERO,
        F::TWO,
        FieldExpressionPrankVals {
            c: Some(F::ZERO),
            opcode_flags: Some([false, false, true, false]),
            ..Default::default()
        },
        VerificationError::ChallengePhaseError,
    );
}

#[test]
fn field_arithmetic_negative_rand() {
    let mut rng = create_seeded_rng();
    run_negative_field_arithmetic_test(
        FieldArithmeticOpcode::DIV,
        F::from_canonical_u32(111),
        F::from_canonical_u32(222),
        FieldExpressionPrankVals {
            a: Some(rng.gen()),
            b: Some(rng.gen()),
            c: Some(rng.gen()),
            opcode_flags: Some([rng.gen(), rng.gen(), rng.gen(), rng.gen()]),
            divisor_inv: Some(rng.gen()),
        },
        VerificationError::OodEvaluationMismatch,
    );
}

#[should_panic]
#[test]
fn new_field_arithmetic_air_test_panic() {
    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip(&tester);
    tester.write(4, 0, [BabyBear::ZERO]);
    // should panic
    tester.execute(
        &mut harness,
        &Instruction::from_usize(
            FieldArithmeticOpcode::DIV.global_opcode(),
            [0, 0, 0, 4, 4, 4],
        ),
    );
}
