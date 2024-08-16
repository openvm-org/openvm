use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use rand::{rngs::StdRng, Rng};

use super::{trace::create_row_from_values, LongArithmeticChip};
use crate::cpu::OpCode;

type F = BabyBear;

const OPCODES: [OpCode; 2] = [OpCode::ADD, OpCode::SUB];

fn generate_long_number<const ARG_SIZE: usize, const LIMB_SIZE: usize>(
    rng: &mut StdRng,
) -> Vec<u32> {
    assert!(ARG_SIZE % LIMB_SIZE == 0);
    (0..ARG_SIZE / LIMB_SIZE)
        .map(|_| rng.gen_range(0..1 << LIMB_SIZE))
        .collect()
}

fn generate_long_program<const ARG_SIZE: usize, const LIMB_SIZE: usize>(
    chip: &mut LongArithmeticChip<ARG_SIZE, LIMB_SIZE>,
    len_ops: usize,
) {
    let mut rng = create_seeded_rng();
    let opcodes = (0..len_ops)
        .map(|_| OPCODES[rng.gen_range(0..OPCODES.len())])
        .collect();
    let operands = (0..len_ops)
        .map(|_| {
            (
                generate_long_number::<ARG_SIZE, LIMB_SIZE>(&mut rng),
                generate_long_number::<ARG_SIZE, LIMB_SIZE>(&mut rng),
            )
        })
        .collect();
    chip.request(opcodes, operands);
}

#[test]
fn long_add_rand_air_test() {
    let len_ops: usize = 15;
    let bus_index: usize = 0;
    let mut chip = LongArithmeticChip::<256, 16>::new(bus_index);

    generate_long_program(&mut chip, len_ops);

    let trace = chip.generate_trace::<F>();
    let range_trace = chip.range_checker_chip.generate_trace::<F>();

    run_simple_test_no_pis(
        vec![&chip.air, &chip.range_checker_chip.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

// Given a fake trace of a single addition, which mostly consists of zeroes
// except for a single position, setup a chip and provide this trace for it.
// The chip will do what it would normally do for this addition query,
// except the generated trace will be what we provide.
fn setup_bad_long_arithmetic_test(
    op: OpCode,
    x1: u32,
    y1: u32,
    sum1: u32,
    carry1: u32,
    pos: usize,
) -> (LongArithmeticChip<256, 16>, RowMajorMatrix<F>) {
    let bus_index: usize = 0;
    let mut chip = LongArithmeticChip::<256, 16>::new(bus_index);

    let mut x = vec![0u32; 16];
    let mut y = vec![0u32; 16];
    let mut sum = [0u32; 16];
    let mut carry = [0u32; 16];

    x[pos] = x1;
    y[pos] = y1;
    sum[pos] = sum1;
    carry[pos] = carry1;

    chip.request(vec![op], vec![(x.clone(), y.clone())]);

    chip.generate_trace::<F>();
    let trace = create_row_from_values::<256, 16, F>(op, &x, &y, &sum, &carry);
    let width = trace.len();
    let trace = RowMajorMatrix::new(trace, width);

    (chip, trace)
}

fn run_bad_long_arithmetic_test(
    chip: &LongArithmeticChip<256, 16>,
    trace: RowMajorMatrix<F>,
    expected_error: VerificationError,
) {
    let range_trace = chip.range_checker_chip.generate_trace::<F>();

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(
            vec![&chip.air, &chip.range_checker_chip.air],
            vec![trace, range_trace],
        ),
        Err(expected_error),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn long_add_wrong_carry_air_test() {
    let (chip, trace) = setup_bad_long_arithmetic_test(OpCode::ADD, 1, 1, 3, 1, 1);
    run_bad_long_arithmetic_test(&chip, trace, VerificationError::OodEvaluationMismatch);
}

#[test]
fn long_add_out_of_range_air_test() {
    let (chip, trace) = setup_bad_long_arithmetic_test(OpCode::ADD, 65_000, 65_000, 130_000, 0, 1);
    run_bad_long_arithmetic_test(&chip, trace, VerificationError::NonZeroCumulativeSum);
}

#[test]
fn long_add_wrong_addition_air_test() {
    let (chip, trace) =
        setup_bad_long_arithmetic_test(OpCode::ADD, 65_000, 65_000, 130_000 - (1 << 16), 0, 1);
    run_bad_long_arithmetic_test(&chip, trace, VerificationError::OodEvaluationMismatch);
}

// We NEED to check that the carry is 0 or 1
#[test]
fn long_add_invalid_carry_air_test() {
    let bad_carry = F::from_canonical_u32(1 << 16).inverse().as_canonical_u32();
    let bus_index: usize = 0;
    let chip = LongArithmeticChip::<256, 16>::new(bus_index);

    let mut x = [0u32; 16];
    let mut y = [0u32; 16];
    let mut sum = vec![0u32; 16];
    let mut carry = [0u32; 16];

    x[15] = 1;
    y[15] = 1;
    sum[15] = 1;
    carry[15] = bad_carry;

    for z in &sum {
        chip.range_checker_chip.add_count(*z);
    }

    let op = OpCode::ADD;
    let trace = create_row_from_values::<256, 16, F>(op, &x, &y, &sum, &carry);
    let width = trace.len();
    let trace = RowMajorMatrix::new(trace, width);

    run_bad_long_arithmetic_test(&chip, trace, VerificationError::OodEvaluationMismatch);
}

#[test]
fn long_sub_out_of_range_air_test() {
    let (chip, trace) =
        setup_bad_long_arithmetic_test(OpCode::SUB, 1, 2, (-F::one()).as_canonical_u32(), 0, 1);
    run_bad_long_arithmetic_test(&chip, trace, VerificationError::NonZeroCumulativeSum);
}

#[test]
fn long_sub_wrong_subtraction_air_test() {
    let (chip, trace) = setup_bad_long_arithmetic_test(OpCode::SUB, 1, 2, (1 << 16) - 1, 0, 1);
    run_bad_long_arithmetic_test(&chip, trace, VerificationError::OodEvaluationMismatch);
}

#[test]
fn long_sub_invalid_carry_air_test() {
    let bad_carry = F::from_canonical_u32(1 << 16).inverse().as_canonical_u32();
    let bus_index: usize = 0;
    let chip = LongArithmeticChip::<256, 16>::new(bus_index);

    let mut x = [0u32; 16];
    let mut y = [0u32; 16];
    let mut sum = vec![0u32; 16];
    let mut carry = [0u32; 16];

    x[15] = 1;
    y[15] = 1;
    sum[15] = 1;
    carry[15] = bad_carry;

    for z in &sum {
        chip.range_checker_chip.add_count(*z);
    }

    let op = OpCode::SUB;
    let trace = create_row_from_values::<256, 16, F>(op, &x, &y, &sum, &carry);
    let width = trace.len();
    let trace = RowMajorMatrix::new(trace, width);

    run_bad_long_arithmetic_test(&chip, trace, VerificationError::OodEvaluationMismatch);
}
