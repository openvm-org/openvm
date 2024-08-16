use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use rand::{rngs::StdRng, Rng};

use super::LongAdditionChip;

fn generate_long_number<const ARG_SIZE: usize, const LIMB_SIZE: usize>(
    rng: &mut StdRng,
) -> Vec<u32> {
    assert!(ARG_SIZE % LIMB_SIZE == 0);
    (0..ARG_SIZE / LIMB_SIZE)
        .map(|_| rng.gen_range(0..1 << LIMB_SIZE))
        .collect()
}
fn generate_long_add_program<const ARG_SIZE: usize, const LIMB_SIZE: usize>(
    chip: &mut LongAdditionChip<ARG_SIZE, LIMB_SIZE>,
    len_ops: usize,
) {
    let mut rng = create_seeded_rng();
    let operands = (0..len_ops)
        .map(|_| {
            (
                generate_long_number::<ARG_SIZE, LIMB_SIZE>(&mut rng),
                generate_long_number::<ARG_SIZE, LIMB_SIZE>(&mut rng),
            )
        })
        .collect();
    chip.request(operands);
}

#[test]
fn add_long_rand_air_test() {
    let len_ops: usize = 15;
    let bus_index: usize = 0;
    let mut chip = LongAdditionChip::<256, 16>::new(bus_index);

    generate_long_add_program(&mut chip, len_ops);

    let trace = chip.generate_trace::<BabyBear>();
    let range_trace = chip.range_checker_chip.generate_trace::<BabyBear>();

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
fn setup_bad_long_addition_test(
    x1: u32,
    y1: u32,
    sum1: u32,
    carry1: u32,
    pos: usize,
) -> (LongAdditionChip<256, 16>, RowMajorMatrix<BabyBear>) {
    let bus_index: usize = 0;
    let mut chip = LongAdditionChip::<256, 16>::new(bus_index);

    let mut x = vec![0u32; 16];
    let mut y = vec![0u32; 16];
    let mut sum = vec![0u32; 16];
    let mut carry = vec![0u32; 16];

    x[pos] = x1;
    y[pos] = y1;
    sum[pos] = sum1;
    carry[pos] = carry1;

    chip.request(vec![(x.clone(), y.clone())]);

    chip.generate_trace::<BabyBear>();
    let trace = [&x, &y, &sum, &carry]
        .into_iter()
        .flatten()
        .map(|&x| BabyBear::from_canonical_u32(x))
        .collect::<Vec<_>>();
    let trace = RowMajorMatrix::new(trace, 64);

    (chip, trace)
}

fn run_bad_long_addition_test(
    chip: &LongAdditionChip<256, 16>,
    trace: RowMajorMatrix<BabyBear>,
    expected_error: VerificationError,
) {
    let range_trace = chip.range_checker_chip.generate_trace::<BabyBear>();

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
fn add_long_wrong_carry_air_test() {
    let (chip, trace) = setup_bad_long_addition_test(1, 1, 3, 1, 1);
    run_bad_long_addition_test(&chip, trace, VerificationError::OodEvaluationMismatch);
}

#[test]
fn add_long_out_of_range_air_test() {
    let (chip, trace) = setup_bad_long_addition_test(65_000, 65_000, 130_000, 0, 1);
    run_bad_long_addition_test(&chip, trace, VerificationError::NonZeroCumulativeSum);
}

#[test]
fn add_long_wrong_addition_air_test() {
    let (chip, trace) = setup_bad_long_addition_test(65_000, 65_000, 130_000 - (1 << 16), 0, 1);
    run_bad_long_addition_test(&chip, trace, VerificationError::OodEvaluationMismatch);
}

// We NEED to check that the carry is 0 or 1
#[test]
fn add_long_invalid_carry_air_test() {
    let bad_carry = BabyBear::from_canonical_u32(1 << 16)
        .inverse()
        .as_canonical_u32();
    let bus_index: usize = 0;
    let chip = LongAdditionChip::<256, 16>::new(bus_index);

    let mut x = vec![0u32; 16];
    let mut y = vec![0u32; 16];
    let mut sum = vec![0u32; 16];
    let mut carry = vec![0u32; 16];

    x[15] = 1;
    y[15] = 1;
    sum[15] = 1;
    carry[15] = bad_carry;

    for z in &sum {
        chip.range_checker_chip.add_count(*z);
    }

    let trace = [&x, &y, &sum, &carry]
        .into_iter()
        .flatten()
        .map(|&x| BabyBear::from_canonical_u32(x))
        .collect::<Vec<_>>();
    let trace = RowMajorMatrix::new(trace, 64);

    run_bad_long_addition_test(&chip, trace, VerificationError::OodEvaluationMismatch);
}
