use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use rand::{rngs::StdRng, Rng};

use super::{columns::LongMultiplicationCols, LongMultiplicationChip};
use crate::cpu::OpCode;

type F = BabyBear;

fn generate_long_number(rng: &mut StdRng, arg_size: usize, limb_size: usize) -> Vec<u32> {
    assert!(arg_size % limb_size == 0);
    (0..arg_size / limb_size)
        .map(|_| rng.gen_range(0..1 << limb_size))
        .collect()
}

fn generate_mul_program(
    chip: &mut LongMultiplicationChip,
    len_ops: usize,
    arg_size: usize,
    limb_size: usize,
) {
    let mut rng = create_seeded_rng();
    let opcodes = vec![OpCode::MUL256; len_ops];
    let operands = (0..len_ops)
        .map(|_| {
            (
                generate_long_number(&mut rng, arg_size, limb_size),
                generate_long_number(&mut rng, arg_size, limb_size),
            )
        })
        .collect();
    chip.request(opcodes, operands);
}

#[test]
fn long_mul_rand_air_test() {
    let len_ops: usize = 15;
    let arg_size: usize = 256;
    let limb_size: usize = 8;
    let bus_index: usize = 0;
    let mut chip = LongMultiplicationChip::new(bus_index, arg_size, limb_size, OpCode::MUL256);

    generate_mul_program(&mut chip, len_ops, arg_size, limb_size);

    let trace = chip.generate_trace::<F>();
    let range_trace = chip.range_checker_chip.generate_trace::<F>();

    run_simple_test_no_pis(
        vec![&chip.air, &chip.range_checker_chip.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn long_mul_large_limbs_air_test() {
    let arg_size: usize = 256;
    let limb_size: usize = 8;
    let bus_index: usize = 0;
    let mut chip = LongMultiplicationChip::new(bus_index, arg_size, limb_size, OpCode::MUL256);

    let opcodes = vec![OpCode::MUL256];
    let large_number = vec![(1u32 << limb_size) - 1; arg_size / limb_size];
    let operands = vec![(large_number.clone(), large_number.clone())];
    chip.request(opcodes, operands);

    let trace = chip.generate_trace::<F>();
    let range_trace = chip.range_checker_chip.generate_trace::<F>();

    run_simple_test_no_pis(
        vec![&chip.air, &chip.range_checker_chip.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

// Given a fake trace of a single multiplication, setup a chip and provide this trace for it.
// The chip can do what it would normally do for this multiplication query,
// except the generated trace will be what we provide. It can also communicate with
// the range checker according to the provided trace. We choose it with the last parameter.
fn setup_bad_long_arithmetic_test(
    x: Vec<u32>,
    y: Vec<u32>,
    z: Vec<u32>,
    carry: Vec<u32>,
    original_interactions: bool,
) -> (LongMultiplicationChip, RowMajorMatrix<F>) {
    let bus_index: usize = 0;
    let mut chip = LongMultiplicationChip::new(bus_index, 256, 8, OpCode::MUL256);

    if original_interactions {
        chip.request(vec![OpCode::MUL256], vec![(x.clone(), y.clone())]);
        chip.generate_trace::<F>();
    } else {
        for z in z.iter() {
            // TODO: replace with a safer range check once we have one
            chip.range_checker_chip.add_count(*z);
            chip.range_checker_chip.add_count(*z * 32u32);
        }
        for c in carry.iter() {
            chip.range_checker_chip.add_count(*c);
        }
    }
    let trace = LongMultiplicationCols {
        rcv_count: 1,
        opcode: OpCode::MUL256 as u32,
        x_limbs: x,
        y_limbs: y,
        z_limbs: z,
        carry,
    }
    .flatten()
    .into_iter()
    .map(F::from_canonical_u32)
    .collect::<Vec<F>>();
    let width = trace.len();
    let trace = RowMajorMatrix::new(trace, width);

    (chip, trace)
}

fn run_bad_long_arithmetic_test(
    chip: &LongMultiplicationChip,
    trace: RowMajorMatrix<F>,
    expected_error: VerificationError,
) {
    let range_trace = chip.range_checker_chip.generate_trace::<F>();

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    let msg = format!(
        "Expected verification to fail with {:?}, but it didn't",
        &expected_error
    );
    assert_eq!(
        run_simple_test_no_pis(
            vec![&chip.air, &chip.range_checker_chip.air],
            vec![trace, range_trace],
        ),
        Err(expected_error),
        "{}",
        msg
    );
}

#[test]
fn long_mul_bad_carry_test() {
    let (chip, trace) = setup_bad_long_arithmetic_test(
        vec![16]
            .into_iter()
            .chain(std::iter::repeat(0).take(31))
            .collect::<Vec<_>>(),
        vec![16]
            .into_iter()
            .chain(std::iter::repeat(0).take(31))
            .collect::<Vec<_>>(),
        vec![256]
            .into_iter()
            .chain(std::iter::repeat(0).take(31))
            .collect::<Vec<_>>(),
        vec![0]
            .into_iter()
            .chain(std::iter::repeat(0).take(31))
            .collect::<Vec<_>>(),
        true,
    );
    run_bad_long_arithmetic_test(&chip, trace, VerificationError::NonZeroCumulativeSum);
}

#[test]
fn long_mul_wrong_calculation_test() {
    let (chip, trace) = setup_bad_long_arithmetic_test(
        vec![255, 1]
            .into_iter()
            .chain(std::iter::repeat(0).take(30))
            .collect::<Vec<_>>(),
        vec![255, 1]
            .into_iter()
            .chain(std::iter::repeat(0).take(30))
            .collect::<Vec<_>>(),
        vec![1, 251, 3] // [1, 252, 3] is correct
            .into_iter()
            .chain(std::iter::repeat(0).take(29))
            .collect::<Vec<_>>(),
        vec![255, 2] // [254, 2] is correct
            .into_iter()
            .chain(std::iter::repeat(0).take(30))
            .collect::<Vec<_>>(),
        false,
    );
    run_bad_long_arithmetic_test(&chip, trace, VerificationError::OodEvaluationMismatch);
}
