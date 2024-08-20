use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis,
    interaction::dummy_interaction_air::DummyInteractionAir, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

use super::{
    columns::{FieldArithmeticCols, FieldArithmeticIoCols},
    FieldArithmeticChip,
};
use crate::cpu::{OpCode, OpCode::FDIV, ARITHMETIC_BUS, FIELD_ARITHMETIC_INSTRUCTIONS};

/// Function for testing that generates a random program consisting only of field arithmetic operations.
fn generate_arith_program(chip: &mut FieldArithmeticChip<BabyBear>, len_ops: usize) {
    let mut rng = create_seeded_rng();
    let ops = (0..len_ops)
        .map(|_| FIELD_ARITHMETIC_INSTRUCTIONS[rng.gen_range(0..4)])
        .collect();
    let operands = (0..len_ops)
        .map(|_| {
            (
                BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
                BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
            )
        })
        .collect();
    chip.request(ops, operands);
}

#[test]
fn au_air_test() {
    let mut rng = create_seeded_rng();
    let len_ops: usize = 3;
    let correct_height = len_ops.next_power_of_two();
    let mut chip = FieldArithmeticChip::new();
    generate_arith_program(&mut chip, len_ops);

    let empty_dummy_row = FieldArithmeticCols::<BabyBear>::blank_row().io.flatten();
    let dummy_trace = RowMajorMatrix::new(
        chip.operations
            .clone()
            .iter()
            .flat_map(|op| {
                [BabyBear::one()]
                    .into_iter()
                    .chain(op.to_vec())
                    .collect::<Vec<_>>()
            })
            .chain((0..(correct_height - len_ops)).flat_map(|_| empty_dummy_row.clone()))
            .collect(),
        FieldArithmeticIoCols::<BabyBear>::get_width(),
    );

    let mut au_trace = chip.generate_trace();

    let page_requester = DummyInteractionAir::new(
        FieldArithmeticIoCols::<BabyBear>::get_width() - 1,
        true,
        ARITHMETIC_BUS,
    );

    // positive test
    run_simple_test_no_pis(
        vec![&chip.air, &page_requester],
        vec![au_trace.clone(), dummy_trace.clone()],
    )
    .expect("Verification failed");

    // negative test pranking each IO value
    for height in 0..(chip.operations.len()) {
        for width in 0..FieldArithmeticIoCols::<BabyBear>::get_width() {
            let prank_value = BabyBear::from_canonical_u32(rng.gen_range(1..=100));
            au_trace.row_mut(height)[width] = prank_value;
        }

        // Run a test after pranking each row
        USE_DEBUG_BUILDER.with(|debug| {
            *debug.lock().unwrap() = false;
        });
        assert_eq!(
            run_simple_test_no_pis(
                vec![&chip.air, &page_requester],
                vec![au_trace.clone(), dummy_trace.clone()],
            ),
            Err(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        )
    }
}

#[test]
fn au_air_zero_div_zero() {
    let mut chip = FieldArithmeticChip::new();
    chip.calculate(OpCode::FDIV, (BabyBear::zero(), BabyBear::one()));

    let mut au_trace = chip.generate_trace();
    au_trace.row_mut(0)[3] = BabyBear::zero();
    let page_requester = DummyInteractionAir::new(
        FieldArithmeticIoCols::<BabyBear>::get_width() - 1,
        true,
        ARITHMETIC_BUS,
    );
    let dummy_trace = RowMajorMatrix::new(
        vec![
            BabyBear::one(),
            BabyBear::from_canonical_u32(FDIV as u32),
            BabyBear::zero(),
            BabyBear::zero(),
            BabyBear::zero(),
        ],
        FieldArithmeticIoCols::<BabyBear>::get_width(),
    );
    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(
            vec![&chip.air, &page_requester],
            vec![au_trace.clone(), dummy_trace.clone()],
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[should_panic]
#[test]
fn au_air_test_panic() {
    let mut chip = FieldArithmeticChip::new();
    // should panic
    chip.calculate(OpCode::FDIV, (BabyBear::zero(), BabyBear::zero()));
}
