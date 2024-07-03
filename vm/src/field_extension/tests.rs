use std::ops::{Add, Div, Mul, Sub};

use crate::cpu::OpCode;
use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use afs_test_utils::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractExtensionField, AbstractField};
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

use super::columns::FieldExtensionArithmeticIoCols;
use super::{FieldExtensionArithmeticAir, FieldExtensionArithmeticChip};

/// Function for testing that generates a random program consisting only of field arithmetic operations.
fn generate_field_extension_program(
    chip: &mut FieldExtensionArithmeticChip<BabyBear>,
    len_ops: usize,
) {
    let mut rng = create_seeded_rng();
    let ops = (0..len_ops)
        .map(|_| OpCode::from_u8(rng.gen_range(13..=13)).unwrap())
        .collect();
    let operands = (0..len_ops)
        .map(|_| {
            (
                [
                    BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
                    BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
                    BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
                    BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
                ],
                [
                    BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
                    BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
                    BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
                    BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
                ],
            )
        })
        .collect();
    chip.request(ops, operands);
}

#[test]
fn field_extension_air_test() {
    let mut rng = create_seeded_rng();
    let len_ops: usize = 1 << 5;
    let mut chip = FieldExtensionArithmeticChip::new();
    generate_field_extension_program(&mut chip, len_ops);

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
            .collect(),
        FieldExtensionArithmeticIoCols::<BabyBear>::get_width() + 1,
    );

    let mut extension_trace = chip.generate_trace();

    let page_requester = DummyInteractionAir::new(
        FieldExtensionArithmeticIoCols::<BabyBear>::get_width(),
        true,
        FieldExtensionArithmeticAir::BUS_INDEX,
    );

    // positive test
    run_simple_test_no_pis(
        vec![&chip.air, &page_requester],
        vec![extension_trace.clone(), dummy_trace.clone()],
    )
    .expect("Verification failed");

    // Run a test after pranking each row
    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });

    // negative test pranking each IO value
    for height in 0..(chip.operations.len()) {
        for width in 0..FieldExtensionArithmeticIoCols::<BabyBear>::get_width() {
            let prank_value = BabyBear::from_canonical_u32(rng.gen_range(1..=100));
            extension_trace.row_mut(height)[width] = prank_value;
        }

        assert_eq!(
            run_simple_test_no_pis(
                vec![&chip.air, &page_requester],
                vec![extension_trace.clone(), dummy_trace.clone()],
            ),
            Err(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        )
    }
}

#[test]
fn field_extension_consistency_test() {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    let len_tests = 100;
    let mut rng = create_seeded_rng();

    let operands: Vec<([F; 4], [F; 4])> = (0..len_tests)
        .map(|_| {
            (
                [
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                ],
                [
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                ],
            )
        })
        .collect();

    for (a, b) in operands {
        let a_ext = EF::from_base_slice(&a);
        let b_ext = EF::from_base_slice(&b);

        let plonky_add = a_ext.add(b_ext);
        let plonky_sub = a_ext.sub(b_ext);
        let plonky_mul = a_ext.mul(b_ext);
        let plonky_div = a_ext.div(b_ext);

        let my_add = FieldExtensionArithmeticAir::solve(OpCode::FE4ADD, a, b);
        let my_sub = FieldExtensionArithmeticAir::solve(OpCode::FE4SUB, a, b);
        let my_mul = FieldExtensionArithmeticAir::solve(OpCode::BBE4MUL, a, b);

        let b_inv = FieldExtensionArithmeticAir::solve(OpCode::BBE4INV, b, [F::zero(); 4]).unwrap();
        let my_div = FieldExtensionArithmeticAir::solve(OpCode::BBE4MUL, a, b_inv);

        assert_eq!(my_add.unwrap(), plonky_add.as_base_slice());
        assert_eq!(my_sub.unwrap(), plonky_sub.as_base_slice());
        assert_eq!(my_mul.unwrap(), plonky_mul.as_base_slice());
        assert_eq!(my_div.unwrap(), plonky_div.as_base_slice());
    }
}
