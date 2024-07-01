use crate::cpu::trace::ProgramExecution;
use crate::cpu::OpCode;
use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use afs_test_utils::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

use super::columns::FieldExtensionArithmeticIOCols;
use super::FieldExtensionArithmeticAir;

/// Function for testing that generates a random program consisting only of field arithmetic operations.
fn generate_field_extension_program(len_ops: usize) -> ProgramExecution<1, BabyBear> {
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
    let field_extension_ops = FieldExtensionArithmeticAir::request(ops, operands);

    ProgramExecution {
        program: vec![],
        trace_rows: vec![],
        execution_frequencies: vec![],
        memory_accesses: vec![],
        arithmetic_ops: vec![],
        field_extension_ops,
    }
}

#[test]
fn field_extension_air_test() {
    let mut rng = create_seeded_rng();
    let len_ops = 1 << 5;
    let prog = generate_field_extension_program(len_ops);
    let extension_air = FieldExtensionArithmeticAir::new();

    let dummy_trace = RowMajorMatrix::new(
        prog.field_extension_ops
            .clone()
            .iter()
            .flat_map(|op| {
                [BabyBear::one()]
                    .into_iter()
                    .chain(op.to_vec())
                    .collect::<Vec<_>>()
            })
            .collect(),
        FieldExtensionArithmeticIOCols::<BabyBear>::get_width() + 1,
    );

    let mut extension_trace = extension_air.generate_trace(&prog);

    let page_requester = DummyInteractionAir::new(
        FieldExtensionArithmeticIOCols::<BabyBear>::get_width(),
        true,
        FieldExtensionArithmeticAir::BUS_INDEX,
    );

    // positive test
    run_simple_test_no_pis(
        vec![&extension_air, &page_requester],
        vec![extension_trace.clone(), dummy_trace.clone()],
    )
    .expect("Verification failed");

    // negative test pranking each IO value
    for height in 0..(prog.field_extension_ops.len()) {
        for width in 0..FieldExtensionArithmeticIOCols::<BabyBear>::get_width() {
            let prank_value = BabyBear::from_canonical_u32(rng.gen_range(1..=100));
            extension_trace.row_mut(height)[width] = prank_value;
        }

        // Run a test after pranking each row
        USE_DEBUG_BUILDER.with(|debug| {
            *debug.lock().unwrap() = false;
        });
        assert_eq!(
            run_simple_test_no_pis(
                vec![&extension_air, &page_requester],
                vec![extension_trace.clone(), dummy_trace.clone()],
            ),
            Err(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        )
    }
}
