use super::columns::FieldArithmeticIOCols;
use super::FieldArithmeticAir;
use crate::cpu::trace::{ArithmeticOperation, ProgramExecution};
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

/// Function for testing that generates a random program consisting only of field arithmetic operations.
fn generate_arith_program(len_ops: usize) -> ProgramExecution<1, BabyBear> {
    let mut rng = create_seeded_rng();
    let ops = (0..len_ops)
        .map(|_| OpCode::from_u8(rng.gen_range(6..=9)).unwrap())
        .collect();
    let operands = (0..len_ops)
        .map(|_| {
            (
                BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
                BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
            )
        })
        .collect();
    let arith_ops = FieldArithmeticAir::request(ops, operands);

    ProgramExecution {
        program: vec![],
        trace_rows: vec![],
        execution_frequencies: vec![],
        memory_accesses: vec![],
        arithmetic_ops: arith_ops,
    }
}

#[test]
fn au_air_test() {
    let mut rng = create_seeded_rng();
    let len_ops = 1 << 5;
    let prog = generate_arith_program(len_ops);
    let au_air = FieldArithmeticAir::new();

    let dummy_trace = RowMajorMatrix::new(
        prog.arithmetic_ops
            .clone()
            .iter()
            .flat_map(|op| {
                [BabyBear::one()]
                    .into_iter()
                    .chain(op.to_vec())
                    .collect::<Vec<_>>()
            })
            .collect(),
        FieldArithmeticIOCols::<BabyBear>::get_width() + 1,
    );

    let mut au_trace = au_air.generate_trace(&prog);

    let page_requester = DummyInteractionAir::new(
        FieldArithmeticIOCols::<BabyBear>::get_width(),
        true,
        FieldArithmeticAir::BUS_INDEX,
    );

    // positive test
    run_simple_test_no_pis(
        vec![&au_air, &page_requester],
        vec![au_trace.clone(), dummy_trace.clone()],
    )
    .expect("Verification failed");

    // negative test pranking each IO value
    for height in 0..(prog.arithmetic_ops.len()) {
        for width in 0..FieldArithmeticIOCols::<BabyBear>::get_width() {
            let prank_value = BabyBear::from_canonical_u32(rng.gen_range(1..=100));
            au_trace.row_mut(height)[width] = prank_value;
        }

        // Run a test after pranking each row
        USE_DEBUG_BUILDER.with(|debug| {
            *debug.lock().unwrap() = false;
        });
        assert_eq!(
            run_simple_test_no_pis(
                vec![&au_air, &page_requester],
                vec![au_trace.clone(), dummy_trace.clone()],
            ),
            Err(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        )
    }

    let zero_div_zero_prog = ProgramExecution {
        program: vec![],
        trace_rows: vec![],
        execution_frequencies: vec![],
        memory_accesses: vec![],
        arithmetic_ops: vec![ArithmeticOperation {
            opcode: OpCode::FDIV,
            operand1: BabyBear::zero(),
            operand2: BabyBear::one(),
            result: BabyBear::zero(),
        }],
    };

    let mut au_trace = au_air.generate_trace(&zero_div_zero_prog);
    au_trace.row_mut(0)[2] = BabyBear::zero();
    let page_requester = DummyInteractionAir::new(
        FieldArithmeticIOCols::<BabyBear>::get_width(),
        true,
        FieldArithmeticAir::BUS_INDEX,
    );
    let dummy_trace = RowMajorMatrix::new(
        vec![
            BabyBear::one(),
            BabyBear::from_canonical_u32(OpCode::FDIV as u32),
            BabyBear::zero(),
            BabyBear::zero(),
            BabyBear::zero(),
        ],
        FieldArithmeticIOCols::<BabyBear>::get_width() + 1,
    );
    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(
            vec![&au_air, &page_requester],
            vec![au_trace.clone(), dummy_trace.clone()],
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[should_panic]
#[test]
fn au_air_test_panic() {
    let au_air = FieldArithmeticAir::new();

    let zero_div_zero_prog = ProgramExecution {
        program: vec![],
        trace_rows: vec![],
        execution_frequencies: vec![],
        memory_accesses: vec![],
        arithmetic_ops: vec![ArithmeticOperation {
            opcode: OpCode::FDIV,
            operand1: BabyBear::zero(),
            operand2: BabyBear::zero(),
            result: BabyBear::zero(),
        }],
    };

    // Should panic
    au_air.generate_trace(&zero_div_zero_prog);
}
