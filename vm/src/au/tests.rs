use crate::au::columns::AUIOCols;
use crate::au::AUAir;
use crate::cpu::trace::ProgramExecution;
use crate::cpu::OpCode;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use afs_test_utils::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

fn generate_arith_program() -> ProgramExecution<BabyBear> {
    let mut rng = create_seeded_rng();
    let len_ops = 1;
    let ops = (0..len_ops)
        .map(|_| OpCode::from_u8(rng.gen_range(5..=8)).unwrap())
        .collect();
    let operands = (0..len_ops)
        .map(|_| {
            (
                BabyBear::from_canonical_u32(rng.gen_range(0..=100)),
                BabyBear::from_canonical_u32(rng.gen_range(0..=100)),
            )
        })
        .collect();
    let arith_ops = AUAir::request(ops, operands);

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
    let prog = generate_arith_program();
    let au_air = AUAir::new();

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
        AUIOCols::<BabyBear>::get_width() + 1,
    );

    let au_trace = au_air.generate_trace(&prog);

    let page_requester =
        DummyInteractionAir::new(AUIOCols::<BabyBear>::get_width(), true, AUAir::BUS_INDEX);

    run_simple_test_no_pis(vec![&au_air, &page_requester], vec![au_trace, dummy_trace])
        .expect("Verification failed");
}
