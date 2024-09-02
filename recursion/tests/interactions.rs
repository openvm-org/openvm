use std::sync::Arc;

use afs_primitives::{range_gate::RangeCheckerGateChip, sum::SumChip};
use afs_stark_backend::rap::AnyRap;
use ax_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Config, fri_params::default_fri_params, setup_tracing,
    },
    interaction::dummy_interaction_air::DummyInteractionAir,
    utils::to_field_vec,
};
use p3_matrix::dense::RowMajorMatrix;

mod common;

#[test]
fn test_interactions() {
    type SC = BabyBearPoseidon2Config;

    const INPUT_BUS: usize = 0;
    const OUTPUT_BUS: usize = 1;
    const RANGE_BUS: usize = 2;
    const RANGE_MAX: u32 = 16;

    setup_tracing();

    let range_checker = Arc::new(RangeCheckerGateChip::new(RANGE_BUS, RANGE_MAX));
    let sum_chip = SumChip::new(INPUT_BUS, OUTPUT_BUS, 4, 4, range_checker);

    let mut sum_trace_u32 = Vec::<(u32, u32, u32, u32)>::new();
    let n = 16;
    for i in 0..n {
        sum_trace_u32.push((0, 1, i + 1, (i == n - 1) as u32));
    }

    let kv: &[(u32, u32)] = &sum_trace_u32
        .iter()
        .map(|&(key, value, _, _)| (key, value))
        .collect::<Vec<_>>();
    let sum_trace = sum_chip.generate_trace(kv);
    let sender_air = DummyInteractionAir::new(2, true, INPUT_BUS);
    let sender_trace = RowMajorMatrix::new(
        to_field_vec(
            sum_trace_u32
                .iter()
                .flat_map(|&(key, val, _, _)| [1, key, val])
                .collect(),
        ),
        sender_air.field_width() + 1,
    );
    let receiver_air = DummyInteractionAir::new(2, false, OUTPUT_BUS);
    let receiver_trace = RowMajorMatrix::new(
        to_field_vec(
            sum_trace_u32
                .iter()
                .flat_map(|&(key, _, sum, is_final)| [is_final, key, sum])
                .collect(),
        ),
        receiver_air.field_width() + 1,
    );
    let range_checker_trace = sum_chip.range_checker.generate_trace();

    let any_raps: Vec<&dyn AnyRap<SC>> = vec![
        &sum_chip.air,
        &sender_air,
        &receiver_air,
        &sum_chip.range_checker.air,
    ];
    let traces = vec![sum_trace, sender_trace, receiver_trace, range_checker_trace];
    let pvs = vec![vec![], vec![], vec![], vec![]];

    common::run_recursive_test(any_raps, traces, pvs, default_fri_params())
}
