use std::{iter, sync::Arc};

use ax_stark_backend::{rap::AnyRap, utils::disable_debug_builder, verifier::VerificationError};
use ax_stark_sdk::{
    any_rap_arc_vec, config::baby_bear_blake3::BabyBearBlake3Engine,
    dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir, engine::StarkFriEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use rand::Rng;

use crate::{range::RangeCheckBus, range_gate::RangeCheckerGateChip};

#[test]
fn test_range_gate_chip() {
    let mut rng = create_seeded_rng();

    const N: usize = 3;
    const MAX: u32 = 1 << N;

    const LOG_LIST_LEN: usize = 6;
    const LIST_LEN: usize = 1 << LOG_LIST_LEN;

    let bus = RangeCheckBus::new(0, MAX);
    let range_checker = RangeCheckerGateChip::new(bus);

    // Generating random lists
    let num_lists = 10;
    let lists_vals = (0..num_lists)
        .map(|_| {
            (0..LIST_LEN)
                .map(|_| rng.gen::<u32>() % MAX)
                .collect::<Vec<u32>>()
        })
        .collect::<Vec<Vec<u32>>>();

    let lists = (0..num_lists)
        .map(|_| DummyInteractionAir::new(1, true, bus.index))
        .collect::<Vec<DummyInteractionAir>>();

    let lists_traces = lists_vals
        .par_iter()
        .map(|list| {
            RowMajorMatrix::new(
                list.clone()
                    .into_iter()
                    .flat_map(|v| {
                        range_checker.add_count(v);
                        iter::once(1).chain(iter::once(v))
                    })
                    .map(AbstractField::from_wrapped_u32)
                    .collect(),
                2,
            )
        })
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    let range_trace = range_checker.generate_trace();

    let mut all_chips = lists
        .into_iter()
        .map(|list| Arc::new(list) as Arc<dyn AnyRap<_>>)
        .collect::<Vec<_>>();
    all_chips.push(Arc::new(range_checker.air));

    let all_traces = lists_traces
        .into_iter()
        .chain(iter::once(range_trace))
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(all_chips, all_traces)
        .expect("Verification failed");
}

#[test]
fn negative_test_range_gate_chip() {
    const N: usize = 3;
    const MAX: u32 = 1 << N;

    let bus = RangeCheckBus::new(0, MAX);
    let range_checker = RangeCheckerGateChip::new(bus);

    // generating a trace with a counter starting from 1
    // instead of 0 to test the AIR constraints in range_checker
    let range_trace = RowMajorMatrix::new(
        (0..MAX)
            .flat_map(|i| {
                let count =
                    range_checker.count[i as usize].load(std::sync::atomic::Ordering::Relaxed);
                iter::once(i + 1).chain(iter::once(count))
            })
            .map(AbstractField::from_wrapped_u32)
            .collect(),
        2,
    );

    disable_debug_builder();
    assert_eq!(
        BabyBearBlake3Engine::run_simple_test_no_pis_fast(
            any_rap_arc_vec![range_checker.air],
            vec![range_trace]
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}
