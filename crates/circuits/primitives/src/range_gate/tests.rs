use std::{iter, sync::Arc};

use openvm_stark_backend::{
    p3_field::FieldAlgebra, p3_matrix::dense::RowMajorMatrix, p3_maybe_rayon::prelude::*,
    prover::types::AirProvingContext, utils::disable_debug_builder, AirRef,
};
use openvm_stark_sdk::{
    any_rap_arc_vec, dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    p3_baby_bear::BabyBear, utils::create_seeded_rng,
};
use rand::Rng;
use stark_backend_v2::{prover::AirProvingContextV2, test_utils::test_engine_small, StarkEngineV2};

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
        .map(|_| DummyInteractionAir::new(1, true, bus.index()))
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
                    .map(FieldAlgebra::from_wrapped_u32)
                    .collect(),
                2,
            )
        })
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    let range_trace = range_checker.generate_trace();

    let mut all_chips = lists
        .into_iter()
        .map(|list| Arc::new(list) as AirRef<_>)
        .collect::<Vec<_>>();
    all_chips.push(Arc::new(range_checker.air));

    let all_traces = lists_traces
        .into_iter()
        .chain(iter::once(range_trace))
        .map(Arc::new)
        .map(AirProvingContext::simple_no_pis)
        .map(AirProvingContextV2::from_v1_no_cached)
        .collect::<Vec<_>>();

    test_engine_small()
        .run_test(all_chips, all_traces)
        .expect("Verification failed");
}

#[test]
#[should_panic]
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
            .map(FieldAlgebra::from_wrapped_u32)
            .collect(),
        2,
    );

    let traces = [range_trace]
        .into_iter()
        .map(Arc::new)
        .map(AirProvingContext::simple_no_pis)
        .map(AirProvingContextV2::from_v1_no_cached)
        .collect::<Vec<_>>();

    disable_debug_builder();
    test_engine_small()
        .run_test(any_rap_arc_vec![range_checker.air], traces)
        .unwrap();
}
