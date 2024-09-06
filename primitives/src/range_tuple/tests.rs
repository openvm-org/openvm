use std::iter;

use afs_stark_backend::{prover::USE_DEBUG_BUILDER, rap::AnyRap, verifier::VerificationError};
use ax_sdk::{
    config::baby_bear_blake3::run_simple_test_no_pis,
    interaction::dummy_interaction_air::DummyInteractionAir, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use rand::Rng;

use crate::range_tuple::RangeTupleCheckerChip;

#[test]
fn test_range_tuple_chip() {
    let mut rng = create_seeded_rng();

    let bus_index = 0;
    let sizes = (0..3)
        .map(|_| 1 << rng.gen_range(1..5))
        .collect::<Vec<u32>>();

    const LIST_LEN: usize = 64;

    let range_checker = RangeTupleCheckerChip::new(bus_index, sizes.clone());
    let mut gen_list = || {
        sizes
            .iter()
            .map(|&size| rng.gen_range(0..size))
            .collect::<Vec<_>>()
    };
    // Generating random lists
    let num_lists = 10;
    let lists_vals = (0..num_lists)
        .map(|_| (0..LIST_LEN).map(|_| gen_list()).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let lists = (0..num_lists)
        .map(|_| DummyInteractionAir::new(sizes.len(), true, bus_index))
        .collect::<Vec<DummyInteractionAir>>();

    let lists_traces = lists_vals
        .par_iter()
        .map(|list| {
            RowMajorMatrix::new(
                list.clone()
                    .into_iter()
                    .flat_map(|v| {
                        range_checker.add_count(&v);
                        iter::once(1).chain(v)
                    })
                    .map(AbstractField::from_wrapped_u32)
                    .collect(),
                sizes.len() + 1,
            )
        })
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    let range_trace = range_checker.generate_trace();

    let mut all_chips = lists
        .iter()
        .map(|list| list as &dyn AnyRap<_>)
        .collect::<Vec<_>>();
    all_chips.push(&range_checker.air);

    let all_traces = lists_traces
        .into_iter()
        .chain(iter::once(range_trace))
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    run_simple_test_no_pis(all_chips, all_traces).expect("Verification failed");
}

#[test]
fn negative_test_range_tuple_chip() {
    let bus_index = 0;

    let sizes = vec![2, 2, 8];

    let range_checker = RangeTupleCheckerChip::new(bus_index, sizes.clone());

    // generating a trace with a "counter" starting from 1
    // instead of 0 to test the AIR constraints in range_checker

    let height = sizes.iter().product();
    let range_trace = RowMajorMatrix::new(
        (1..=height)
            .flat_map(|idx| {
                let mut idx = idx;
                let mut v = vec![];
                for size in sizes.iter().rev() {
                    let val = idx % size;
                    idx /= size;
                    v.push(val);
                }
                v.reverse();
                v.into_iter().chain(iter::once(0))
            })
            .map(AbstractField::from_wrapped_u32)
            .collect(),
        sizes.len() + 1,
    );

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&range_checker.air], vec![range_trace]),
        Err(VerificationError::NonZeroCumulativeSum),
        "Expected constraint to fail"
    );
}
