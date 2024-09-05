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

use crate::var_range::{bus::VariableRangeCheckBus, VariableRangeCheckerChip};

#[test]
fn test_variable_range_checker_chip() {
    let mut rng = create_seeded_rng();

    const MAX_BITS: u32 = 3;
    const LOG_LIST_LEN: usize = 8;
    const LIST_LEN: usize = 1 << LOG_LIST_LEN;

    let bus = VariableRangeCheckBus::new(0, MAX_BITS);
    let var_range_checker = VariableRangeCheckerChip::new(bus);

    // generate randomized valid values-bits pairs
    let num_lists = 10;
    let lists_vals = (0..num_lists)
        .map(|_| {
            (0..LIST_LEN)
                .map(|_| {
                    let bits = rng.gen::<u32>() % (MAX_BITS + 1);
                    let val = rng.gen::<u32>() % (1 << bits);
                    [val, bits]
                })
                .collect::<Vec<[u32; 2]>>()
        })
        .collect::<Vec<Vec<[u32; 2]>>>();

    // generate AIR chips for each list
    let lists_airs = (0..num_lists)
        .map(|_| DummyInteractionAir::new(2, true, bus.index))
        .collect::<Vec<DummyInteractionAir>>();

    let mut all_chips = lists_airs
        .iter()
        .map(|list| list as &dyn AnyRap<_>)
        .collect::<Vec<_>>();
    all_chips.push(&var_range_checker.air);

    // generate traces for each list
    let lists_traces = lists_vals
        .par_iter()
        .map(|list| {
            RowMajorMatrix::new(
                list.clone()
                    .into_iter()
                    .flat_map(|[val, bits]| {
                        var_range_checker.add_count(val, bits);
                        iter::once(1).chain(iter::once(val)).chain(iter::once(bits))
                    })
                    .map(AbstractField::from_wrapped_u32)
                    .collect(),
                3,
            )
        })
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    let var_range_checker_trace = var_range_checker.generate_trace();

    let all_traces = lists_traces
        .into_iter()
        .chain(iter::once(var_range_checker_trace))
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    run_simple_test_no_pis(all_chips, all_traces).expect("Verification failed");
}

#[test]
fn negative_test_variable_range_checker_chip() {
    // test that the constraint fails when some val >= 2^max_bits
    let mut rng = create_seeded_rng();

    const MAX_BITS: u32 = 3;
    const LOG_LIST_LEN: usize = 8;
    const LIST_LEN: usize = 1 << LOG_LIST_LEN;

    let bus = VariableRangeCheckBus::new(0, MAX_BITS);
    let var_range_checker = VariableRangeCheckerChip::new(bus);

    // generate randomized valid values-bits pairs
    let list_vals = (0..(LIST_LEN - 1))
        .map(|_| {
            let bits = rng.gen::<u32>() % (MAX_BITS + 1);
            let val = rng.gen::<u32>() % (1 << bits);
            [val, bits]
        })
        .collect::<Vec<[u32; 2]>>();

    // generate dummy AIR chip
    let list_chip = DummyInteractionAir::new(2, true, bus.index);
    let all_chips = vec![&list_chip as &dyn AnyRap<_>, &var_range_checker.air];

    // generate trace with a [val, bits] pair such that val >= 2^bits (i.e. [4, 2])
    let list_trace = RowMajorMatrix::new(
        list_vals
            .clone()
            .into_iter()
            .chain(iter::once([4, 2]))
            .flat_map(|[val, bits]| {
                var_range_checker.add_count(val, bits);
                iter::once(1).chain(iter::once(val)).chain(iter::once(bits))
            })
            .map(AbstractField::from_wrapped_u32)
            .collect(),
        3,
    );
    let var_range_trace = var_range_checker.generate_trace();
    let all_traces = vec![list_trace, var_range_trace];

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(all_chips, all_traces),
        Err(VerificationError::NonZeroCumulativeSum),
        "Expected constraint to fail"
    );
}
