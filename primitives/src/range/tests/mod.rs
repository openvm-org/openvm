use std::{iter, sync::Arc};

use afs_stark_backend::rap::AnyRap;
use ax_sdk::{config::baby_bear_blake3::run_simple_test_no_pis, utils::create_seeded_rng};
use list::ListChip;
use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use rand::Rng;

use crate::range::{bus::RangeCheckBus, RangeCheckerChip};

/// List chip for testing
pub mod list;

#[test]
fn test_list_range_checker() {
    let mut rng = create_seeded_rng();

    const LOG_TRACE_DEGREE_RANGE: usize = 3;
    const MAX: u32 = 1 << LOG_TRACE_DEGREE_RANGE;

    let bus = RangeCheckBus::new(0, MAX);

    const LOG_TRACE_DEGREE_LIST: usize = 6;
    const LIST_LEN: usize = 1 << LOG_TRACE_DEGREE_LIST;

    // Creating a RangeCheckerChip
    let range_checker = Arc::new(RangeCheckerChip::new(bus));

    // Generating random lists
    let num_lists = 10;
    let lists_vals = (0..num_lists)
        .map(|_| {
            (0..LIST_LEN)
                .map(|_| rng.gen::<u32>() % MAX)
                .collect::<Vec<u32>>()
        })
        .collect::<Vec<Vec<u32>>>();

    // define a bunch of ListChips
    let lists = lists_vals
        .iter()
        .map(|vals| ListChip::new(vals.to_vec(), Arc::clone(&range_checker)))
        .collect::<Vec<ListChip>>();

    let lists_traces = lists
        .par_iter()
        .map(|list| list.generate_trace())
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    let range_trace = range_checker.generate_trace();

    let mut all_chips: Vec<&dyn AnyRap<_>> = vec![];
    for list in &lists {
        all_chips.push(&list.air);
    }
    all_chips.push(&range_checker.air);

    let all_traces = lists_traces
        .into_iter()
        .chain(iter::once(range_trace))
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    run_simple_test_no_pis(all_chips, all_traces).expect("Verification failed");
}
