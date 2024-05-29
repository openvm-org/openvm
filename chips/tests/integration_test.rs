use std::{iter, sync::Arc};

use afs_chips::{range, range_gate, xor_bits, xor_limbs};
use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::rap::AnyRap;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use afs_test_utils::{
    interaction::dummy_interaction_air::DummyInteractionAir, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use rand::Rng;

mod list;
mod xor_requester;

type Val = BabyBear;

#[test]
fn test_list_range_checker() {
    let mut rng = create_seeded_rng();

    use list::ListChip;
    use range::RangeCheckerChip;

    let bus_index = 0;

    const LOG_TRACE_DEGREE_RANGE: usize = 3;
    const MAX: u32 = 1 << LOG_TRACE_DEGREE_RANGE;

    const LOG_TRACE_DEGREE_LIST: usize = 6;
    const LIST_LEN: usize = 1 << LOG_TRACE_DEGREE_LIST;

    // Creating a RangeCheckerChip
    let range_checker = Arc::new(RangeCheckerChip::<MAX>::new(bus_index));

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
        .map(|vals| ListChip::new(bus_index, vals.to_vec(), Arc::clone(&range_checker)))
        .collect::<Vec<ListChip<MAX>>>();

    let lists_traces = lists
        .par_iter()
        .map(|list| list.generate_trace())
        .collect::<Vec<DenseMatrix<BabyBear>>>();

    let range_trace = range_checker.generate_trace();

    let mut all_chips: Vec<&dyn AnyRap<_>> = vec![];
    for list in &lists {
        all_chips.push(list);
    }
    all_chips.push(&*range_checker);

    let all_traces = lists_traces
        .into_iter()
        .chain(iter::once(range_trace))
        .collect::<Vec<DenseMatrix<BabyBear>>>();

    run_simple_test_no_pis(all_chips, all_traces).expect("Verification failed");
}

#[test]
fn test_xor_bits_chip() {
    let mut rng = create_seeded_rng();

    use xor_bits::XorBitsChip;
    use xor_requester::XorRequesterChip;

    let bus_index = 0;

    const BITS: usize = 3;
    const MAX: u32 = 1 << BITS;

    const LOG_XOR_REQUESTS: usize = 4;
    const XOR_REQUESTS: usize = 1 << LOG_XOR_REQUESTS;

    const LOG_NUM_REQUESTERS: usize = 3;
    const NUM_REQUESTERS: usize = 1 << LOG_NUM_REQUESTERS;

    let xor_chip = Arc::new(XorBitsChip::<BITS>::new(bus_index, vec![]));

    let mut requesters = (0..NUM_REQUESTERS)
        .map(|_| XorRequesterChip::new(bus_index, vec![], Arc::clone(&xor_chip)))
        .collect::<Vec<XorRequesterChip<BITS>>>();

    for requester in &mut requesters {
        for _ in 0..XOR_REQUESTS {
            requester.add_request(rng.gen::<u32>() % MAX, rng.gen::<u32>() % MAX);
        }
    }

    let requesters_traces = requesters
        .par_iter()
        .map(|requester| requester.generate_trace())
        .collect::<Vec<DenseMatrix<BabyBear>>>();

    let xor_chip_trace = xor_chip.generate_trace();

    let mut all_chips: Vec<&dyn AnyRap<_>> = vec![];
    for requester in &requesters {
        all_chips.push(requester);
    }
    all_chips.push(&*xor_chip);

    let all_traces = requesters_traces
        .into_iter()
        .chain(iter::once(xor_chip_trace))
        .collect::<Vec<DenseMatrix<BabyBear>>>();

    run_simple_test_no_pis(all_chips, all_traces).expect("Verification failed");
}

#[test]
fn negative_test_xor_bits_chip() {
    let mut rng = create_seeded_rng();

    use xor_bits::XorBitsChip;

    let bus_index = 0;

    const BITS: usize = 3;
    const MAX: u32 = 1 << BITS;

    const LOG_XOR_REQUESTS: usize = 4;
    const XOR_REQUESTS: usize = 1 << LOG_XOR_REQUESTS;

    let xor_chip = Arc::new(XorBitsChip::<BITS>::new(bus_index, vec![]));

    let dummy_requester = DummyInteractionAir::new(3, true, 0);

    let mut reqs = vec![];
    for _ in 0..XOR_REQUESTS {
        let x = rng.gen::<u32>() % MAX;
        let y = rng.gen::<u32>() % MAX;
        reqs.push((1, vec![x, y, x ^ y]));
        xor_chip.request(x, y);
    }

    // Modifying one of the values to send incompatible values
    reqs[0].1[2] += 1;

    let xor_chip_trace = xor_chip.generate_trace();

    let dummy_trace = RowMajorMatrix::new(
        reqs.into_iter()
            .flat_map(|(count, fields)| iter::once(count).chain(fields))
            .map(Val::from_wrapped_u32)
            .collect(),
        4,
    );

    let result = run_simple_test_no_pis(
        vec![&dummy_requester, &*xor_chip],
        vec![dummy_trace, xor_chip_trace],
    );

    assert_eq!(
        result,
        Err(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_xor_limbs_chip() {
    let mut rng = create_seeded_rng();

    use xor_limbs::XorLimbsChip;

    let bus_index = 0;

    const N: usize = 6;
    const M: usize = 2;
    const LOG_XOR_REQUESTS: usize = 1;
    const LOG_NUM_REQUESTERS: usize = 2;

    const MAX_INPUT: u32 = 1 << N;
    const XOR_REQUESTS: usize = 1 << LOG_XOR_REQUESTS;
    const NUM_REQUESTERS: usize = 1 << LOG_NUM_REQUESTERS;

    let xor_chip = XorLimbsChip::<N, M>::new(bus_index, vec![]);

    let requesters_lists = (0..NUM_REQUESTERS)
        .map(|_| {
            (0..XOR_REQUESTS)
                .map(|_| {
                    let x = rng.gen::<u32>() % MAX_INPUT;
                    let y = rng.gen::<u32>() % MAX_INPUT;

                    (1, vec![x, y])
                })
                .collect::<Vec<(u32, Vec<u32>)>>()
        })
        .collect::<Vec<Vec<(u32, Vec<u32>)>>>();

    let requesters = (0..NUM_REQUESTERS)
        .map(|_| DummyInteractionAir::new(3, true, 0))
        .collect::<Vec<DummyInteractionAir>>();

    let requesters_traces = requesters_lists
        .par_iter()
        .map(|list| {
            RowMajorMatrix::new(
                list.clone()
                    .into_iter()
                    .flat_map(|(count, fields)| {
                        let x = fields[0];
                        let y = fields[1];
                        let z = xor_chip.request(x, y);
                        iter::once(count).chain(fields).chain(iter::once(z))
                    })
                    .map(Val::from_wrapped_u32)
                    .collect(),
                4,
            )
        })
        .collect::<Vec<DenseMatrix<BabyBear>>>();

    let xor_limbs_chip_trace = xor_chip.generate_trace();
    let xor_lookup_chip_trace = xor_chip.xor_lookup_chip.generate_trace();

    let mut all_chips: Vec<&dyn AnyRap<_>> = vec![];
    for requester in &requesters {
        all_chips.push(requester);
    }
    all_chips.push(&xor_chip);
    all_chips.push(&xor_chip.xor_lookup_chip);

    let all_traces = requesters_traces
        .into_iter()
        .chain(iter::once(xor_limbs_chip_trace))
        .chain(iter::once(xor_lookup_chip_trace))
        .collect::<Vec<DenseMatrix<BabyBear>>>();

    run_simple_test_no_pis(all_chips, all_traces).expect("Verification failed");
}

#[test]
fn negative_test_xor_limbs_chip() {
    let mut rng = create_seeded_rng();

    use xor_limbs::XorLimbsChip;

    let bus_index = 0;

    const N: usize = 6;
    const M: usize = 2;
    const LOG_XOR_REQUESTS: usize = 3;

    const MAX_INPUT: u32 = 1 << N;
    const XOR_REQUESTS: usize = 1 << LOG_XOR_REQUESTS;

    let xor_chip = XorLimbsChip::<N, M>::new(bus_index, vec![]);

    let pairs = (0..XOR_REQUESTS)
        .map(|_| {
            let x = rng.gen::<u32>() % MAX_INPUT;
            let y = rng.gen::<u32>() % MAX_INPUT;

            (1, vec![x, y])
        })
        .collect::<Vec<(u32, Vec<u32>)>>();

    let requester = DummyInteractionAir::new(3, true, 0);

    let requester_trace = RowMajorMatrix::new(
        pairs
            .clone()
            .into_iter()
            .enumerate()
            .flat_map(|(index, (count, fields))| {
                let x = fields[0];
                let y = fields[1];
                let z = xor_chip.request(x, y);

                if index == 0 {
                    // Modifying one of the values to send incompatible values
                    iter::once(count).chain(fields).chain(iter::once(z + 1))
                } else {
                    iter::once(count).chain(fields).chain(iter::once(z))
                }
            })
            .map(Val::from_wrapped_u32)
            .collect(),
        4,
    );

    let xor_limbs_chip_trace = xor_chip.generate_trace();
    let xor_lookup_chip_trace = xor_chip.xor_lookup_chip.generate_trace();

    let result = run_simple_test_no_pis(
        vec![&requester, &xor_chip, &xor_chip.xor_lookup_chip],
        vec![requester_trace, xor_limbs_chip_trace, xor_lookup_chip_trace],
    );

    assert_eq!(
        result,
        Err(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_range_gate_chip() {
    let mut rng = create_seeded_rng();

    use range_gate::RangeCheckerGateChip;

    let bus_index = 0;

    const N: usize = 3;
    const MAX: u32 = 1 << N;

    const LOG_LIST_LEN: usize = 6;
    const LIST_LEN: usize = 1 << LOG_LIST_LEN;

    let range_checker = RangeCheckerGateChip::<MAX>::new(bus_index);

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
        .map(|_| DummyInteractionAir::new(1, true, bus_index))
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
                    .map(Val::from_wrapped_u32)
                    .collect(),
                2,
            )
        })
        .collect::<Vec<DenseMatrix<BabyBear>>>();

    let range_trace = range_checker.generate_trace();

    let mut all_chips = lists
        .iter()
        .map(|list| list as &dyn AnyRap<_>)
        .collect::<Vec<_>>();
    all_chips.push(&range_checker);

    let all_traces = lists_traces
        .into_iter()
        .chain(iter::once(range_trace))
        .collect::<Vec<DenseMatrix<BabyBear>>>();

    run_simple_test_no_pis(all_chips, all_traces).expect("Verification failed");
}

#[test]
fn negative_test_range_gate_chip() {
    use range_gate::RangeCheckerGateChip;

    let bus_index = 0;

    const N: usize = 3;
    const MAX: u32 = 1 << N;

    let range_checker = RangeCheckerGateChip::<MAX>::new(bus_index);

    // generating a trace with a counter starting from 1
    // instead of 0 to test the AIR constraints in range_checker
    let range_trace = RowMajorMatrix::new(
        (0..MAX)
            .flat_map(|i| {
                let count =
                    range_checker.count[i as usize].load(std::sync::atomic::Ordering::Relaxed);
                iter::once(i + 1).chain(iter::once(count))
            })
            .map(Val::from_wrapped_u32)
            .collect(),
        2,
    );

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&range_checker], vec![range_trace]),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}
