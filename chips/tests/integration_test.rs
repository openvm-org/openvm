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

mod xor_requester;

type Val = BabyBear;

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
    all_chips.push(&xor_chip.air);

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

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    let result = run_simple_test_no_pis(
        vec![&dummy_requester, &xor_chip.air],
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
    all_chips.push(&xor_chip.air);
    all_chips.push(&xor_chip.xor_lookup_chip.air);

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

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    let result = run_simple_test_no_pis(
        vec![&requester, &xor_chip.air, &xor_chip.xor_lookup_chip.air],
        vec![requester_trace, xor_limbs_chip_trace, xor_lookup_chip_trace],
    );

    assert_eq!(
        result,
        Err(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it passed"
    );
}
