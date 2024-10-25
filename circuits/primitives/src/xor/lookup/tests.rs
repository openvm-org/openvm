use std::{iter, sync::Arc};

use afs_stark_backend::{rap::AnyRap, utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::{
    any_rap_arc_vec, config::baby_bear_blake3::BabyBearBlake3Engine,
    dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir, engine::StarkFriEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use rand::Rng;

use crate::xor::XorLookupChip;

// duplicated here from vm/src/system/vm/chip_set.rs to avoid importing vm in afs-primitives
const BYTE_XOR_BUS: usize = 10;

#[test]
fn test_xor_limbs_chip() {
    let mut rng = create_seeded_rng();

    const M: usize = 6;
    const LOG_XOR_REQUESTS: usize = 2;
    const LOG_NUM_REQUESTERS: usize = 1;

    const MAX_INPUT: u32 = 1 << M;
    const XOR_REQUESTS: usize = 1 << LOG_XOR_REQUESTS;
    const NUM_REQUESTERS: usize = 1 << LOG_NUM_REQUESTERS;

    let xor_chip = XorLookupChip::<M>::new(BYTE_XOR_BUS);

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
        .map(|_| DummyInteractionAir::new(3, true, BYTE_XOR_BUS))
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
                    .map(AbstractField::from_wrapped_u32)
                    .collect(),
                4,
            )
        })
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    let xor_trace = xor_chip.generate_trace();

    let mut all_chips: Vec<Arc<dyn AnyRap<_>>> = vec![];
    for requester in requesters {
        all_chips.push(Arc::new(requester));
    }
    all_chips.push(Arc::new(xor_chip.air));

    let all_traces = requesters_traces
        .into_iter()
        .chain(iter::once(xor_trace))
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(all_chips, all_traces)
        .expect("Verification failed");
}

#[test]
fn negative_test_xor_limbs_chip() {
    let mut rng = create_seeded_rng();

    const M: usize = 6;
    const LOG_XOR_REQUESTS: usize = 3;

    const MAX_INPUT: u32 = 1 << M;
    const XOR_REQUESTS: usize = 1 << LOG_XOR_REQUESTS;

    let xor_chip = XorLookupChip::<M>::new(BYTE_XOR_BUS);

    let pairs = (0..XOR_REQUESTS)
        .map(|_| {
            let x = rng.gen::<u32>() % MAX_INPUT;
            let y = rng.gen::<u32>() % MAX_INPUT;

            (1, vec![x, y])
        })
        .collect::<Vec<(u32, Vec<u32>)>>();

    let requester = DummyInteractionAir::new(3, true, BYTE_XOR_BUS);

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
            .map(AbstractField::from_wrapped_u32)
            .collect(),
        4,
    );

    let xor_trace = xor_chip.generate_trace();

    disable_debug_builder();
    let result = BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![requester, xor_chip.air],
        vec![requester_trace, xor_trace],
    );
    assert_eq!(
        result.err(),
        Some(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it passed"
    );
}
