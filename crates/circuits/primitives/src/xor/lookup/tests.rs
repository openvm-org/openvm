use std::{iter, sync::Arc};

use openvm_stark_backend::{
    interaction::BusIndex,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
    poseidon2::sponge::DuplexSponge,
    prover::AirProvingContext,
    test_utils::{test_engine_small, test_system_params_small},
    utils::disable_debug_builder,
    AirRef, BabyBearPoseidon2CpuEngine, StarkEngine,
};
use openvm_stark_sdk::{
    any_rap_arc_vec, dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    p3_baby_bear::BabyBear, utils::create_seeded_rng,
};
use rand::Rng;

use crate::xor::XorLookupChip;

const BYTE_XOR_BUS: BusIndex = 10;

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
                    let x = rng.random::<u32>() % MAX_INPUT;
                    let y = rng.random::<u32>() % MAX_INPUT;

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
                    .map(PrimeCharacteristicRing::from_u32)
                    .collect(),
                4,
            )
        })
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    let xor_trace = xor_chip.generate_trace();

    let mut all_chips: Vec<AirRef<_>> = vec![];
    for requester in requesters {
        all_chips.push(Arc::new(requester));
    }
    all_chips.push(Arc::new(xor_chip.air));

    let all_traces = requesters_traces
        .into_iter()
        .chain(iter::once(xor_trace))
        .map(Arc::new)
        .map(AirProvingContext::simple_no_pis)
        .map(AirProvingContext::from_v1_no_cached)
        .collect::<Vec<_>>();

    BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(test_system_params_small(3, 9, 3))
        .run_test(all_chips, all_traces)
        .expect("Verification failed");
}

#[test]
#[should_panic]
fn negative_test_xor_limbs_chip() {
    let mut rng = create_seeded_rng();

    const M: usize = 6;
    const LOG_XOR_REQUESTS: usize = 3;

    const MAX_INPUT: u32 = 1 << M;
    const XOR_REQUESTS: usize = 1 << LOG_XOR_REQUESTS;

    let xor_chip = XorLookupChip::<M>::new(BYTE_XOR_BUS);

    let pairs = (0..XOR_REQUESTS)
        .map(|_| {
            let x = rng.random::<u32>() % MAX_INPUT;
            let y = rng.random::<u32>() % MAX_INPUT;

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
            .map(PrimeCharacteristicRing::from_u32)
            .collect(),
        4,
    );

    let xor_trace = xor_chip.generate_trace();

    let traces = [requester_trace, xor_trace]
        .into_iter()
        .map(Arc::new)
        .map(AirProvingContext::simple_no_pis)
        .map(AirProvingContext::from_v1_no_cached)
        .collect::<Vec<_>>();

    disable_debug_builder();
    test_engine_small()
        .run_test(any_rap_arc_vec![requester, xor_chip.air], traces)
        .unwrap();
}
