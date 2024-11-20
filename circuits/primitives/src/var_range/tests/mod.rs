use std::{iter, sync::Arc};

use ax_stark_backend::{prover::USE_DEBUG_BUILDER, rap::AnyRap, verifier::VerificationError};
use ax_stark_sdk::{
    any_rap_arc_vec, config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkFriEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use rand::Rng;

use crate::var_range::{
    bus::VariableRangeCheckerBus,
    tests::dummy_airs::{TestRangeCheckAir, TestSendAir},
    VariableRangeCheckerChip,
};

pub mod dummy_airs;

#[test]
fn test_variable_range_checker_chip_send() {
    let mut rng = create_seeded_rng();

    const MAX_BITS: u32 = 3;
    const LOG_LIST_LEN: usize = 8;
    const LIST_LEN: usize = 1 << LOG_LIST_LEN;

    let bus = VariableRangeCheckerBus::new(0, MAX_BITS as usize);
    let var_range_checker = VariableRangeCheckerChip::new(bus);

    // generate lists of randomized valid values-bits pairs
    let num_lists = 10;
    let lists_vals = (0..num_lists)
        .map(|_| {
            (0..LIST_LEN)
                .map(|_| {
                    let bits = rng.gen_range(0..=MAX_BITS);
                    let val = rng.gen_range(0..(1 << bits));
                    [val, bits]
                })
                .collect::<Vec<[u32; 2]>>()
        })
        .collect::<Vec<Vec<[u32; 2]>>>();

    // generate dummy AIR chips for each list
    let lists_airs = (0..num_lists)
        .map(|_| TestSendAir::new(bus))
        .collect::<Vec<TestSendAir>>();

    let mut all_chips = lists_airs
        .into_iter()
        .map(|list| Arc::new(list) as Arc<dyn AnyRap<_>>)
        .collect::<Vec<_>>();
    all_chips.push(Arc::new(var_range_checker.air));

    // generate traces for each list
    let lists_traces = lists_vals
        .par_iter()
        .map(|list| {
            RowMajorMatrix::new(
                list.iter()
                    .flat_map(|&[val, bits]| {
                        var_range_checker.add_count(val, bits as usize);
                        iter::once(val).chain(iter::once(bits))
                    })
                    .map(AbstractField::from_canonical_u32)
                    .collect(),
                2,
            )
        })
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    let var_range_checker_trace = var_range_checker.generate_trace();

    let all_traces = lists_traces
        .into_iter()
        .chain(iter::once(var_range_checker_trace))
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(all_chips, all_traces)
        .expect("Verification failed");
}

#[test]
fn negative_test_variable_range_checker_chip_send() {
    // test that the constraint fails when some val >= 2^max_bits
    let mut rng = create_seeded_rng();

    const MAX_BITS: u32 = 3;
    const LOG_LIST_LEN: usize = 8;
    const LIST_LEN: usize = 1 << LOG_LIST_LEN;

    let bus = VariableRangeCheckerBus::new(0, MAX_BITS as usize);
    let var_range_checker = VariableRangeCheckerChip::new(bus);

    // generate randomized valid values-bits pairs with one invalid pair (i.e. [4, 2])
    let list_vals = (0..(LIST_LEN - 1))
        .map(|_| {
            let bits = rng.gen_range(0..=MAX_BITS);
            let val = rng.gen_range(0..(1 << bits));
            [val, bits]
        })
        .chain(iter::once([4, 2]))
        .collect::<Vec<[u32; 2]>>();

    // generate dummy AIR chip
    let list_chip = TestSendAir::new(bus);
    let all_chips = any_rap_arc_vec![list_chip, var_range_checker.air];

    // generate trace with a [val, bits] pair such that val >= 2^bits (i.e. [4, 2])
    let list_trace = RowMajorMatrix::new(
        list_vals
            .iter()
            .flat_map(|&[val, bits]| {
                var_range_checker.add_count(val, bits as usize);
                iter::once(val).chain(iter::once(bits))
            })
            .map(AbstractField::from_canonical_u32)
            .collect(),
        2,
    );
    let var_range_trace = var_range_checker.generate_trace();
    let all_traces = vec![list_trace, var_range_trace];

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        BabyBearBlake3Engine::run_simple_test_no_pis_fast(all_chips, all_traces).err(),
        Some(VerificationError::ChallengePhaseError),
        "Expected constraint to fail"
    );
}

#[test]
fn test_variable_range_checker_chip_range_check() {
    let mut rng = create_seeded_rng();

    const MAX_BITS: usize = 3;
    const MAX_VAL: u32 = 1 << MAX_BITS;
    const LOG_LIST_LEN: usize = 6;
    const LIST_LEN: usize = 1 << LOG_LIST_LEN;

    // test case where constant max_bits < range_max_bits
    let bus = VariableRangeCheckerBus::new(0, MAX_BITS + 1);
    let var_range_checker = VariableRangeCheckerChip::new(bus);

    // generate lists of randomized valid values
    let num_lists = 10;
    let lists_vals = (0..num_lists)
        .map(|_| {
            (0..LIST_LEN)
                .map(|_| rng.gen_range(0..MAX_VAL))
                .collect::<Vec<u32>>()
        })
        .collect::<Vec<Vec<u32>>>();

    // generate dummy AIR chips for each list
    let lists_airs = (0..num_lists)
        .map(|_| TestRangeCheckAir::new(bus, MAX_BITS))
        .collect::<Vec<TestRangeCheckAir>>();

    let mut all_chips = lists_airs
        .into_iter()
        .map(|list| Arc::new(list) as Arc<dyn AnyRap<_>>)
        .collect::<Vec<_>>();
    all_chips.push(Arc::new(var_range_checker.air));

    // generate traces for each list
    let lists_traces = lists_vals
        .par_iter()
        .map(|list| {
            RowMajorMatrix::new(
                list.iter()
                    .flat_map(|&val| {
                        var_range_checker.add_count(val, MAX_BITS);
                        iter::once(val)
                    })
                    .map(AbstractField::from_canonical_u32)
                    .collect(),
                1,
            )
        })
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    let var_range_checker_trace = var_range_checker.generate_trace();

    let all_traces = lists_traces
        .into_iter()
        .chain(iter::once(var_range_checker_trace))
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(all_chips, all_traces)
        .expect("Verification failed");
}

#[test]
fn negative_test_variable_range_checker_chip_range_check() {
    // test that the constraint fails when some val >= 2^max_bits
    let mut rng = create_seeded_rng();

    const MAX_BITS: usize = 3;
    const MAX_VAL: u32 = 1 << MAX_BITS;
    const LOG_LIST_LEN: usize = 6;
    const LIST_LEN: usize = 1 << LOG_LIST_LEN;

    // test case where constant max_bits < range_max_bits
    let bus = VariableRangeCheckerBus::new(0, MAX_BITS + 1);
    let var_range_checker = VariableRangeCheckerChip::new(bus);

    // generate randomized valid values with one invalid value (i.e. MAX_VAL)
    let list_vals = (0..(LIST_LEN - 1))
        .map(|_| rng.gen_range(0..MAX_VAL))
        .chain(iter::once(MAX_VAL))
        .collect::<Vec<u32>>();

    // generate dummy AIR chip
    let list_chip = TestRangeCheckAir::new(bus, MAX_BITS);
    let all_chips = any_rap_arc_vec![list_chip, var_range_checker.air];

    // generate trace with one value >= 2^max_bits (i.e. MAX_VAL)
    let list_trace = RowMajorMatrix::new(
        list_vals
            .iter()
            .flat_map(|&val| {
                var_range_checker.add_count(val, MAX_BITS);
                iter::once(val)
            })
            .map(AbstractField::from_canonical_u32)
            .collect(),
        1,
    );
    let var_range_trace = var_range_checker.generate_trace();
    let all_traces = vec![list_trace, var_range_trace];

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        BabyBearBlake3Engine::run_simple_test_no_pis_fast(all_chips, all_traces).err(),
        Some(VerificationError::ChallengePhaseError),
        "Expected constraint to fail"
    );
}
