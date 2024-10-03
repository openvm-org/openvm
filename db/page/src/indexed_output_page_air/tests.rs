use std::{collections::HashSet, iter, sync::Arc};

use afs_primitives::var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip};
use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder, prover::USE_DEBUG_BUILDER, verifier::VerificationError,
};
use ax_sdk::{
    any_rap_vec,
    config::{
        self,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
    },
    engine::{StarkFriEngine, VerificationDataWithFriParams},
    utils::create_seeded_rng,
};
use rand::Rng;

use crate::common::page::Page;

fn test_single_page(
    page: &Page,
    final_page_chip: &super::IndexedOutputPageAir,
    range_checker: Arc<VariableRangeCheckerChip>,
) -> Result<VerificationDataWithFriParams<BabyBearPoseidon2Config>, VerificationError> {
    let page_trace = final_page_chip.gen_page_trace::<BabyBearPoseidon2Config>(page);
    let aux_trace =
        final_page_chip.gen_aux_trace::<BabyBearPoseidon2Config>(page, range_checker.clone());
    let range_checker_trace = range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_test_no_pis(
        &any_rap_vec![final_page_chip, &range_checker.air],
        vec![
            vec![page_trace.clone(), aux_trace.clone()],
            vec![range_checker_trace.clone()],
        ],
    )
}

#[test]
fn final_page_chip_test() {
    let mut rng = create_seeded_rng();
    let range_bus_index = 0;

    use super::IndexedOutputPageAir;

    let log_page_height = 3;

    let page_width = 6;
    let page_height = 1 << log_page_height;

    let idx_len = rng.gen::<usize>() % ((page_width - 1) - 1) + 1;
    let data_len = (page_width - 1) - idx_len;

    let idx_limb_bits = 5;
    let idx_decomp = 2;

    let max_idx: u32 = 1 << idx_limb_bits;

    let allocated_rows = ((page_height as f64) * (3.0 / 4.0)) as usize;

    // Creating a list of sorted distinct indices
    let mut all_idx = HashSet::new();
    while all_idx.len() < allocated_rows {
        all_idx.insert(
            (0..idx_len)
                .map(|_| (rng.gen::<u32>() % max_idx))
                .collect::<Vec<u32>>(),
        );
    }
    let mut all_idx: Vec<Vec<u32>> = all_idx.into_iter().collect();
    all_idx.sort();

    let page: Vec<Vec<u32>> = (0..page_height)
        .map(|x| {
            if x < allocated_rows {
                iter::once(1)
                    .chain(all_idx[x].iter().cloned())
                    .chain((0..data_len).map(|_| (rng.gen::<u32>() % max_idx)))
                    .collect()
            } else {
                vec![0; idx_len + data_len + 1]
            }
        })
        .collect();

    let page = Page::from_2d_vec(&page, idx_len, data_len);

    let final_page_chip = IndexedOutputPageAir::new(
        range_bus_index,
        idx_len,
        data_len,
        idx_limb_bits,
        idx_decomp,
    );
    let range_bus = VariableRangeCheckerBus::new(range_bus_index, idx_decomp);
    let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(idx_decomp));

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    let page_data_ptr = keygen_builder.add_cached_main_matrix(final_page_chip.page_width());
    let page_aux_ptr = keygen_builder.add_main_matrix(final_page_chip.aux_width());
    keygen_builder.add_partitioned_air(&final_page_chip, vec![page_data_ptr, page_aux_ptr]);

    keygen_builder.add_air(&range_checker.air);

    test_single_page(&page, &final_page_chip, range_checker.clone()).expect("Verification Failed");

    // Creating a new page with the first two rows swapped
    let mut page_rows = page.to_2d_vec();
    page_rows.swap(0, 1);
    let page = Page::from_2d_vec(&page_rows, idx_len, data_len);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });

    assert!(
        matches!(
            test_single_page(&page, &final_page_chip, range_checker.clone()),
            Err(VerificationError::OodEvaluationMismatch),
        ),
        "Expected constraints to fail"
    );
}
