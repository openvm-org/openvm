use super::page_controller;
use std::collections::{HashMap, HashSet};
use std::{iter, panic};

use afs_stark_backend::{
    keygen::{types::MultiStarkPartialProvingKey, MultiStarkKeygenBuilder},
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver, USE_DEBUG_BUILDER},
    verifier::VerificationError,
};
use afs_test_utils::{
    config::{
        self,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
    },
    engine::StarkEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

use crate::final_page;
use crate::group_by::group_by_input::GroupByChip;
use crate::group_by::page_controller::PageController;
use crate::page_rw_checker::{
    self,
    page_controller::{self, OpType, Operation},
};

#[test]
fn group_by_test() {
    let mut rng = create_seeded_rng();

    let internal_bus_index = 0;
    let output_bus_index = 1;
    let range_bus_index = 2;

    const MAX_VAL: usize = 0x78000001 / 2; // The prime used by BabyBear / 2

    let log_page_height = 4;

    let page_width = 8;
    let num_groups = rng.gen::<usize>() % (page_width - 2) + 1;
    let page_height = 1 << log_page_height;

    let idx_len = page_width - 2;
    let data_len = 1;
    let idx_limb_bits = 10;
    let idx_decomp = 4;
    let max_idx = 1 << idx_limb_bits;

    // Generating a random page
    let mut page: Vec<Vec<usize>> = vec![];
    for _ in 0..page_height {
        let mut idx;
        idx = (0..idx_len)
            .map(|_| rng.gen::<usize>() % max_idx)
            .collect::<Vec<usize>>();

        let data: Vec<usize> = (0..data_len)
            .map(|_| rng.gen::<usize>() % MAX_VAL)
            .collect();
        page.push(iter::once(1).chain(idx).chain(data).collect());
    }

    let page_F: Vec<Vec<BabyBear>> = page
        .iter()
        .map(|row| {
            row.iter()
                .map(|&x| BabyBear::from_canonical_u64(x as u64))
                .collect()
        })
        .collect();

    let page_u32: Vec<Vec<u32>> = page
        .iter()
        .map(|row| row.iter().map(|&x| x as u32).collect())
        .collect();

    let mut group_by_cols = vec![];
    while group_by_cols.len() < num_groups as usize {
        let col = rng.gen::<usize>() % page_width;
        if !group_by_cols.contains(&col) {
            group_by_cols.push(col);
        }
    }

    let aggregated_col = group_by_cols.pop().unwrap();

    let page_controller = PageController::new(
        page_width,
        group_by_cols,
        aggregated_col,
        internal_bus_index,
        output_bus_index,
        range_bus_index,
        idx_limb_bits,
        idx_decomp,
    );

    let group_by_trace = page_controller.group_by.generate_trace(page_F);
    let final_page_trace = page_controller
        .final_chip
        .gen_page_trace::<BabyBearPoseidon2Config>(page_u32.clone());
    let range_checker_trace = page_controller.range_checker.generate_trace();
    let final_page_aux_trace = page_controller
        .final_chip
        .gen_aux_trace::<BabyBearPoseidon2Config>(page_u32.clone(), page_controller.range_checker);

    let traces = vec![
        group_by_trace,
        final_page_trace,
        range_checker_trace,
        final_page_aux_trace,
    ];

    let chips = vec![
        page_controller.group_by,
        page_controller.final_chip,
        page_controller.range_checker,
    ];
}
