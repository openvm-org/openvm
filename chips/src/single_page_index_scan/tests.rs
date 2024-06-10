use std::sync::Arc;

use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;

use super::{
    page_index_scan_input::PageIndexScanInputChip, page_index_scan_output::PageIndexScanOutputChip,
};
use crate::range_gate::RangeCheckerGateChip;

#[test]
fn test_single_page_index_scan() {
    let bus_index: usize = 0;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: Vec<usize> = vec![16, 16];
    let range_max: u32 = 1 << decomp;

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, range_max));

    let page_index_scan_input_chip = PageIndexScanInputChip::new(
        bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits.clone(),
        decomp,
        range_checker.clone(),
    );
    let page_index_scan_output_chip = PageIndexScanOutputChip::new(
        bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits.clone(),
        decomp,
        range_checker.clone(),
    );
    let range_checker_chip = range_checker.as_ref();

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];

    let page_indexed: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![0, 0, 0, 0, 0, 0],
    ];

    let x: Vec<u32> = vec![2177, 5880];

    let page_index_scan_chip_trace = page_index_scan_input_chip.generate_trace(page.clone(), x);
    let page_index_scan_verify_chip_trace =
        page_index_scan_output_chip.generate_trace(page_indexed.clone());
    let range_checker_trace = range_checker_chip.generate_trace();

    run_simple_test_no_pis(
        vec![
            &page_index_scan_input_chip.air,
            &page_index_scan_output_chip.air,
            &range_checker_chip.air,
        ],
        vec![
            page_index_scan_chip_trace,
            page_index_scan_verify_chip_trace,
            range_checker_trace,
        ],
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_wrong_order() {
    let bus_index: usize = 0;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: Vec<usize> = vec![16, 16];
    let range_max: u32 = 1 << decomp;

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, range_max));

    let page_index_scan_input_chip = PageIndexScanInputChip::new(
        bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits.clone(),
        decomp,
        range_checker.clone(),
    );
    let page_index_scan_output_chip = PageIndexScanOutputChip::new(
        bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits.clone(),
        decomp,
        range_checker.clone(),
    );
    let range_checker_chip = range_checker.as_ref();

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];

    let page_indexed: Vec<Vec<u32>> = vec![
        vec![0, 0, 0, 0, 0, 0],
        vec![1, 443, 376, 22278, 13998, 58327],
    ];

    let x: Vec<u32> = vec![2177, 5880];

    let page_index_scan_chip_trace = page_index_scan_input_chip.generate_trace(page.clone(), x);
    let page_index_scan_verify_chip_trace =
        page_index_scan_output_chip.generate_trace(page_indexed.clone());
    let range_checker_trace = range_checker_chip.generate_trace();

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(
            vec![
                &page_index_scan_input_chip.air,
                &page_index_scan_output_chip.air,
                &range_checker_chip.air,
            ],
            vec![
                page_index_scan_chip_trace,
                page_index_scan_verify_chip_trace,
                range_checker_trace,
            ],
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_single_page_index_scan_unsorted() {
    let bus_index: usize = 0;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: Vec<usize> = vec![16, 16];
    let range_max: u32 = 1 << decomp;

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, range_max));

    let page_index_scan_input_chip = PageIndexScanInputChip::new(
        bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits.clone(),
        decomp,
        range_checker.clone(),
    );
    let page_index_scan_output_chip = PageIndexScanOutputChip::new(
        bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits.clone(),
        decomp,
        range_checker.clone(),
    );
    let range_checker_chip = range_checker.as_ref();

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];

    let page_indexed: Vec<Vec<u32>> = vec![
        vec![1, 2883, 7769, 51171, 3989, 12770],
        vec![1, 443, 376, 22278, 13998, 58327],
    ];

    let x: Vec<u32> = vec![2883, 7770];

    let page_index_scan_chip_trace = page_index_scan_input_chip.generate_trace(page.clone(), x);
    let page_index_scan_verify_chip_trace =
        page_index_scan_output_chip.generate_trace(page_indexed.clone());
    let range_checker_trace = range_checker_chip.generate_trace();

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(
            vec![
                &page_index_scan_input_chip.air,
                &page_index_scan_output_chip.air,
                &range_checker_chip.air,
            ],
            vec![
                page_index_scan_chip_trace,
                page_index_scan_verify_chip_trace,
                range_checker_trace,
            ],
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
