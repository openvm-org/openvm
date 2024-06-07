use std::sync::Arc;

use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;

use super::{page_index_scan::PageIndexScanChip, page_index_scan_verify::PageIndexScanVerifyChip};
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

    let page_index_scan_chip = PageIndexScanChip::new(
        bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits,
        decomp,
        range_checker,
    );
    let page_index_scan_verify_chip = PageIndexScanVerifyChip::new(bus_index, idx_len, data_len);
    let range_checker = page_index_scan_chip.range_checker.as_ref();

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];

    let page_indexed: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![0, 0, 0, 0, 0, 0],
    ];

    let x: Vec<u32> = vec![2177, 5880];

    let page_index_scan_chip_trace = page_index_scan_chip.generate_trace(page.clone(), x);
    let page_index_scan_verify_chip_trace =
        page_index_scan_verify_chip.generate_trace(page_indexed.clone());
    let range_checker_trace = page_index_scan_chip.range_checker.generate_trace();

    run_simple_test_no_pis(
        vec![
            &page_index_scan_chip.air,
            &page_index_scan_verify_chip.air,
            range_checker,
        ],
        vec![
            page_index_scan_chip_trace,
            page_index_scan_verify_chip_trace,
            range_checker_trace,
        ],
    )
    .expect("Verification failed");
}
