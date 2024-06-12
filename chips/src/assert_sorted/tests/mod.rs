use std::sync::Arc;

use crate::range_gate::RangeCheckerGateChip;

use super::super::assert_sorted;

use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use assert_sorted::AssertSortedChip;
use p3_baby_bear::BabyBear;
use p3_matrix::dense::DenseMatrix;

/*
 * Testing strategy for the assert sorted chip:
 *     partition on limb_bits:
 *         limb_bits < 20
 *         limb_bits >= 20
 *     partition on key_vec_len:
 *         key_vec_len < 4
 *         key_vec_len >= 4
 *     partition on decomp:
 *         limb_bits % decomp == 0
 *         limb_bits % decomp != 0
 *     partition on number of rows:
 *         number of rows < 4
 *         number of rows >= 4
 *     partition on row order:
 *         rows are sorted lexicographically
 *         rows are not sorted lexicographically
 */

// covers limb_bits < 20, key_vec_len < 4, limb_bits % decomp == 0, number of rows < 4, rows are sorted lexicographically
#[test]
fn test_assert_sorted_chip_small_positive() {
    let bus_index: usize = 0;
    let limb_bits: Vec<usize> = vec![16, 16];
    let decomp: usize = 8;

    let range_max: u32 = 1 << decomp;

    let requests = vec![
        vec![7784, 35423],
        vec![17558, 44832],
        vec![22843, 12786],
        vec![32886, 24834],
    ];

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, range_max));

    let assert_sorted_chip = AssertSortedChip::new(
        bus_index,
        range_max,
        limb_bits,
        decomp,
        range_checker.clone(),
    );
    let range_checker_chip = assert_sorted_chip.range_checker.as_ref();

    let assert_sorted_chip_trace: DenseMatrix<BabyBear> =
        assert_sorted_chip.generate_trace(requests.clone());
    let range_checker_trace = assert_sorted_chip.range_checker.generate_trace();

    run_simple_test_no_pis(
        vec![&assert_sorted_chip.air, &range_checker_chip.air],
        vec![assert_sorted_chip_trace, range_checker_trace],
    )
    .expect("Verification failed");
}

// covers limb_bits >= 20, key_vec_len >= 4, limb_bits % decomp != 0, number of rows >= 4, rows are sorted lexicographically
#[test]
fn test_assert_sorted_chip_large_positive() {
    let bus_index: usize = 0;
    let limb_bits: Vec<usize> = vec![30, 30, 30, 30];
    let decomp: usize = 8;

    let range_max: u32 = 1 << decomp;

    let requests = vec![
        vec![44832, 12786, 318434, 35867],
        vec![487111, 42421, 369315, 704210],
        vec![783571, 729789, 37202, 370183],
        vec![887921, 196209, 767547, 875005],
    ];

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, range_max));

    let assert_sorted_chip = AssertSortedChip::new(
        bus_index,
        range_max,
        limb_bits,
        decomp,
        range_checker.clone(),
    );
    let range_checker_chip = assert_sorted_chip.range_checker.as_ref();

    let assert_sorted_chip_trace: DenseMatrix<BabyBear> =
        assert_sorted_chip.generate_trace(requests.clone());
    let range_checker_trace = assert_sorted_chip.range_checker.generate_trace();

    run_simple_test_no_pis(
        vec![&assert_sorted_chip.air, &range_checker_chip.air],
        vec![assert_sorted_chip_trace, range_checker_trace],
    )
    .expect("Verification failed");
}

// covers limb_bits >= 20, key_vec_len >= 4, limb_bits % decomp != 0, number of rows >= 4, rows are not sorted lexicographically
#[test]
fn test_assert_sorted_chip_unsorted_negative() {
    let bus_index: usize = 0;
    let limb_bits: Vec<usize> = vec![30, 30, 30, 30];
    let decomp: usize = 8;

    let range_max: u32 = 1 << decomp;

    // the first and second rows are not in sorted order
    let requests = vec![
        vec![44832, 42421, 369315, 704210],
        vec![44832, 12786, 318434, 35867],
        vec![783571, 729789, 37202, 370183],
        vec![887921, 196209, 767547, 875005],
    ];

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, range_max));

    let assert_sorted_chip = AssertSortedChip::new(
        bus_index,
        range_max,
        limb_bits,
        decomp,
        range_checker.clone(),
    );
    let range_checker_chip = assert_sorted_chip.range_checker.as_ref();

    let assert_sorted_chip_trace: DenseMatrix<BabyBear> =
        assert_sorted_chip.generate_trace(requests.clone());
    let range_checker_trace = assert_sorted_chip.range_checker.generate_trace();

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(
            vec![&assert_sorted_chip.air, &range_checker_chip.air],
            vec![assert_sorted_chip_trace, range_checker_trace],
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
