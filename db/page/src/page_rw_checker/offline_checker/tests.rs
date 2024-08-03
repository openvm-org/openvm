use super::{columns::PageOfflineCheckerCols, PageOfflineChecker};

#[test]
fn offline_checker_from_slice_test() {
    let offline_checker = PageOfflineChecker::new(0, 0, 0, 16, 32, 4, 64, 2);

    let all_cols = (0..offline_checker.air_width()).collect::<Vec<usize>>();
    let cols = PageOfflineCheckerCols::<usize>::from_slice(&all_cols, &offline_checker);

    assert!(cols.flatten() == all_cols);
}
