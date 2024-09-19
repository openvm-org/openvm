use p3_baby_bear::BabyBear;

use super::UiChip;

#[test]
fn solve_lui_sanity_test() {
    let b = 10;
    let x = UiChip::<BabyBear>::solve_lui(b);
    assert_eq!(x, [0, 0, 160, 0]);
}
