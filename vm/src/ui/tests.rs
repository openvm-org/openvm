use super::UiChip;

use p3_baby_bear::BabyBear;

#[test]
fn solve_lui_sanity_test() {
    let b = 10;
    let x = UiChip::<BabyBear>::solve_lui(b);
    assert_eq!(x, [40960]);
}
