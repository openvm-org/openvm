use super::core::solve_less_than;
use crate::arch::instructions::LessThanOpcode;

const RV32_NUM_LIMBS: usize = 4;
const RV32_LIMB_BITS: usize = 8;

#[test]
fn solve_sltu_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [145, 34, 25, 205];
    let y: [u32; RV32_NUM_LIMBS] = [73, 35, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_less_than::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(LessThanOpcode::SLTU, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned
}

#[test]
fn solve_slt_same_sign_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [145, 34, 25, 205];
    let y: [u32; RV32_NUM_LIMBS] = [73, 35, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_less_than::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(LessThanOpcode::SLT, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(x_sign); // negative
    assert!(y_sign); // negative
}

#[test]
fn solve_slt_diff_sign_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [45, 35, 25, 55];
    let y: [u32; RV32_NUM_LIMBS] = [173, 34, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_less_than::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(LessThanOpcode::SLT, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, 3);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative
}
