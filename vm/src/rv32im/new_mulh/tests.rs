use super::core::run_mulh;
use crate::arch::instructions::MulHOpcode;

const RV32_NUM_LIMBS: usize = 4;
const RV32_LIMB_BITS: usize = 8;

#[test]
fn run_mulh_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [197, 85, 150, 32];
    let y: [u32; RV32_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u32; RV32_NUM_LIMBS] = [130, 9, 135, 241];
    let z_mul: [u32; RV32_NUM_LIMBS] = [63, 247, 125, 232];
    let (res, res_mul, x_ext, y_ext) =
        run_mulh::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(MulHOpcode::MULH, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
    }
    assert_eq!(x_ext, 0);
    assert_eq!(y_ext, 255);
}

#[test]
fn run_mulhu_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [197, 85, 150, 32];
    let y: [u32; RV32_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u32; RV32_NUM_LIMBS] = [71, 95, 29, 18];
    let z_mul: [u32; RV32_NUM_LIMBS] = [63, 247, 125, 232];
    let (res, res_mul, x_ext, y_ext) =
        run_mulh::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(MulHOpcode::MULHU, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
    }
    assert_eq!(x_ext, 0);
    assert_eq!(y_ext, 0);
}

#[test]
fn run_mulhsu_pos_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [197, 85, 150, 32];
    let y: [u32; RV32_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u32; RV32_NUM_LIMBS] = [71, 95, 29, 18];
    let z_mul: [u32; RV32_NUM_LIMBS] = [63, 247, 125, 232];
    let (res, res_mul, x_ext, y_ext) =
        run_mulh::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(MulHOpcode::MULHSU, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
    }
    assert_eq!(x_ext, 0);
    assert_eq!(y_ext, 0);
}

#[test]
fn run_mulhsu_neg_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [197, 85, 150, 160];
    let y: [u32; RV32_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u32; RV32_NUM_LIMBS] = [174, 40, 246, 202];
    let z_mul: [u32; RV32_NUM_LIMBS] = [63, 247, 125, 104];
    let (res, res_mul, x_ext, y_ext) =
        run_mulh::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(MulHOpcode::MULHSU, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
    }
    assert_eq!(x_ext, 255);
    assert_eq!(y_ext, 0);
}
