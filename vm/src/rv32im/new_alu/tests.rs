use super::core::solve_alu;
use crate::arch::instructions::AluOpcode;

const RV32_NUM_LIMBS: usize = 4;
const RV32_LIMB_BITS: usize = 8;

#[test]
fn solve_add_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_NUM_LIMBS] = [23, 205, 73, 49];
    let result = solve_alu::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(AluOpcode::ADD, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn solve_sub_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_NUM_LIMBS] = [179, 118, 240, 172];
    let result = solve_alu::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(AluOpcode::SUB, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn solve_xor_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_NUM_LIMBS] = [215, 138, 49, 173];
    let result = solve_alu::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(AluOpcode::XOR, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn solve_or_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_NUM_LIMBS] = [247, 171, 61, 239];
    let result = solve_alu::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(AluOpcode::OR, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn solve_and_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_NUM_LIMBS] = [32, 33, 12, 66];
    let result = solve_alu::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(AluOpcode::AND, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}
