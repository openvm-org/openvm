use super::core::solve_shift;
use crate::arch::instructions::ShiftOpcode;

const RV32_NUM_LIMBS: usize = 4;
const RV32_LIMB_BITS: usize = 8;

#[test]
fn solve_sll_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [45, 7, 61, 186];
    let y: [u32; RV32_NUM_LIMBS] = [27, 0, 0, 0];
    let z: [u32; RV32_NUM_LIMBS] = [0, 0, 0, 104];
    let (result, limb_shift, bit_shift) =
        solve_shift::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(ShiftOpcode::SLL, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    assert_eq!((y[0] as usize) / RV32_LIMB_BITS, limb_shift);
    assert_eq!((y[0] as usize) % RV32_LIMB_BITS, bit_shift);
}

#[test]
fn solve_srl_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [31, 190, 221, 200];
    let y: [u32; RV32_NUM_LIMBS] = [17, 0, 0, 0];
    let z: [u32; RV32_NUM_LIMBS] = [110, 100, 0, 0];
    let (result, limb_shift, bit_shift) =
        solve_shift::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(ShiftOpcode::SRL, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    assert_eq!((y[0] as usize) / RV32_LIMB_BITS, limb_shift);
    assert_eq!((y[0] as usize) % RV32_LIMB_BITS, bit_shift);
}

#[test]
fn solve_sra_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [31, 190, 221, 200];
    let y: [u32; RV32_NUM_LIMBS] = [17, 0, 0, 0];
    let z: [u32; RV32_NUM_LIMBS] = [110, 228, 255, 255];
    let (result, limb_shift, bit_shift) =
        solve_shift::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(ShiftOpcode::SRA, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    assert_eq!((y[0] as usize) / RV32_LIMB_BITS, limb_shift);
    assert_eq!((y[0] as usize) % RV32_LIMB_BITS, bit_shift);
}
