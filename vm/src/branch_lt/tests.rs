use std::sync::Arc;

use afs_primitives::xor::lookup::XorLookupChip;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::integration::{solve_cmp, BranchLessThanIntegration};
use crate::{
    arch::{
        instructions::{BranchLessThanOpcode, UsizeOpcode},
        VmIntegration, Rv32BranchAdapter,
    },
    core::BYTE_XOR_BUS,
    program::Instruction,
};

const RV32_NUM_LIMBS: usize = 4;
const RV32_LIMB_BITS: usize = 8;
type F = BabyBear;

#[test]
fn execute_pc_increment_sanity_test() {
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_LIMB_BITS>::new(BYTE_XOR_BUS));
    let integration =
        BranchLessThanIntegration::<RV32_NUM_LIMBS, RV32_LIMB_BITS>::new(xor_lookup_chip, 0);

    let mut instruction = Instruction::<F> {
        opcode: BranchLessThanOpcode::BLT.as_usize(),
        op_c: F::from_canonical_u8(8),
        ..Default::default()
    };
    let x: [F; RV32_NUM_LIMBS] = [145, 34, 25, 205].map(F::from_canonical_u32);

    let result =
        <BranchLessThanIntegration<RV32_NUM_LIMBS, RV32_LIMB_BITS> as VmIntegration<
            F,
            Rv32BranchAdapter<F>,
        >>::execute_instruction(&integration, &instruction, F::zero(), [x, x]);
    let (output, _) = result.expect("execute_instruction failed");
    assert!(output.to_pc.is_none());

    instruction.opcode = BranchLessThanOpcode::BGE.as_usize();
    let result =
        <BranchLessThanIntegration<RV32_NUM_LIMBS, RV32_LIMB_BITS> as VmIntegration<
            F,
            Rv32BranchAdapter<F>,
        >>::execute_instruction(&integration, &instruction, F::zero(), [x, x]);
    let (output, _) = result.expect("execute_instruction failed");
    assert!(output.to_pc.is_some());
    assert_eq!(output.to_pc.unwrap(), F::from_canonical_u8(8));
}

#[test]
fn solve_cmp_unsigned_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [145, 34, 25, 205];
    let y: [u32; RV32_NUM_LIMBS] = [73, 35, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_cmp::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(BranchLessThanOpcode::BLTU, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned

    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_cmp::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(BranchLessThanOpcode::BGEU, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned
}

#[test]
fn solve_cmp_same_sign_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [145, 34, 25, 205];
    let y: [u32; RV32_NUM_LIMBS] = [73, 35, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_cmp::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(BranchLessThanOpcode::BLT, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(x_sign); // negative
    assert!(y_sign); // negative

    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_cmp::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(BranchLessThanOpcode::BGE, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(x_sign); // negative
    assert!(y_sign); // negative
}

#[test]
fn solve_cmp_diff_sign_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [45, 35, 25, 55];
    let y: [u32; RV32_NUM_LIMBS] = [173, 34, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_cmp::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(BranchLessThanOpcode::BLT, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, 3);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative

    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_cmp::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(BranchLessThanOpcode::BGE, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 3);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative
}
