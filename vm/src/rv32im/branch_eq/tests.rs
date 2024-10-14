use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::core::{solve_eq, BranchEqualCoreChip};
use crate::{
    arch::{
        instructions::{BranchEqualOpcode, UsizeOpcode},
        VmCoreChip,
    },
    rv32im::adapters::Rv32BranchAdapter,
    system::program::Instruction,
};

const RV32_NUM_LIMBS: usize = 4;
type F = BabyBear;

#[test]
fn execute_pc_increment_sanity_test() {
    let core = BranchEqualCoreChip::<RV32_NUM_LIMBS>::new(0);

    let mut instruction = Instruction::<F> {
        opcode: BranchEqualOpcode::BEQ.as_usize(),
        op_c: F::from_canonical_u8(8),
        ..Default::default()
    };
    let x: [F; RV32_NUM_LIMBS] = [19, 4, 1790, 60].map(F::from_canonical_u32);
    let y: [F; RV32_NUM_LIMBS] = [19, 32, 1804, 60].map(F::from_canonical_u32);

    let result =
        <BranchEqualCoreChip<RV32_NUM_LIMBS> as VmCoreChip<F, Rv32BranchAdapter<F>>>::execute_instruction(
            &core,
            &instruction,
            F::zero(),
            [x, y],
        );
    let (output, _) = result.expect("execute_instruction failed");
    assert!(output.to_pc.is_none());

    instruction.opcode = BranchEqualOpcode::BNE.as_usize();
    let result =
        <BranchEqualCoreChip<RV32_NUM_LIMBS> as VmCoreChip<F, Rv32BranchAdapter<F>>>::execute_instruction(
            &core,
            &instruction,
            F::zero(),
            [x, y],
        );
    let (output, _) = result.expect("execute_instruction failed");
    assert!(output.to_pc.is_some());
    assert_eq!(output.to_pc.unwrap(), F::from_canonical_u8(8));
}

#[test]
fn solve_eq_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [19, 4, 1790, 60];
    let (cmp_result, _, diff_val) = solve_eq::<F, RV32_NUM_LIMBS>(BranchEqualOpcode::BEQ, &x, &x);
    assert!(cmp_result);
    assert_eq!(diff_val, F::zero());

    let (cmp_result, _, diff_val) = solve_eq::<F, RV32_NUM_LIMBS>(BranchEqualOpcode::BNE, &x, &x);
    assert!(!cmp_result);
    assert_eq!(diff_val, F::zero());
}

#[test]
fn solve_ne_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [19, 4, 1790, 60];
    let y: [u32; RV32_NUM_LIMBS] = [19, 32, 1804, 60];
    let (cmp_result, diff_idx, diff_val) =
        solve_eq::<F, RV32_NUM_LIMBS>(BranchEqualOpcode::BEQ, &x, &y);
    assert!(!cmp_result);
    assert_eq!(
        diff_val * (F::from_canonical_u32(x[diff_idx]) - F::from_canonical_u32(y[diff_idx])),
        F::one()
    );

    let (cmp_result, diff_idx, diff_val) =
        solve_eq::<F, RV32_NUM_LIMBS>(BranchEqualOpcode::BNE, &x, &y);
    assert!(cmp_result);
    assert_eq!(
        diff_val * (F::from_canonical_u32(x[diff_idx]) - F::from_canonical_u32(y[diff_idx])),
        F::one()
    );
}
