use std::{array, borrow::Borrow};

use afs_primitives::{utils, xor::bus::XorBus};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::UintArithmeticCols;
use crate::{
    arch::{bridge::ExecutionBridge, instructions::UINT256_ARITHMETIC_INSTRUCTIONS},
    memory::offline_checker::MemoryBridge,
};

#[derive(Copy, Clone, Debug)]
pub struct UintArithmeticAir<const ARG_SIZE: usize, const LIMB_SIZE: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub bus: XorBus,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for UintArithmeticAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        UintArithmeticCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}

impl<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize> Air<AB>
    for UintArithmeticAir<NUM_LIMBS, LIMB_BITS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);

        let UintArithmeticCols::<_, NUM_LIMBS, LIMB_BITS> { io, aux } = (*local).borrow();
        builder.assert_bool(aux.is_valid);

        let flags = [
            aux.opcode_add_flag,
            aux.opcode_sub_flag,
            aux.opcode_lt_flag,
            aux.opcode_eq_flag,
            aux.opcode_xor_flag,
            aux.opcode_and_flag,
            aux.opcode_or_flag,
            aux.opcode_slt_flag,
        ];
        for flag in flags {
            builder.assert_bool(flag);
        }

        builder.assert_eq(
            aux.is_valid,
            flags
                .iter()
                .fold(AB::Expr::zero(), |acc, &flag| acc + flag.into()),
        );

        let x_limbs = &io.x.data;
        let y_limbs = &io.y.data;
        let z_limbs = &io.z.data;

        // For ADD, define carry[i] = (x[i] + y[i] + carry[i - 1] - z[i]) / 2^LIMB_BITS. If
        // each carry[i] is boolean and 0 <= z[i] < 2^NUM_LIMBS, it can be proven that
        // z[i] = (x[i] + y[i]) % 256 as necessary. The same holds for SUB when carry[i] is
        // (z[i] + y[i] - x[i] + carry[i - 1]) / 2^LIMB_BITS.
        let mut carry_add: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::zero());
        let mut carry_sub: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::zero());
        let carry_divide = AB::F::from_canonical_usize(1 << LIMB_BITS).inverse();

        for i in 0..NUM_LIMBS {
            // We explicitly separate the constraints for ADD and SUB in order to keep degree
            // cubic. Because we constrain that the carry is bool, if carry has degree larger
            // than 1 the max-degree constrain is automatically at least 4.
            carry_add[i] = AB::Expr::from(carry_divide)
                * (x_limbs[i] + y_limbs[i] - z_limbs[i]
                    + if i > 0 {
                        carry_add[i - 1].clone()
                    } else {
                        AB::Expr::zero()
                    });
            builder
                .when(aux.opcode_add_flag)
                .assert_bool(carry_add[i].clone());
            carry_sub[i] = AB::Expr::from(carry_divide)
                * (z_limbs[i] + y_limbs[i] - x_limbs[i]
                    + if i > 0 {
                        carry_sub[i - 1].clone()
                    } else {
                        AB::Expr::zero()
                    });
            builder
                .when(aux.opcode_sub_flag + aux.opcode_lt_flag + aux.opcode_slt_flag)
                .assert_bool(carry_sub[i].clone());
        }

        // For LT, cmp_result must be equal to the last carry. For SLT, cmp_result ^ x_sign ^ y_sign must
        // be equal to the last carry. To ensure maximum cubic degree constraints, we set aux.x_msb_masked
        // and aux.y_msb_masked such that x_sign and y_sign are 0 when not computing an SLT.
        let sign_divide = AB::F::from_canonical_usize(1 << (LIMB_BITS - 1)).inverse();
        let x_sign = AB::Expr::from(sign_divide) * (x_limbs[NUM_LIMBS - 1] - aux.x_msb_masked);
        let y_sign = AB::Expr::from(sign_divide) * (y_limbs[NUM_LIMBS - 1] - aux.y_msb_masked);

        builder.assert_bool(x_sign.clone());
        builder.assert_bool(y_sign.clone());
        builder
            .when(utils::not(aux.opcode_slt_flag))
            .assert_zero(x_sign.clone());
        builder
            .when(utils::not(aux.opcode_slt_flag))
            .assert_zero(y_sign.clone());

        let slt_xor = (aux.opcode_lt_flag + aux.opcode_slt_flag) * io.cmp_result
            + x_sign.clone()
            + y_sign.clone()
            - AB::Expr::from_canonical_u32(2)
                * (io.cmp_result * x_sign.clone()
                    + io.cmp_result * y_sign.clone()
                    + x_sign.clone() * y_sign.clone())
            + AB::Expr::from_canonical_u32(4) * (io.cmp_result * x_sign.clone() * y_sign.clone());
        builder.assert_zero(
            slt_xor - (aux.opcode_lt_flag + aux.opcode_slt_flag) * carry_sub[NUM_LIMBS - 1].clone(),
        );

        // For EQ, z is filled with 0 except at the lowest index i such that x[i] != y[i]. If
        // such an i exists z[i] is the inverse of x[i] - y[i], meaning sum_eq should be 1.
        let mut sum_eq: AB::Expr = io.cmp_result.into();
        for i in 0..NUM_LIMBS {
            sum_eq += (x_limbs[i] - y_limbs[i]) * z_limbs[i];
            builder
                .when(aux.opcode_eq_flag)
                .assert_zero(io.cmp_result * (x_limbs[i] - y_limbs[i]));
        }
        builder
            .when(aux.opcode_eq_flag)
            .assert_zero(sum_eq - AB::Expr::one());

        let expected_opcode = flags
            .iter()
            .zip(UINT256_ARITHMETIC_INSTRUCTIONS)
            .fold(AB::Expr::zero(), |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
            });

        self.eval_interactions(builder, io, aux, expected_opcode);
    }
}
