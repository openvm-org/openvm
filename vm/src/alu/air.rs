use std::{array, borrow::Borrow};

use afs_primitives::{utils, xor::bus::XorBus};
use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::ArithmeticLogicCols;
use crate::{
    arch::{instructions::ALU_256_INSTRUCTIONS, ExecutionBridge},
    memory::offline_checker::MemoryBridge,
};

#[derive(Copy, Clone, Debug)]
pub struct ArithmeticLogicAir<const ARG_SIZE: usize, const LIMB_SIZE: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub bus: XorBus,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> PartitionedBaseAir<F>
    for ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>
{
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ArithmeticLogicCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize> Air<AB>
    for ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);

        let ArithmeticLogicCols::<_, NUM_LIMBS, LIMB_BITS> { io, aux } = (*local).borrow();
        builder.assert_bool(aux.is_valid);

        let flags = [
            aux.opcode_add_flag,
            aux.opcode_sub_flag,
            aux.opcode_sltu_flag,
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
            // cubic. Because we constrain that the carry (which is arbitrary) is bool, if
            // carry has degree larger than 1 the max-degree constrain could be at least 4.
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
                .when(aux.opcode_sub_flag + aux.opcode_sltu_flag + aux.opcode_slt_flag)
                .assert_bool(carry_sub[i].clone());
        }

        // For LT, cmp_result must be equal to the last carry. For SLT, cmp_result ^ x_sign ^ y_sign must
        // be equal to the last carry. To ensure maximum cubic degree constraints, we set aux.x_sign and
        // aux.y_sign are 0 when not computing an SLT.
        builder.assert_bool(aux.x_sign);
        builder.assert_bool(aux.y_sign);
        builder
            .when(utils::not(aux.opcode_slt_flag))
            .assert_zero(aux.x_sign);
        builder
            .when(utils::not(aux.opcode_slt_flag))
            .assert_zero(aux.y_sign);

        let slt_xor =
            (aux.opcode_sltu_flag + aux.opcode_slt_flag) * io.cmp_result + aux.x_sign + aux.y_sign
                - AB::Expr::from_canonical_u32(2)
                    * (io.cmp_result * aux.x_sign
                        + io.cmp_result * aux.y_sign
                        + aux.x_sign * aux.y_sign)
                + AB::Expr::from_canonical_u32(4) * (io.cmp_result * aux.x_sign * aux.y_sign);
        builder.assert_eq(
            slt_xor,
            (aux.opcode_sltu_flag + aux.opcode_slt_flag) * carry_sub[NUM_LIMBS - 1].clone(),
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
        builder.when(aux.opcode_eq_flag).assert_one(sum_eq);

        let expected_opcode = flags
            .iter()
            .zip(ALU_256_INSTRUCTIONS)
            .fold(AB::Expr::zero(), |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
            });

        self.eval_interactions(builder, io, aux, expected_opcode);
    }
}
