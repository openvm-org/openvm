use std::{array, borrow::Borrow};

use afs_primitives::xor::bus::XorBus;
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
        let mut carry: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::zero());
        let divide = AB::F::from_canonical_usize(1 << LIMB_BITS).inverse();

        for i in 0..NUM_LIMBS {
            let y_and_carry = y_limbs[i]
                + if i > 0 {
                    carry[i - 1].clone()
                } else {
                    AB::Expr::zero()
                };
            let x_and_z = x_limbs[i] - z_limbs[i];
            carry[i] = AB::Expr::from(divide)
                * (y_and_carry
                    + x_and_z * (aux.opcode_add_flag - aux.opcode_sub_flag - aux.opcode_lt_flag));
            builder
                .when(aux.opcode_add_flag + aux.opcode_sub_flag + aux.opcode_lt_flag)
                .assert_bool(carry[i].clone());
        }

        // For LT, cmp_result must be equal to the last carry.
        builder
            .when(aux.opcode_lt_flag)
            .assert_zero(io.cmp_result - carry[NUM_LIMBS - 1].clone());

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
