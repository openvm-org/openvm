use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{columns::LongArithmeticCols, num_limbs};
use crate::cpu::OpCode;

/// AIR for the long addition circuit. ARG_SIZE is the size of the arguments in bits, and LIMB_SIZE is the size of the limbs in bits.
#[derive(Copy, Clone, Debug)]
pub struct LongArithmeticAir<const ARG_SIZE: usize, const LIMB_SIZE: usize> {
    pub bus_index: usize, // to communicate with the range checker that checks that all limbs are < 2^LIMB_SIZE
    pub base_op: OpCode,
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize> LongArithmeticAir<ARG_SIZE, LIMB_SIZE> {
    pub fn new(bus_index: usize, base_op: OpCode) -> Self {
        Self { bus_index, base_op }
    }
}

impl<F: Field, const ARG_SIZE: usize, const LIMB_SIZE: usize> BaseAir<F>
    for LongArithmeticAir<ARG_SIZE, LIMB_SIZE>
{
    fn width(&self) -> usize {
        LongArithmeticCols::<ARG_SIZE, LIMB_SIZE, F>::get_width()
    }
}

impl<AB: InteractionBuilder, const ARG_SIZE: usize, const LIMB_SIZE: usize> Air<AB>
    for LongArithmeticAir<ARG_SIZE, LIMB_SIZE>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local = (*local).borrow();

        let cols = LongArithmeticCols::<ARG_SIZE, LIMB_SIZE, AB::Var>::from_slice(local);
        let (io, aux) = (&cols.io, &cols.aux);

        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();

        builder.assert_bool(aux.opcode_add_flag);
        builder.assert_bool(aux.opcode_sub_flag);
        builder.assert_bool(aux.opcode_lt_flag);
        builder.assert_bool(aux.opcode_eq_flag);
        builder.assert_eq(
            aux.opcode_add_flag * AB::Expr::from_canonical_u8(OpCode::ADD256 as u8)
                + aux.opcode_sub_flag * AB::Expr::from_canonical_u8(OpCode::SUB256 as u8)
                + aux.opcode_lt_flag * AB::Expr::from_canonical_u8(OpCode::LT256 as u8)
                + aux.opcode_eq_flag * AB::Expr::from_canonical_u8(OpCode::EQ256 as u8),
            io.opcode,
        );
        builder.assert_one(
            aux.opcode_add_flag + aux.opcode_sub_flag + aux.opcode_lt_flag + aux.opcode_eq_flag,
        );

        for i in 0..num_limbs {
            // If we need to perform an arithmetic operation, we will use "buffer"
            // as a "carry/borrow" vector. We refer to it as "carry" in this section.

            // For addition, we have the following:
            // z[i] + carry[i] * 2^LIMB_SIZE = x[i] + y[i] + carry[i - 1]
            // For subtraction, we have the following:
            // z[i] = x[i] - y[i] - carry[i - 1] + carry[i] * 2^LIMB_SIZE
            // Separating the summands with the same sign from the others, we get:
            // z[i] - x[i] = \pm (y[i] + carry[i - 1] - carry[i] * 2^LIMB_SIZE)

            // Or another way to think about it: we essentially either check that
            // z = x + y, or that x = z + y; and "carry" is always the carry of
            // the addition. So it is natural that x and z are separated from
            // everything else.

            // lhs = +rhs if opcode_add_flag = 1,
            // lhs = -rhs if opcode_sub_flag = 1.
            let lhs = io.y_limbs[i]
                + if i > 0 {
                    aux.buffer[i - 1].into()
                } else {
                    AB::Expr::zero()
                }
                - aux.buffer[i] * AB::Expr::from_canonical_u32(1 << LIMB_SIZE);
            let rhs = io.z_limbs[i] - io.x_limbs[i];
            builder.assert_zero((lhs.clone() - rhs.clone()) * aux.opcode_add_flag);
            builder.assert_zero(
                (lhs.clone() + rhs.clone()) * (aux.opcode_sub_flag + aux.opcode_lt_flag),
            );
            // TODO: degree < 3?
            builder.assert_zero(
                (AB::Expr::one() - aux.opcode_eq_flag)
                    * aux.buffer[i]
                    * (AB::Expr::one() - aux.buffer[i]),
            );
        }

        // If we wanted LT, then cmp_result must equal the last carry.
        builder.assert_zero((io.cmp_result - aux.buffer[num_limbs - 1]) * aux.opcode_lt_flag);
        // If we wanted EQ, we will do as we would do for checking a single number,
        // but we will use "buffer" vector for inverses.
        // Namely, we check that:
        // - cmp_result * (x[i] - y[i]) = 0,
        // - cmp_result + sum_{i < num_limbs} (x[i] - y[i]) * buffer[i] = 1.
        let mut sum_eq: AB::Expr = io.cmp_result.into();
        for i in 0..num_limbs {
            sum_eq += (io.x_limbs[i] - io.y_limbs[i]) * aux.buffer[i];
            // TODO: degree < 3?
            builder
                .assert_zero(aux.opcode_eq_flag * io.cmp_result * (io.x_limbs[i] - io.y_limbs[i]));
        }
        builder.assert_zero((sum_eq - AB::Expr::one()) * aux.opcode_eq_flag);

        self.eval_interactions(builder, cols);
    }
}
