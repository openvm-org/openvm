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
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize> LongArithmeticAir<ARG_SIZE, LIMB_SIZE> {
    pub const BASE_OP: OpCode = OpCode::ADD;

    pub fn new(bus_index: usize) -> Self {
        Self { bus_index }
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

        builder.assert_bool(aux.opcode_lo);
        builder.assert_bool(aux.opcode_hi);
        builder.assert_eq(
            aux.opcode_hi * AB::Expr::two()
                + aux.opcode_lo
                + AB::Expr::from_canonical_u8(Self::BASE_OP as u8),
            io.opcode,
        );

        builder.assert_zero(aux.opcode_hi);
        let sign = AB::Expr::one() - AB::Expr::two() * aux.opcode_lo;

        for i in 0..num_limbs {
            // For addition, we have the following:
            // z[i] + carry[i] * 2^LIMB_SIZE = x[i] + y[i] + carry[i - 1]
            // For subtraction, we have the following:
            // z[i] = x[i] - y[i] - carry[i - 1] + carry[i] * 2^LIMB_SIZE
            // Separating the summands with the same sign from the others, we get:
            // z[i] - x[i] = \pm (y[i] + carry[i - 1] - carry[i] * 2^LIMB_SIZE)
            let lhs = io.y_limbs[i]
                + if i > 0 {
                    aux.carry[i - 1].into()
                } else {
                    AB::Expr::zero()
                }
                - aux.carry[i] * AB::Expr::from_canonical_u32(1 << LIMB_SIZE);
            let rhs = io.z_limbs[i] - io.x_limbs[i];
            builder.assert_eq(lhs * sign.clone(), rhs);
            builder.assert_bool(aux.carry[i]);
        }

        self.eval_interactions(builder, cols);
    }
}
