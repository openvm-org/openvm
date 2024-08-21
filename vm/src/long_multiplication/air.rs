use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use num_traits::cast::ToPrimitive;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::LongMultiplicationCols;
use crate::{cpu::OpCode, long_multiplication::num_limbs};

pub struct LongMultiplicationAir {
    pub arg_size: usize,
    pub limb_size: usize,
    pub bus_index: usize, // for range checking
    pub mul_op: OpCode,   // MUL256 or MUL32
}

impl<F: Field> BaseAir<F> for LongMultiplicationAir {
    fn width(&self) -> usize {
        4 * num_limbs(self.arg_size, self.limb_size) + 2
    }
}

impl<AB: InteractionBuilder> Air<AB> for LongMultiplicationAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local = (*local).borrow();

        let LongMultiplicationCols::<AB::Var> {
            rcv_count: _,
            opcode,
            x_limbs,
            y_limbs,
            z_limbs,
            carry,
        } = LongMultiplicationCols::<AB::Var>::from_slice(local);
        let num_limbs = num_limbs(self.arg_size, self.limb_size);

        assert!(
            num_limbs * ((1 << self.limb_size) - 1) * ((1 << self.limb_size) - 1)
                < AB::F::order().to_usize().unwrap()
        );

        builder.assert_eq(opcode, AB::Expr::from_canonical_u8(self.mul_op as u8));

        for i in 0..num_limbs {
            builder.assert_eq(
                (0..=i)
                    .map(|j| x_limbs[j] * y_limbs[i - j])
                    .fold(AB::Expr::zero(), |acc, x| acc + x)
                    + if i == 0 {
                        AB::Expr::zero()
                    } else {
                        carry[i - 1].into()
                    },
                z_limbs[i] + carry[i] * AB::Expr::from_canonical_u32(1 << self.limb_size),
            );
        }

        self.eval_interactions(
            builder,
            LongMultiplicationCols::<AB::Var>::from_slice(local),
        );
    }
}
