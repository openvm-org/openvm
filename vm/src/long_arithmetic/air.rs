use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::LongAdditionCols;

/// AIR for the long addition circuit. ARG_SIZE is the size of the arguments in bits, and LIMB_SIZE is the size of the limbs in bits.
#[derive(Copy, Clone, Debug)]
pub struct LongAdditionAir<const ARG_SIZE: usize, const LIMB_SIZE: usize> {
    pub bus_index: usize, // to communicate with the range checker that checks that all limbs are < 2^LIMB_SIZE
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize> LongAdditionAir<ARG_SIZE, LIMB_SIZE> {
    pub fn new(bus_index: usize) -> Self {
        Self { bus_index }
    }
}

impl<F: Field, const ARG_SIZE: usize, const LIMB_SIZE: usize> BaseAir<F>
    for LongAdditionAir<ARG_SIZE, LIMB_SIZE>
{
    fn width(&self) -> usize {
        LongAdditionCols::<ARG_SIZE, LIMB_SIZE, F>::get_width()
    }
}

impl<AB: InteractionBuilder, const ARG_SIZE: usize, const LIMB_SIZE: usize> Air<AB>
    for LongAdditionAir<ARG_SIZE, LIMB_SIZE>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let long_cols = LongAdditionCols::<ARG_SIZE, LIMB_SIZE, AB::Var>::from_slice(local);
        let num_limbs = LongAdditionCols::<ARG_SIZE, LIMB_SIZE, AB::Var>::num_limbs();

        for i in 0..num_limbs {
            let limb_sum = long_cols.x_limbs[i]
                + long_cols.y_limbs[i]
                + if i > 0 {
                    long_cols.carry[i - 1].into()
                } else {
                    AB::Expr::zero()
                };

            builder.assert_eq(
                limb_sum - long_cols.z_limbs[i],
                long_cols.carry[i] * AB::Expr::from_canonical_u32(1 << LIMB_SIZE),
            );
            builder.assert_bool(long_cols.carry[i]);
        }

        self.eval_interactions(builder, long_cols);
    }
}
