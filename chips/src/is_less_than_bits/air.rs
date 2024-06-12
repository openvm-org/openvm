use std::borrow::Borrow;

use afs_stark_backend::interaction::AirBridge;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::sub_chip::{AirConfig, SubAir};

use super::columns::{IsLessThanBitsAuxCols, IsLessThanBitsCols, IsLessThanBitsIOCols};
use super::IsLessThanBitsAir;

impl AirConfig for IsLessThanBitsAir {
    type Cols<T> = IsLessThanBitsCols<T>;
}

// No interactions
impl<F: Field> AirBridge<F> for IsLessThanBitsAir {}

impl<F> BaseAir<F> for IsLessThanBitsAir {
    fn width(&self) -> usize {
        3 + (self.limb_bits * 3)
    }
}

impl<AB: AirBuilder> Air<AB> for IsLessThanBitsAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let local_cols = IsLessThanBitsCols::<AB::Var>::from_slice(self.limb_bits, local);

        SubAir::eval(self, builder, local_cols.io, local_cols.aux);
    }
}

impl<AB: AirBuilder> SubAir<AB> for IsLessThanBitsAir {
    type IoView = IsLessThanBitsIOCols<AB::Var>;
    type AuxView = IsLessThanBitsAuxCols<AB::Var>;

    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        let x = io.x;
        let y = io.y;
        let is_less_than = io.is_less_than;
        let x_bits = aux.x_bits;
        let y_bits = aux.y_bits;
        let comparisons = aux.comparisons;

        for d in 0..self.limb_bits {
            builder.assert_bool(x_bits[d] * (x_bits[d] - AB::Expr::one()));
            builder.assert_bool(y_bits[d] * (y_bits[d] - AB::Expr::one()));
        }
        let mut sum_bits_x = AB::Expr::zero();
        let mut sum_bits_y = AB::Expr::zero();
        for d in 0..self.limb_bits {
            sum_bits_x += AB::Expr::from_canonical_u64(1 << d) * x_bits[d];
            sum_bits_y += AB::Expr::from_canonical_u64(1 << d) * y_bits[d];
        }
        builder.assert_eq(sum_bits_x, x);
        builder.assert_eq(sum_bits_y, y);

        builder.assert_eq(comparisons[0], (AB::Expr::one() - x_bits[0]) * y_bits[0]);
        for d in 1..self.limb_bits {
            let comparison_check = ((AB::Expr::one()
                - ((x_bits[d] - y_bits[d]) * (x_bits[d] - y_bits[d])))
                * comparisons[d - 1])
                + (AB::Expr::one() - x_bits[d]) * y_bits[d];
            builder.assert_eq(comparisons[d], comparison_check);
        }

        builder.assert_eq(is_less_than, comparisons[self.limb_bits - 1]);
    }
}
