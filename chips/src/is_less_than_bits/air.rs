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
        3 + (self.limb_bits + 1)
    }
}

impl<AB: AirBuilder> Air<AB> for IsLessThanBitsAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let local_cols = IsLessThanBitsCols::<AB::Var>::from_slice(local);

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
        let source_bits = aux.source_bits;

        for source_bit in &source_bits {
            builder.assert_bool(*source_bit);
        }
        let mut sum_source_bits = AB::Expr::zero();
        for (d, &source_bit) in source_bits.iter().enumerate().take(self.limb_bits + 1) {
            sum_source_bits += AB::Expr::from_canonical_u64(1 << d) * source_bit;
        }
        builder.assert_eq(
            sum_source_bits,
            x - y + AB::Expr::from_canonical_u64(1 << self.limb_bits),
        );

        let most_significant = source_bits[self.limb_bits];
        builder.assert_eq(is_less_than, AB::Expr::one() - most_significant);
    }
}
