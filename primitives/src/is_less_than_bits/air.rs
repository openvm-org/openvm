use std::borrow::Borrow;

use afs_stark_backend::rap::{BaseAirWithPublicValues, PartitionedBaseAir};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use super::columns::{IsLessThanBitsAuxCols, IsLessThanBitsCols, IsLessThanBitsIoCols};
use crate::sub_chip::{AirConfig, SubAir};

#[derive(Copy, Clone, Debug)]
pub struct IsLessThanBitsAir {
    pub limb_bits: usize,
}

impl IsLessThanBitsAir {
    pub fn new(limb_bits: usize) -> Self {
        Self { limb_bits }
    }
}

impl AirConfig for IsLessThanBitsAir {
    type Cols<T> = IsLessThanBitsCols<T>;
}

impl<F> BaseAirWithPublicValues<F> for IsLessThanBitsAir {}
impl<F> PartitionedBaseAir<F> for IsLessThanBitsAir {}
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
    type IoView = IsLessThanBitsIoCols<AB::Var>;
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
