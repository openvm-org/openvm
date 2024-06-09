use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::sub_chip::{AirConfig, SubAir};

use super::{
    columns::{IsLessThanAuxCols, IsLessThanCols, IsLessThanIOCols},
    IsLessThanAir,
};

impl AirConfig for IsLessThanAir {
    type Cols<T> = IsLessThanCols<T>;
}

impl<F: Field> BaseAir<F> for IsLessThanAir {
    fn width(&self) -> usize {
        IsLessThanCols::<F>::get_width(self.limb_bits(), self.decomp())
    }
}

impl<AB: AirBuilder> Air<AB> for IsLessThanAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let local_cols = IsLessThanCols::<AB::Var>::from_slice(local);

        SubAir::eval(self, builder, local_cols.io, local_cols.aux);
    }
}

// sub-chip with constraints to check whether one number is less than another
impl<AB: AirBuilder> SubAir<AB> for IsLessThanAir {
    type IoView = IsLessThanIOCols<AB::Var>;
    type AuxView = IsLessThanAuxCols<AB::Var>;

    // constrain that the result of x < y is given by less_than
    // warning: send for range check must be included for the constraints to be sound
    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        let x = io.x;
        let y = io.y;
        let less_than = io.less_than;

        let local_aux = &aux;

        let lower = local_aux.lower;
        let lower_decomp = local_aux.lower_decomp.clone();

        // to range check the last limb of the decomposed lower_bits, we need to shift it to make sure it is in
        // the correct range
        let last_limb_shift = (self.decomp() - (self.limb_bits() % self.decomp())) % self.decomp();

        // this is the desired intermediate value (i.e. 2^limb_bits + y - x - 1)
        let intermed_val =
            y - x + AB::Expr::from_canonical_u64(1 << self.limb_bits()) - AB::Expr::one();

        // constrain that the lower_bits + less_than * 2^limb_bits is the correct intermediate sum
        // note that the intermediate value will be >= 2^limb_bits if and only if x < y, and check_val will therefore be
        // the correct value if and only if less_than is the indicator for whether x < y
        let check_val = lower + less_than * AB::Expr::from_canonical_u64(1 << self.limb_bits());

        builder.assert_eq(intermed_val, check_val);

        // constrain that the decomposition of lower_bits is correct
        // each limb will be range checked
        let lower_from_decomp = lower_decomp
            .iter()
            .enumerate()
            .take(self.num_limbs())
            .fold(AB::Expr::zero(), |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u64(1 << (i * self.decomp()))
            });

        builder.assert_eq(lower_from_decomp, lower);

        let shifted_val =
            lower_decomp[self.num_limbs() - 1] * AB::Expr::from_canonical_u64(1 << last_limb_shift);

        // constrain that the shifted last limb is shifted correctly
        // this shifted last limb will also be range checked
        builder.assert_eq(lower_decomp[self.num_limbs()], shifted_val);

        // constrain that less_than is a boolean
        builder.assert_bool(less_than);
    }
}
