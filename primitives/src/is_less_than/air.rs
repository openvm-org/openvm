use std::borrow::Borrow;

use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::{IsLessThanAuxCols, IsLessThanCols, IsLessThanIoCols};
use crate::{
    sub_chip::{AirConfig, SubAir},
    var_range::bus::VariableRangeCheckerBus,
};

#[derive(Copy, Clone, Debug)]
pub struct IsLessThanAir {
    /// The bus for sends to range chip
    pub bus: VariableRangeCheckerBus,
    /// The maximum number of bits for the numbers to compare
    pub max_bits: usize,
    /// num_limbs is the number of limbs we decompose each input into
    pub num_limbs: usize,
}

impl IsLessThanAir {
    pub fn new(bus: VariableRangeCheckerBus, max_bits: usize) -> Self {
        Self {
            bus,
            max_bits,
            num_limbs: (max_bits + bus.range_max_bits - 1) / bus.range_max_bits,
        }
    }

    pub fn range_max_bits(&self) -> usize {
        self.bus.range_max_bits
    }

    /// FOR INTERNAL USE ONLY.
    /// This AIR is only sound if interactions are enabled
    ///
    /// Constraints between `io` and `aux` are only enforced when `condition != 0`.
    /// This means `aux` can be all zero independent of what `io` is by setting `condition = 0`.
    pub(crate) fn conditional_eval_without_interactions<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        io: IsLessThanIoCols<AB::Expr>,
        aux: IsLessThanAuxCols<AB::Var>,
        condition: impl Into<AB::Expr>,
    ) {
        let x = io.x;
        let y = io.y;
        let less_than = io.less_than;

        let local_aux = &aux;

        let lower_decomp = local_aux.lower_decomp.clone();

        // this is the desired intermediate value (i.e. 2^limb_bits + y - x - 1)
        let intermed_val =
            y - x + AB::Expr::from_canonical_u64(1 << self.max_bits) - AB::Expr::one();

        // Construct lower from lower_decomp:
        // - each limb of lower_decomp will be range checked
        let lower = lower_decomp
            .iter()
            .enumerate()
            .fold(AB::Expr::zero(), |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u64(1 << (i * self.range_max_bits()))
            });

        // constrain that the lower + less_than * 2^limb_bits is the correct intermediate sum
        // note that the intermediate value will be >= 2^limb_bits if and only if x < y, and check_val will therefore be
        // the correct value if and only if less_than is the indicator for whether x < y
        let check_val =
            lower + less_than.clone() * AB::Expr::from_canonical_u64(1 << self.max_bits);

        builder.when(condition).assert_eq(intermed_val, check_val);

        // constrain that less_than is a boolean
        builder.assert_bool(less_than);
    }
}

impl AirConfig for IsLessThanAir {
    type Cols<T> = IsLessThanCols<T>;
}

impl<F: Field> BaseAirWithPublicValues<F> for IsLessThanAir {}
impl<F: Field> PartitionedBaseAir<F> for IsLessThanAir {}
impl<F: Field> BaseAir<F> for IsLessThanAir {
    fn width(&self) -> usize {
        IsLessThanCols::<F>::width(self)
    }
}

impl<AB: InteractionBuilder> Air<AB> for IsLessThanAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let local_cols = IsLessThanCols::<AB::Var>::from_slice(local);

        SubAir::eval(self, builder, local_cols.io, local_cols.aux);
    }
}

// sub-air with constraints to check whether one number is less than another
impl<AB: InteractionBuilder> SubAir<AB> for IsLessThanAir {
    type IoView = IsLessThanIoCols<AB::Var>;
    type AuxView = IsLessThanAuxCols<AB::Var>;

    // constrain that the result of x < y is given by less_than
    // warning: send for range check must be included for the constraints to be sound
    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        // Note: every AIR that uses this sub-AIR must include the interactions for soundness
        self.conditional_eval(
            builder,
            IsLessThanIoCols::<AB::Expr>::new(io.x, io.y, io.less_than),
            aux,
            AB::F::one(),
        );
    }
}

impl IsLessThanAir {
    /// `count` is the frequency of each range check.
    /// The primary use case is when `count` is boolean, so if `count == 0` then no
    /// range checks are done and the aux columns can all be zero without affecting
    pub fn conditional_eval<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: IsLessThanIoCols<AB::Expr>,
        aux: IsLessThanAuxCols<AB::Var>,
        count: impl Into<AB::Expr>,
    ) {
        let io_exprs = IsLessThanIoCols::<AB::Expr>::new(io.x, io.y, io.less_than);
        let count = count.into();

        self.eval_interactions(builder, aux.lower_decomp.clone(), count.clone());
        self.conditional_eval_without_interactions(builder, io_exprs, aux, count);
    }

    /// Imposes the non-interaction constraints on all except the last row. This is
    /// intended for use when the comparators `x, y` are on adjacent rows.
    ///
    /// This function does also enable the interaction constraints _on every row_.
    /// The `eval_interactions` performs range checks on `lower_decomp` on every row, even
    /// though in this AIR the lower_decomp is not used on the last row.
    /// This simply means the trace generation must fill in the last row with numbers in
    /// range (e.g., with zeros)
    pub fn eval_when_transition<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: IsLessThanIoCols<impl Into<AB::Expr>>,
        aux: IsLessThanAuxCols<AB::Var>,
    ) {
        let io_exprs = IsLessThanIoCols::<AB::Expr>::new(io.x, io.y, io.less_than);
        let count = AB::F::one();

        self.eval_interactions(builder, aux.lower_decomp.clone(), count);
        self.conditional_eval_without_interactions(
            &mut builder.when_transition(),
            io_exprs,
            aux,
            count,
        );
    }
}
