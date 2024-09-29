use std::borrow::Borrow;

use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::{AssertLessThanAuxCols, AssertLessThanCols, AssertLessThanIoCols};
use crate::{
    sub_chip::{AirConfig, SubAir},
    var_range::bus::VariableRangeCheckerBus,
};

/// AUX_LEN is expected to be (max_bits + bus.range_max_bits - 1) / bus.range_max_bits
///
/// The number of bits to decompose each number into is inferred from bus.range_max_bits
///
/// The expected max constraint degree of conditional_eval is
///     deg(condition) + max(1, deg(io.x), deg(io.y))
#[derive(Copy, Clone, Debug)]
pub struct AssertLessThanAir<const AUX_LEN: usize> {
    /// The bus for sends to range chip
    pub bus: VariableRangeCheckerBus,
    /// The maximum number of bits for the numbers to compare
    /// Soundness requirement: max_bits <= 29
    ///     max_bits > 29 doesn't work: the approach is to check that y-x-1 is non-negative.
    ///     For a field with prime modular, this is equivalent to checking that y-x-1 is in
    ///     the range [0, 2^max_bits - 1]. However, for max_bits > 29, if y is small enough
    ///     and x is large enough, then y-x-1 is negative but can still be in the range due
    ///     to the field size not being big enough.
    pub max_bits: usize,
}

impl<const AUX_LEN: usize> AssertLessThanAir<AUX_LEN> {
    pub fn new(bus: VariableRangeCheckerBus, max_bits: usize) -> Self {
        debug_assert!(AUX_LEN == (max_bits + bus.range_max_bits - 1) / bus.range_max_bits);
        Self { bus, max_bits }
    }

    /// FOR INTERNAL USE ONLY.
    /// This AIR is only sound if interactions are enabled
    ///
    /// Constraints between `io` and `aux` are only enforced when `condition != 0`.
    /// This means `aux` can be all zero independent of what `io` is by setting `condition = 0`.
    pub(super) fn conditional_eval_without_interactions<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        io: AssertLessThanIoCols<AB::Expr>,
        aux: AssertLessThanAuxCols<AB::Var, AUX_LEN>,
        condition: impl Into<AB::Expr>,
    ) {
        let x = io.x;
        let y = io.y;

        let lower_decomp = aux.lower_decomp;

        // this is the desired intermediate value (i.e. y - x - 1)
        // deg(intermed_val) = deg(io)
        let intermed_val = y - x - AB::Expr::one();

        // Construct lower from lower_decomp:
        // - each limb of lower_decomp will be range checked
        // deg(lower) = 1
        let lower = lower_decomp
            .iter()
            .enumerate()
            .fold(AB::Expr::zero(), |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u64(1 << (i * self.bus.range_max_bits))
            });

        // constrain that y-x-1 is equal to the constructed lower value.
        // this enforces that the intermediate value is in the range [0, 2^max_bits - 1], which is equivalent to x < y
        builder.when(condition).assert_eq(intermed_val, lower);
        // the degree of this constraint is expected to be deg(condition) + max(deg(intermed_val), deg(lower))
        // since we are constraining condition * intermed_val == condition * lower,
    }
}

impl<const AUX_LEN: usize> AirConfig for AssertLessThanAir<AUX_LEN> {
    type Cols<T> = AssertLessThanCols<T, AUX_LEN>;
}

impl<F: Field, const AUX_LEN: usize> BaseAirWithPublicValues<F> for AssertLessThanAir<AUX_LEN> {}
impl<F: Field, const AUX_LEN: usize> BaseAir<F> for AssertLessThanAir<AUX_LEN> {
    fn width(&self) -> usize {
        AssertLessThanCols::<F, AUX_LEN>::width()
    }
}

impl<AB: InteractionBuilder, const AUX_LEN: usize> Air<AB> for AssertLessThanAir<AUX_LEN> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local_cols: &AssertLessThanCols<_, AUX_LEN> = (*local).borrow();

        SubAir::eval(self, builder, local_cols.io, local_cols.aux);
    }
}

// sub-air with constraints to check whether one number is less than another
impl<AB: InteractionBuilder, const AUX_LEN: usize> SubAir<AB> for AssertLessThanAir<AUX_LEN> {
    type IoView = AssertLessThanIoCols<AB::Var>;
    type AuxView = AssertLessThanAuxCols<AB::Var, AUX_LEN>;

    // constrain that x < y
    // warning: send for range check must be included for the constraints to be sound
    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        // Note: every AIR that uses this sub-AIR must include the interactions for soundness
        self.conditional_eval(
            builder,
            AssertLessThanIoCols::<AB::Expr>::new(io.x, io.y),
            aux,
            AB::F::one(),
        );
    }
}

impl<const AUX_LEN: usize> AssertLessThanAir<AUX_LEN> {
    /// `count` is the frequency of each range check.
    /// The primary use case is when `count` is boolean, so if `count == 0` then no
    /// range checks are done and the aux columns can all be zero without affecting
    pub fn conditional_eval<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: AssertLessThanIoCols<AB::Expr>,
        aux: AssertLessThanAuxCols<AB::Var, AUX_LEN>,
        count: impl Into<AB::Expr>,
    ) {
        let io_exprs = AssertLessThanIoCols::<AB::Expr>::new(io.x, io.y);
        let count = count.into();

        self.eval_interactions(builder, aux.lower_decomp, count.clone());
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
        io: AssertLessThanIoCols<impl Into<AB::Expr>>,
        aux: AssertLessThanAuxCols<AB::Var, AUX_LEN>,
    ) {
        let io_exprs = AssertLessThanIoCols::<AB::Expr>::new(io.x, io.y);
        let count = AB::F::one();

        self.eval_interactions(builder, aux.lower_decomp, count);
        self.conditional_eval_without_interactions(
            &mut builder.when_transition(),
            io_exprs,
            aux,
            count,
        );
    }
}
