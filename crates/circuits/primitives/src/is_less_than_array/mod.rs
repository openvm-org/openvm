use itertools::izip;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::AirBuilder,
    p3_field::{FieldAlgebra, PrimeField32},
};

use crate::{
    is_less_than::{IsLtSubAir, LessThanAuxCols},
    utils::not,
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    SubAir, TraceSubRowGenerator,
};

#[cfg(test)]
pub mod tests;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IsLtArrayIo<T, const NUM: usize> {
    pub x: [T; NUM],
    pub y: [T; NUM],
    /// The boolean output, constrained to equal (x < y) when `condition != 0`. The less than
    /// comparison is done lexicographically.
    pub out: T,
    /// Constraints only hold when `count != 0`. When `count == 0`, setting all trace values
    /// to zero still passes the constraints.
    /// `count` is **assumed** to be boolean and must be constrained as such by the caller.
    pub count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug)]
pub struct IsLtArrayAuxCols<T, const NUM: usize, const AUX_LEN: usize> {
    // `diff_marker` is filled with 0 except at the lowest index i such that
    // `x[i] != y[i]`. If such an `i` exists then it is constrained that `diff_inv = inv(y[i] -
    // x[i])`.
    pub diff_marker: [T; NUM],
    pub diff_inv: T,
    pub lt_aux: LessThanAuxCols<T, AUX_LEN>,
}

#[derive(Clone, Copy, Debug)]
pub struct IsLtArrayAuxColsRef<'a, T> {
    pub diff_marker: &'a [T],
    pub diff_inv: &'a T,
    pub lt_decomp: &'a [T],
}

#[derive(Debug)]
pub struct IsLtArrayAuxColsMut<'a, T> {
    pub diff_marker: &'a mut [T],
    pub diff_inv: &'a mut T,
    pub lt_decomp: &'a mut [T],
}

impl<'a, T, const NUM: usize, const AUX_LEN: usize> From<&'a IsLtArrayAuxCols<T, NUM, AUX_LEN>>
    for IsLtArrayAuxColsRef<'a, T>
{
    fn from(value: &'a IsLtArrayAuxCols<T, NUM, AUX_LEN>) -> Self {
        Self {
            diff_marker: &value.diff_marker,
            diff_inv: &value.diff_inv,
            lt_decomp: &value.lt_aux.lower_decomp,
        }
    }
}

impl<'a, T, const NUM: usize, const AUX_LEN: usize> From<&'a mut IsLtArrayAuxCols<T, NUM, AUX_LEN>>
    for IsLtArrayAuxColsMut<'a, T>
{
    fn from(value: &'a mut IsLtArrayAuxCols<T, NUM, AUX_LEN>) -> Self {
        Self {
            diff_marker: &mut value.diff_marker,
            diff_inv: &mut value.diff_inv,
            lt_decomp: &mut value.lt_aux.lower_decomp,
        }
    }
}

/// This SubAir constrains the boolean equal to 1 iff `x < y` (lexicographic comparison) assuming
/// that all elements of both arrays `x, y` each have at most `max_bits` bits.
///
/// The constraints will constrain a selector for the first index where `x[i] != y[i]` and then
/// use [IsLtSubAir] on `x[i], y[i]`.
///
/// The expected max constraint degree of `eval` is
///     deg(count) + max(1, deg(x), deg(y))
#[derive(Copy, Clone, Debug)]
pub struct IsLtArraySubAir<const NUM: usize> {
    pub lt: IsLtSubAir,
}

impl<const NUM: usize> IsLtArraySubAir<NUM> {
    pub fn new(bus: VariableRangeCheckerBus, max_bits: usize) -> Self {
        Self {
            lt: IsLtSubAir::new(bus, max_bits),
        }
    }

    pub fn when_transition(self) -> IsLtArrayWhenTransitionAir<NUM> {
        IsLtArrayWhenTransitionAir(self)
    }

    pub fn max_bits(&self) -> usize {
        self.lt.max_bits
    }

    pub fn range_max_bits(&self) -> usize {
        self.lt.range_max_bits()
    }

    /// Constrain that `out` is boolean equal to `x < y` (lexicographic comparison)
    /// **without** doing range checks on `lt_decomp`.
    fn eval_without_range_checks<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        io: IsLtArrayIo<AB::Expr, NUM>,
        diff_marker: &[AB::Var],
        diff_inv: AB::Var,
        lt_decomp: &[AB::Var],
    ) {
        assert_eq!(diff_marker.len(), NUM);
        let mut prefix_sum = AB::Expr::ZERO;
        let mut diff_val = AB::Expr::ZERO;
        for (x, y, &marker) in izip!(io.x, io.y, diff_marker) {
            let diff = y - x;
            diff_val += diff.clone() * marker.into();
            prefix_sum += marker.into();
            builder.assert_bool(marker);
            builder
                .when(io.count.clone())
                .assert_zero(not::<AB::Expr>(prefix_sum.clone()) * diff.clone());
            builder.when(marker).assert_one(diff * diff_inv);
        }
        builder.assert_bool(prefix_sum.clone());
        // When condition != 0,
        // - If `x != y`, then `prefix_sum = 1` so marker[i] must be nonzero iff i is the first
        //   index where `x[i] != y[i]`. Constrains that `diff_inv * (y[i] - x[i]) = 1` (`diff_val`
        //   is non-zero).
        // - If `x == y`, then `prefix_sum = 0` and `out == 0` (below)
        //     - `prefix_sum` cannot be 1 because all diff are zero and it would be impossible to
        //       find an inverse.

        builder
            .when(io.count.clone())
            .when(not::<AB::Expr>(prefix_sum))
            .assert_zero(io.out.clone());

        self.lt
            .eval_without_range_checks(builder, diff_val, io.out, io.count, lt_decomp);
    }
}

impl<AB: InteractionBuilder, const NUM: usize> SubAir<AB> for IsLtArraySubAir<NUM> {
    type AirContext<'a>
        = (IsLtArrayIo<AB::Expr, NUM>, IsLtArrayAuxColsRef<'a, AB::Var>)
    where
        AB::Expr: 'a,
        AB::Var: 'a,
        AB: 'a;

    fn eval<'a>(
        &'a self,
        builder: &'a mut AB,
        (io, aux): (IsLtArrayIo<AB::Expr, NUM>, IsLtArrayAuxColsRef<'a, AB::Var>),
    ) where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        self.lt
            .eval_range_checks(builder, aux.lt_decomp, io.count.clone());
        self.eval_without_range_checks(builder, io, aux.diff_marker, *aux.diff_inv, aux.lt_decomp);
    }
}

/// The same subair as [IsLtArraySubAir] except that non-range check
/// constraints are not imposed on the last row.
/// Intended use case is for asserting less than between entries in adjacent rows.
#[derive(Copy, Clone, Debug)]
pub struct IsLtArrayWhenTransitionAir<const NUM: usize>(pub IsLtArraySubAir<NUM>);

impl<AB: InteractionBuilder, const NUM: usize> SubAir<AB> for IsLtArrayWhenTransitionAir<NUM> {
    type AirContext<'a>
        = (IsLtArrayIo<AB::Expr, NUM>, IsLtArrayAuxColsRef<'a, AB::Var>)
    where
        AB::Expr: 'a,
        AB::Var: 'a,
        AB: 'a;

    fn eval<'a>(
        &'a self,
        builder: &'a mut AB,
        (io, aux): (IsLtArrayIo<AB::Expr, NUM>, IsLtArrayAuxColsRef<'a, AB::Var>),
    ) where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        self.0
            .lt
            .eval_range_checks(builder, aux.lt_decomp, io.count.clone());
        self.0.eval_without_range_checks(
            &mut builder.when_transition(),
            io,
            aux.diff_marker,
            *aux.diff_inv,
            aux.lt_decomp,
        );
    }
}

impl<F: PrimeField32, const NUM: usize> TraceSubRowGenerator<F> for IsLtArraySubAir<NUM> {
    /// `(range_checker, x, y)`
    type TraceContext<'a> = (&'a VariableRangeCheckerChip, &'a [F], &'a [F]);
    /// `(aux, out)`
    type ColsMut<'a> = (IsLtArrayAuxColsMut<'a, F>, &'a mut F);

    /// Only use this when `count != 0`.
    #[inline(always)]
    fn generate_subrow<'a>(
        &'a self,
        (range_checker, x, y): (&'a VariableRangeCheckerChip, &'a [F], &'a [F]),
        (aux, out): (IsLtArrayAuxColsMut<'a, F>, &'a mut F),
    ) {
        tracing::trace!("IsLtArraySubAir::generate_subrow x={:?}, y={:?}", x, y);
        let mut is_eq = true;
        let mut diff_val = F::ZERO;
        *aux.diff_inv = F::ZERO;
        for (x_i, y_i, diff_marker) in izip!(x, y, aux.diff_marker.iter_mut()) {
            if x_i != y_i && is_eq {
                is_eq = false;
                *diff_marker = F::ONE;
                diff_val = *y_i - *x_i;
                *aux.diff_inv = diff_val.inverse();
            } else {
                *diff_marker = F::ZERO;
            }
        }
        // diff_val can be "negative" but shifted_diff is in [0, 2^{max_bits+1})
        let shifted_diff =
            (diff_val + F::from_canonical_u32((1 << self.max_bits()) - 1)).as_canonical_u32();
        let lower_u32 = shifted_diff & ((1 << self.max_bits()) - 1);
        *out = F::from_bool(shifted_diff != lower_u32);

        // decompose lower_u32 into limbs and range check
        range_checker.decompose(lower_u32, self.max_bits(), aux.lt_decomp);
    }
}
