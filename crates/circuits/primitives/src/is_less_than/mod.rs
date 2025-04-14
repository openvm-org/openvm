use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::AirBuilder,
    p3_field::{Field, FieldAlgebra},
};

use crate::{
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    SubAir, TraceSubRowGenerator,
};

#[cfg(test)]
pub mod tests;

// Aux cols are the same for assert_less_than and is_less_than
pub use super::assert_less_than::LessThanAuxCols;

/// The IO is typically provided with `T = AB::Expr` as external context.
// This does not derive AlignedBorrow because it is usually **not** going to be
// direct columns in an AIR.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IsLessThanIo<T> {
    pub x: T,
    pub y: T,
    /// The boolean output, constrained to equal (x < y) when `count != 0`.
    pub out: T,
    /// Range checks are done with multiplicity `count`.
    /// If `count == 0` then no range checks are done.
    /// `count` is **assumed** to be boolean and must be constrained as such by
    /// the caller.
    ///
    /// N.B.: in fact range checks could always be done, if the aux
    /// subrow values are set to 0 when `count == 0`. This would slightly
    /// simplify the range check interactions, although usually doesn't change
    /// the overall constraint degree. It however leads to the annoyance that
    /// you must update the RangeChecker's multiplicities even on dummy padding
    /// rows. To improve quality of life,
    /// we currently use this more complex constraint.
    pub count: T,
}
impl<T> IsLessThanIo<T> {
    pub fn new(x: impl Into<T>, y: impl Into<T>, out: impl Into<T>, count: impl Into<T>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            out: out.into(),
            count: count.into(),
        }
    }
}

/// This is intended for use as a **SubAir**, not as a standalone Air.
///
/// This SubAir constrains the boolean equal to 1 iff `x < y`, assuming
/// the two numbers both have a max number of bits, given by `max_bits`.
/// The SubAir compares the numbers by decomposing `y - x - 1 + 2^max_bits`
/// into limbs of size `bus.range_max_bits`, and interacts with a
/// `VariableRangeCheckerBus` to range check the decompositions.
///
/// The SubAir will own auxiliary columns to store the decomposed limbs.
/// The number of limbs is `max_bits.div_ceil(bus.range_max_bits)`.
///
/// The expected max constraint degree of `eval` is
///     deg(count) + max(1, deg(x), deg(y))
///
/// N.B.: AssertLtSubAir could be implemented by directly passing through
/// to IsLtSubAir with `out = AB::Expr::ONE`. The only additional
/// constraint in this air is `assert_bool(io.out)`. However since both Airs
/// are fundamental and the constraints are simple, we opt to keep the two
/// versions separate.
#[derive(Copy, Clone, Debug)]
pub struct IsLtSubAir {
    /// The bus for sends to range chip
    pub bus: VariableRangeCheckerBus,
    /// The maximum number of bits for the numbers to compare
    /// Soundness requirement: max_bits <= 29
    ///     max_bits > 29 doesn't work: the approach is to decompose and range check `y - x - 1 +
    /// 2^max_bits` is non-negative.     This requires 2^{max_bits+1} < |F|.
    ///     When F::bits() = 31, this implies max_bits <= 29.
    pub max_bits: usize,
    /// `decomp_limbs = max_bits.div_ceil(bus.range_max_bits)`.
    pub decomp_limbs: usize,
}

impl IsLtSubAir {
    pub fn new(bus: VariableRangeCheckerBus, max_bits: usize) -> Self {
        assert!(max_bits <= 29); // see soundness requirement above
        let decomp_limbs = max_bits.div_ceil(bus.range_max_bits);
        Self {
            bus,
            max_bits,
            decomp_limbs,
        }
    }

    pub fn range_max_bits(&self) -> usize {
        self.bus.range_max_bits
    }

    pub fn when_transition(self) -> IsLtWhenTransitionAir {
        IsLtWhenTransitionAir(self)
    }

    /// FOR INTERNAL USE ONLY.
    /// This AIR is only sound if interactions are enabled
    ///
    /// Constraints between `io` and `aux` are only enforced when `condition != 0`.
    /// This means `aux` can be all zero independent on what `io` is by setting `condition = 0`.
    #[inline(always)]
    pub(crate) fn eval_without_range_checks<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        y_minus_x: impl Into<AB::Expr>,
        out: impl Into<AB::Expr>,
        condition: impl Into<AB::Expr>,
        lower_decomp: &[AB::Var],
    ) {
        assert_eq!(lower_decomp.len(), self.decomp_limbs);
        // this is the desired intermediate value (i.e. y - x - 1)
        // deg(intermed_val) = deg(io)
        let intermed_val =
            y_minus_x.into() + AB::Expr::from_canonical_usize((1 << self.max_bits) - 1);

        // Construct lower from lower_decomp:
        // - each limb of lower_decomp will be range checked
        // deg(lower) = 1
        let lower = lower_decomp
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_usize(1 << (i * self.range_max_bits()))
            });

        let out = out.into();
        // constrain that the lower + out * 2^max_bits is the correct intermediate sum
        let check_val = lower + out.clone() * AB::Expr::from_canonical_usize(1 << self.max_bits);
        // the degree of this constraint is expected to be deg(count) + max(deg(intermed_val),
        // deg(lower))
        builder.when(condition).assert_eq(intermed_val, check_val);
        builder.assert_bool(out);
    }

    #[inline(always)]
    pub(crate) fn eval_range_checks<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        lower_decomp: &[AB::Var],
        count: impl Into<AB::Expr>,
    ) {
        let count = count.into();
        let mut bits_remaining = self.max_bits;
        // we range check the limbs of the lower_decomp so that we know each element
        // of lower_decomp has the correct number of bits
        for limb in lower_decomp {
            // the last limb might have fewer than `bus.range_max_bits` bits
            let range_bits = bits_remaining.min(self.bus.range_max_bits);
            self.bus
                .range_check(*limb, range_bits)
                .eval(builder, count.clone());
            bits_remaining = bits_remaining.saturating_sub(self.bus.range_max_bits);
        }
    }
}

impl<AB: InteractionBuilder> SubAir<AB> for IsLtSubAir {
    type AirContext<'a>
        = (IsLessThanIo<AB::Expr>, &'a [AB::Var])
    where
        AB::Expr: 'a,
        AB::Var: 'a,
        AB: 'a;

    // constrain that out == (x < y) when count != 0
    // warning: send for range check must be included for the constraints to be sound
    fn eval<'a>(
        &'a self,
        builder: &'a mut AB,
        (io, lower_decomp): (IsLessThanIo<AB::Expr>, &'a [AB::Var]),
    ) where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        // Note: every AIR that uses this sub-AIR must include the range checks for soundness
        self.eval_range_checks(builder, lower_decomp, io.count.clone());
        self.eval_without_range_checks(builder, io.y - io.x, io.out, io.count, lower_decomp);
    }
}

/// The same subair as [IsLtSubAir] except that non-range check
/// constraints are not imposed on the last row.
/// Intended use case is for asserting less than between entries in
/// adjacent rows.
#[derive(Clone, Copy, Debug)]
pub struct IsLtWhenTransitionAir(pub IsLtSubAir);

impl<AB: InteractionBuilder> SubAir<AB> for IsLtWhenTransitionAir {
    type AirContext<'a>
        = (IsLessThanIo<AB::Expr>, &'a [AB::Var])
    where
        AB::Expr: 'a,
        AB::Var: 'a,
        AB: 'a;

    /// Imposes the non-interaction constraints on all except the last row. This is
    /// intended for use when the comparators `x, y` are on adjacent rows.
    ///
    /// This function does also enable the interaction constraints _on every row_.
    /// The `eval_interactions` performs range checks on `lower_decomp` on every row, even
    /// though in this AIR the lower_decomp is not used on the last row.
    /// This simply means the trace generation must fill in the last row with numbers in
    /// range (e.g., with zeros)
    fn eval<'a>(
        &'a self,
        builder: &'a mut AB,
        (io, lower_decomp): (IsLessThanIo<AB::Expr>, &'a [AB::Var]),
    ) where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        self.0
            .eval_range_checks(builder, lower_decomp, io.count.clone());
        self.0.eval_without_range_checks(
            &mut builder.when_transition(),
            io.y - io.x,
            io.out,
            io.count,
            lower_decomp,
        );
    }
}

impl<F: Field> TraceSubRowGenerator<F> for IsLtSubAir {
    /// `(range_checker, x, y)`
    type TraceContext<'a> = (&'a VariableRangeCheckerChip, u32, u32);
    /// `(lower_decomp, out)`
    type ColsMut<'a> = (&'a mut [F], &'a mut F);

    /// Only use this when `count != 0`.
    #[inline(always)]
    fn generate_subrow<'a>(
        &'a self,
        (range_checker, x, y): (&'a VariableRangeCheckerChip, u32, u32),
        (lower_decomp, out): (&'a mut [F], &'a mut F),
    ) {
        debug_assert_eq!(lower_decomp.len(), self.decomp_limbs);
        debug_assert!(
            x < (1 << self.max_bits),
            "{x} has more than {} bits",
            self.max_bits
        );
        debug_assert!(
            y < (1 << self.max_bits),
            "{y} has more than {} bits",
            self.max_bits
        );
        *out = F::from_bool(x < y);

        // obtain the lower_bits
        let check_less_than = (1 << self.max_bits) + y - x - 1;
        let lower_u32 = check_less_than & ((1 << self.max_bits) - 1);

        // decompose lower_u32 into limbs and range check
        range_checker.decompose(lower_u32, self.max_bits, lower_decomp);
    }
}
