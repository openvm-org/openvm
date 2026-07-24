use derive_new::new;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::AirBuilder,
    p3_field::{Field, PrimeCharacteristicRing},
};

use crate::{
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    StructReflection, StructReflectionHelper, SubAir, TraceSubRowGenerator,
};

#[cfg(test)]
pub mod tests;

/// The IO is typically provided with `T = AB::Expr` as external context.
// This does not derive AlignedBorrow because it is usually **not** going to be
// direct columns in an AIR.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AssertLessThanIo<T> {
    pub x: T,
    pub y: T,
    /// Will only apply constraints when `count != 0`.
    /// Range checks are done with multiplicity `count`.
    /// If `count == 0` then no range checks are done.
    /// `count` **assumed** to be boolean and must be constrained as such by
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
impl<T> AssertLessThanIo<T> {
    pub fn new(x: impl Into<T>, y: impl Into<T>, count: impl Into<T>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            count: count.into(),
        }
    }
}

/// Auxiliary columns owned by the SubAir. Typically used with `T = AB::Var`.
///
/// `AUX_LEN` must equal `max_bits.div_ceil(bus.range_max_bits)`.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug, new)]
pub struct LessThanAuxCols<T, const AUX_LEN: usize> {
    /// The decomposition limbs range checked by the SubAir. The final limb may use fewer bits than
    /// `bus.range_max_bits`.
    pub diff_decomp: [T; AUX_LEN],
}

/// This is intended for use as a **SubAir**, not as a standalone Air.
///
/// This SubAir constrains that `x < y` when `count != 0`, assuming
/// the two numbers both have a max number of bits, given by `max_bits`.
/// The SubAir decomposes `y - x - 1` into limbs of
/// size `bus.range_max_bits`, and interacts with a
/// `VariableRangeCheckerBus` to range check the decompositions.
///
/// The SubAir will own auxiliary columns to store the decomposed limbs.
/// The number of limbs is `max_bits.div_ceil(bus.range_max_bits)`.
///
/// The expected max constraint degree of `eval` is
///     deg(count) + max(1, deg(x), deg(y))
#[derive(Copy, Clone, Debug)]
pub struct AssertLtSubAir {
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
    /// `decomp_limbs = max_bits.div_ceil(bus.range_max_bits)` is the
    /// number of limbs that `y - x - 1` will be decomposed into.
    pub decomp_limbs: usize,
}

impl AssertLtSubAir {
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

    /// Reconstructs the value that `diff_decomp` decomposes.
    ///
    /// The returned expression has degree 1.
    #[inline(always)]
    fn compose<AB: AirBuilder<Var: Copy>>(&self, diff_decomp: &[AB::Var]) -> AB::Expr {
        assert_eq!(diff_decomp.len(), self.decomp_limbs);
        diff_decomp
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                acc + val * AB::Expr::from_usize(1 << (i * self.range_max_bits()))
            })
    }

    /// Range checks `diff_decomp` and returns `x = y - 1 - compose(diff_decomp)`.
    ///
    /// Since `diff_decomp` is a complete decomposition of `y - x - 1`, the returned expression uses
    /// `y` and the committed `diff_decomp` columns. It has degree `max(1, deg(y))`.
    ///
    /// When `count != 0`, the range checks bound `compose(diff_decomp)` to `[0, 2^max_bits)`, so
    /// `x < y` holds. Callers use the returned value in their external constraints or interactions
    /// on enabled rows.
    #[must_use]
    #[inline(always)]
    pub fn eval_derive_x<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        y: AB::Expr,
        diff_decomp: &[AB::Var],
        count: impl Into<AB::Expr>,
    ) -> AB::Expr {
        self.eval_range_checks(builder, diff_decomp, count);
        y - AB::Expr::ONE - self.compose::<AB>(diff_decomp)
    }

    /// Constrains `diff_decomp` to compose to `y - x - 1` when `io.count != 0`.
    #[inline(always)]
    fn eval_composition<AB: AirBuilder<Var: Copy>>(
        &self,
        builder: &mut AB,
        io: AssertLessThanIo<AB::Expr>,
        diff_decomp: &[AB::Var],
    ) {
        // this is the desired intermediate value (i.e. y - x - 1)
        // deg(intermed_val) = deg(io)
        let intermed_val = io.y - io.x - AB::Expr::ONE;

        // each limb of diff_decomp will be range checked
        // deg(composed) = 1
        let composed = self.compose::<AB>(diff_decomp);

        // constrain that y - x - 1 is equal to the composed value.
        // this enforces that the intermediate value is in the range [0, 2^max_bits - 1], which is
        // equivalent to x < y
        builder.when(io.count).assert_eq(intermed_val, composed);
        // the degree of this constraint is expected to be deg(count) + max(deg(intermed_val),
        // deg(composed)) since we are constraining count * intermed_val == count * composed
    }

    #[inline(always)]
    fn eval_range_checks<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        diff_decomp: &[AB::Var],
        count: impl Into<AB::Expr>,
    ) {
        let count = count.into();
        let mut bits_remaining = self.max_bits;
        // we range check the limbs of diff_decomp so that we know each element
        // of diff_decomp has the correct number of bits
        for limb in diff_decomp {
            // the last limb might have fewer than `bus.range_max_bits` bits
            let range_bits = bits_remaining.min(self.range_max_bits());
            self.bus
                .range_check(*limb, range_bits)
                .eval(builder, count.clone());
            bits_remaining = bits_remaining.saturating_sub(self.range_max_bits());
        }
    }
}

impl<AB: InteractionBuilder> SubAir<AB> for AssertLtSubAir {
    type AirContext<'a>
        = (AssertLessThanIo<AB::Expr>, &'a [AB::Var])
    where
        AB::Expr: 'a,
        AB::Var: 'a,
        AB: 'a;

    // constrain that x < y
    // warning: send for range check must be included for the constraints to be sound
    fn eval<'a>(
        &'a self,
        builder: &'a mut AB,
        (io, diff_decomp): (AssertLessThanIo<AB::Expr>, &'a [AB::Var]),
    ) where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        // Note: every AIR that uses this sub-AIR must include the range checks for soundness
        self.eval_range_checks(builder, diff_decomp, io.count.clone());
        self.eval_composition(builder, io, diff_decomp);
    }
}

impl<F: Field> TraceSubRowGenerator<F> for AssertLtSubAir {
    /// (range_checker, x, y)
    // x, y are u32 because memory records are storing u32 and there would be needless conversions.
    // It also prevents a F: PrimeField32 trait bound.
    type TraceContext<'a> = (&'a VariableRangeCheckerChip, u32, u32);
    /// diff_decomp
    type ColsMut<'a> = &'a mut [F];

    /// Should only be used when `io.count != 0` i.e. only on non-padding rows.
    #[inline(always)]
    fn generate_subrow<'a>(
        &'a self,
        (range_checker, x, y): (&'a VariableRangeCheckerChip, u32, u32),
        diff_decomp: &'a mut [F],
    ) {
        debug_assert!(x < y, "assert {x} < {y} failed");
        debug_assert_eq!(diff_decomp.len(), self.decomp_limbs);
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

        // Note: if x < y then y - x - 1 should already have <= max_bits bits
        range_checker.decompose(y - x - 1, self.max_bits, diff_decomp);
    }
}
