use openvm_circuit_primitives::SubAir;
use p3_air::AirBuilder;
use p3_field::PrimeCharacteristicRing;
use stark_recursion_circuit_derive::AlignedBorrow;

/// A SubAir that constrains the `is_first` flags for nested for-loops.
///
/// Enabled rows must appear contiguously at the beginning of the trace (no interspersed padding),
/// and the `is_enabled` flag is enforced to be boolean.
///
/// Tracks `DEPTH_MINUS_ONE` loop counters (all loops except the innermost).
#[derive(Default)]
pub struct NestedForLoopSubAir<const DEPTH_MINUS_ONE: usize, const DEPTH_MINUS_TWO: usize>;

#[repr(C)]
#[derive(AlignedBorrow, Copy, Clone, Debug)]
pub struct NestedForLoopIoCols<T, const DEPTH_MINUS_ONE: usize> {
    /// Whether the current row is enabled (i.e. not padding)
    pub is_enabled: T,
    /// Array of loop counters for all parent loops (excludes innermost loop).
    /// For DEPTH=3 (i,j,k loops): contains [i, j].
    /// The innermost loop counter is managed by the caller.
    pub counter: [T; DEPTH_MINUS_ONE],
    /// Array of flags indicating the first row of each loop iteration (excludes outermost loop).
    /// For DEPTH=3 (i,j,k loops): contains [j_is_first, k_is_first].
    /// The outermost loop's `is_first` is handled by `when_first_row()` and has no stored column.
    pub is_first: [T; DEPTH_MINUS_ONE],
}

impl<T, const DEPTH_MINUS_ONE: usize> NestedForLoopIoCols<T, DEPTH_MINUS_ONE> {
    pub fn map_into<S>(self) -> NestedForLoopIoCols<S, DEPTH_MINUS_ONE>
    where
        T: Into<S>,
    {
        NestedForLoopIoCols {
            is_enabled: self.is_enabled.into(),
            counter: self.counter.map(Into::into),
            is_first: self.is_first.map(Into::into),
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Copy, Clone, Debug)]
pub struct NestedForLoopAuxCols<T, const DEPTH_MINUS_TWO: usize> {
    /// Array of flags indicating parent loop transitions (excludes outermost loop).
    /// For DEPTH=3 (i,j,k loops): contains only j_is_transition.
    /// The outermost loop's transitions are handled by `when_transition()` and have no stored
    /// column.
    pub is_transition: [T; DEPTH_MINUS_TWO],
}

impl<T> Default for NestedForLoopAuxCols<T, 0> {
    fn default() -> Self {
        Self { is_transition: [] }
    }
}

impl<T, const DEPTH_MINUS_ONE: usize> NestedForLoopAuxCols<T, DEPTH_MINUS_ONE> {
    pub fn map_into<S>(self) -> NestedForLoopAuxCols<S, DEPTH_MINUS_ONE>
    where
        T: Into<S>,
    {
        NestedForLoopAuxCols {
            is_transition: self.is_transition.map(Into::into),
        }
    }
}

impl<AB: AirBuilder, const DEPTH_MINUS_ONE: usize, const DEPTH_MINUS_TWO: usize> SubAir<AB>
    for NestedForLoopSubAir<DEPTH_MINUS_ONE, DEPTH_MINUS_TWO>
{
    type AirContext<'a>
        = (
        (
            NestedForLoopIoCols<AB::Expr, DEPTH_MINUS_ONE>,
            NestedForLoopIoCols<AB::Expr, DEPTH_MINUS_ONE>,
        ),
        NestedForLoopAuxCols<AB::Expr, DEPTH_MINUS_TWO>,
    )
    where
        AB: 'a,
        AB::Var: 'a,
        AB::Expr: 'a;

    fn eval<'a>(&'a self, builder: &'a mut AB, ctx: Self::AirContext<'a>)
    where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        debug_assert_eq!(DEPTH_MINUS_ONE, DEPTH_MINUS_TWO + 1);

        let ((local_io, next_io), local_aux) = ctx;

        // Enforce boolean enabled flag and forbid re-enabling after a disabled row.
        builder.assert_bool(local_io.is_enabled.clone());
        builder
            .when_transition()
            .when_ne(local_io.is_enabled.clone(), AB::Expr::ONE)
            .assert_zero(next_io.is_enabled.clone());

        for level in 0..DEPTH_MINUS_ONE {
            let counter_diff = next_io.counter[level].clone() - local_io.counter[level].clone();
            let local_is_first = local_io.is_first[level].clone();
            let next_is_first = next_io.is_first[level].clone();

            builder.assert_bool(local_is_first.clone());
            builder.assert_bool(next_is_first.clone());
            builder
                .when(local_io.is_first[level].clone())
                .assert_one(local_io.is_enabled.clone());

            // First row constraint
            let mut builder_when_first_row = if level == 0 {
                builder.when_first_row()
            } else {
                let parent_level = level - 1;
                let parent_is_first = local_io.is_first[parent_level].clone();
                builder.when(parent_is_first)
            };
            self.eval_first_row(&mut builder_when_first_row, &local_io, local_is_first);

            // Transition constraints
            let mut builder_when_transition = if level == 0 {
                builder.when_transition()
            } else {
                let parent_level = level - 1;
                let parent_next_is_first = next_io.is_first[parent_level].clone();
                let parent_is_transition: AB::Expr =
                    Self::local_is_transition(next_io.is_enabled.clone(), parent_next_is_first);

                // Constrain is_transition[parent_level] to equal the calculated
                // parent_is_transition
                builder.assert_eq(
                    local_aux.is_transition[parent_level].clone(),
                    parent_is_transition.clone(),
                );

                builder.when(local_aux.is_transition[parent_level].clone())
            };

            self.eval_transition(
                &mut builder_when_transition,
                &local_io,
                &next_io,
                next_is_first.clone(),
                counter_diff.clone(),
            );

            // At Loop Boundaries (Δcounter ≠ 0)
            builder
                .when(next_io.is_enabled.clone())
                .when(counter_diff)
                .assert_one(next_is_first.clone());
        }
    }
}

impl<const DEPTH_MINUS_ONE: usize, const DEPTH_MINUS_TWO: usize>
    NestedForLoopSubAir<DEPTH_MINUS_ONE, DEPTH_MINUS_TWO>
{
    /// Evaluates first row constraint for a loop level.
    ///
    /// Constraint: First row enabled sets `is_first`
    fn eval_first_row<AB: AirBuilder>(
        &self,
        builder_first_row: &mut AB,
        local_io: &NestedForLoopIoCols<AB::Expr, DEPTH_MINUS_ONE>,
        local_is_first: AB::Expr,
    ) {
        builder_first_row
            .when(local_io.is_enabled.clone())
            .assert_one(local_is_first);
    }

    /// Evaluates transition constraints for a loop level.
    ///
    /// Constraints:
    /// 1. When the next row remains enabled, `counter[level]` increments by 0 or 1
    /// 2. Within Loop: `is_first` not set within iteration
    fn eval_transition<AB: AirBuilder>(
        &self,
        builder_transition: &mut AB,
        local_io: &NestedForLoopIoCols<AB::Expr, DEPTH_MINUS_ONE>,
        next_io: &NestedForLoopIoCols<AB::Expr, DEPTH_MINUS_ONE>,
        next_is_first: AB::Expr,
        counter_diff: AB::Expr,
    ) {
        // 1. Base Constraint: `counter[level]` increments by 0 or 1 while enabled
        builder_transition
            .when(next_io.is_enabled.clone())
            .assert_bool(counter_diff.clone());

        // 2. Within Loop (Δcounter ≠ 1): `is_first` not set within iteration
        builder_transition
            .when(local_io.is_enabled.clone())
            .when_ne(counter_diff, AB::Expr::ONE)
            .assert_zero(next_is_first);
    }

    /// Returns an expression for `is_transition` on enabled rows.
    ///
    /// True when:
    /// - The next row is enabled, AND
    /// - The next row does not have `is_first` set (continuation within same loop iteration)
    pub fn local_is_transition<FA>(
        next_is_enabled: impl Into<FA>,
        next_is_first: impl Into<FA>,
    ) -> FA
    where
        FA: PrimeCharacteristicRing,
    {
        next_is_enabled.into() - next_is_first.into()
    }

    /// Returns an expression for `is_last` on enabled rows.
    ///
    /// Equivalent to `!is_transition`.
    /// True when either:
    /// - The next row is disabled, OR
    /// - The next row is enabled and has `is_first` set (boundary between loop iterations)
    pub fn local_is_last<FA>(
        local_is_enabled: impl Into<FA>,
        next_is_enabled: impl Into<FA>,
        next_is_first: impl Into<FA>,
    ) -> FA
    where
        FA: PrimeCharacteristicRing,
    {
        local_is_enabled.into() - next_is_enabled.into() + next_is_first.into()
    }
}
