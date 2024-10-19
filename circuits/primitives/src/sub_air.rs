use p3_air::AirBuilder;

/// Trait with associated types intended to allow re-use of constraint logic
/// inside other AIRs.
pub trait SubAir<AB: AirBuilder> {
    /// Type to define the context, typically in terms of `AB::Expr` that are needed
    /// to define the SubAir's constraints.
    type AirContext;

    fn eval(&self, builder: &mut AB, ctx: impl Into<Self::AirContext>);
}

/// This is a helper for generation of the trace on a subset of the columns in a single row
/// of the trace matrix.
pub trait TraceSubRowGenerator {
    /// The minimal amount of information needed to generate the sub-row of the trace matrix.
    /// This type has a lifetime so other context, such as references to other chips, can be provided.
    type TraceContext<'a>
    where
        Self: 'a;
    /// The type for the columns to mutate. Often this can be `&'a mut Cols<F>` if `Cols` is on the stack.
    /// For structs that use the heap, this should be a struct that contains mutable slices.
    type ColsMut<'a>
    where
        Self: 'a;

    fn generate_subrow<'a>(&'a self, ctx: Self::TraceContext<'a>, sub_row: Self::ColsMut<'a>);
}
