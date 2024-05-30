use p3_air::AirBuilder;

/// Trait with associated types intended to allow re-use of constraint logic
/// inside other AIRs.
pub trait SubAir {
    /// Column struct over generic type `T`.
    type Cols<T>;

    fn eval<AB: AirBuilder>(&self, builder: &mut AB, cols: Self::Cols<AB::Var>);
}

pub trait SubAirLocalTraceInstructions<F>: SubAir {
    /// Logical inputs needed to generate a single row of the trace.
    type LocalInput;

    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F>;
}
