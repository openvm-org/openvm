use p3_air::AirBuilder;

pub trait AirConfig {
    /// Column struct over generic type
    type Cols<T>;
}

/// Trait with associated types intended to allow re-use of constraint logic
/// inside other AIRs.
pub trait SubAir<AB: AirBuilder> {
    /// View of the parts of matrix relevant for IO.
    /// Typically this is either 'local' IO columns or 'local' and 'next' IO columns.
    type IoView;
    /// View of auxiliary parts of matrix necessary for constraint evaluation.
    /// Typically this is either a subset of 'local' columns or subset of 'local' and 'next' columns.
    type AuxView;

    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView);
}

/// This is a helper for simple trace row generation.
/// Not every AIR will need this.
pub trait LocalTraceInstructions<F>: AirConfig {
    /// Logical inputs needed to generate a single row of the trace.
    type LocalInput;

    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F>;
}
