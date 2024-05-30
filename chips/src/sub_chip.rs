use afs_stark_backend::interaction::Interaction;
use p3_air::AirBuilder;
use p3_field::Field;

pub trait AirConfig {
    /// Column struct over generic type
    type Cols<T>;
}

/// Trait with associated types intended to allow re-use of constraint logic
/// inside other AIRs.
pub trait SubAir<AB: AirBuilder>: AirConfig {
    fn eval(&self, builder: &mut AB, cols: <Self as AirConfig>::Cols<AB::Var>);
}

pub trait LocalTraceInstructions<F>: AirConfig {
    /// Logical inputs needed to generate a single row of the trace.
    type LocalInput;

    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F>;
}

pub trait SubAirWithInteractions<F: Field>: AirConfig {
    fn sends(&self, col_indices: Self::Cols<usize>) -> Vec<Interaction<F>> {
        let _ = col_indices;
        vec![]
    }
    fn receives(&self, col_indices: Self::Cols<usize>) -> Vec<Interaction<F>> {
        let _ = col_indices;
        vec![]
    }
}
