mod cubic_interpolation;
mod ext_field_multiply;

pub use cubic_interpolation::{
    InterpolateCubicAuxCols, InterpolateCubicIo, InterpolateCubicSubAir,
};
pub use ext_field_multiply::{ExtFieldMultAuxCols, ExtFieldMultiplyIo, ExtFieldMultiplySubAir};
use p3_air::AirBuilder;

/// Trait for sub-AIRs that encapsulate auxiliary columns and constraints
pub trait SubAir<AB: AirBuilder> {
    /// The context passed to eval, typically (IO, &AuxCols)
    type AirContext<'a>
    where
        Self: 'a,
        AB: 'a,
        AB::Var: 'a,
        AB::Expr: 'a;

    /// Evaluate the constraints for this sub-AIR
    fn eval<'a>(&'a self, builder: &'a mut AB, ctx: Self::AirContext<'a>)
    where
        AB::Var: 'a,
        AB::Expr: 'a;
}

/// Trait for generating trace data for sub-AIR auxiliary columns
pub trait TraceSubRowGenerator<F> {
    /// The context needed to generate the trace
    type TraceContext<'a>
    where
        Self: 'a;
    /// Mutable reference to the auxiliary columns
    type ColsMut<'a>
    where
        Self: 'a;

    /// Generate the trace data for auxiliary columns
    fn generate_subrow<'a>(&'a self, ctx: Self::TraceContext<'a>, sub_row: Self::ColsMut<'a>);
}
