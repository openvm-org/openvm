use p3_air::AirBuilder;

/// Trait with associated types intended to allow re-use of constraint logic
/// inside other AIRs.
pub trait SubAir<AB: AirBuilder> {
    /// Column struct over generic type `T`.
    type Cols<T>;

    fn eval(&self, builder: &mut AB, cols: Self::Cols<AB::Var>);
}
