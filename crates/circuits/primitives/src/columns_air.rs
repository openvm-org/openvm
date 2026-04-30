use openvm_stark_backend::p3_air::BaseAir;

/// If available, returns the names of columns used in this AIR.
pub trait ColumnsAir<F>: BaseAir<F> {
    /// If the result is `Some(names)`, `names.len() == air.width()` should always
    /// be true.
    fn columns(&self) -> Option<Vec<String>> {
        None
    }
}
