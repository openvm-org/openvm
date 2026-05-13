/// Allows AIRs to indicate the names of their columns, for debugging purposes.
/// A default implementation is provided that returns `None`, indicating that
/// column names are not available.
pub trait ColumnsAir {
    /// If available, returns the names of columns used in this AIR.
    /// If the result is `Some(names)`, `names.len() == air.width()` should always
    /// be true.
    fn columns(&self) -> Option<Vec<String>> {
        None
    }
}
