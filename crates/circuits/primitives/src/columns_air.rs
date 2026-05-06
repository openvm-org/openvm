/// If available, returns the names of columns used in this AIR.
pub trait ColumnsAir {
    /// If the result is `Some(names)`, `names.len() == air.width()` should always
    /// be true.
    fn columns(&self) -> Option<Vec<String>> {
        None
    }
}
