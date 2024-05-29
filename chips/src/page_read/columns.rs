pub struct PageReadCols<T> {
    pub mult: T,
    pub index: T,
    pub page_row: Vec<T>,
}

/// Columns are
/// [page] | [index] | [mult]
/// This gets partitioned into a chached trace part ([page]) and a main trace part ([index] | [mult])
impl<T> PageReadCols<T> {
    pub fn cols_numbered(cols: &[usize]) -> PageReadCols<usize> {
        assert!(cols.len() >= 2);
        PageReadCols {
            mult: cols[cols.len() - 1],
            index: cols[cols.len() - 2],
            page_row: cols[0..cols.len() - 2].to_vec(),
        }
    }
}
