use crate::common::page_cols::PageCols;

#[derive(Debug)]
pub struct TableCols<T> {
    pub page_cols: PageCols<T>,

    /// The multiplicity with which we will send (idx, data) to output_chip
    pub out_mult: T,
}

impl<T: Clone> TableCols<T> {
    pub fn from_slice(cols: &[T], idx_len: usize, data_len: usize) -> TableCols<T> {
        TableCols {
            page_cols: PageCols::from_slice(&cols[..cols.len() - 1], idx_len, data_len),
            out_mult: cols[idx_len + data_len + 1].clone(),
        }
    }
}
