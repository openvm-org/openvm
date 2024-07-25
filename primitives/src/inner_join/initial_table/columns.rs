use crate::common::page_cols::PageCols;

#[derive(Debug)]
pub struct TableCols<T> {
    /// Partition 0
    pub page_cols: PageCols<T>,

    /// Partition 1
    /// The multiplicity with which we will send (idx, data) to output_chip
    pub out_mult: T,
}

impl<T: Clone> TableCols<T> {
    pub fn from_partitioned_slice(
        page: &[T],
        aux: &[T],
        idx_len: usize,
        data_len: usize,
    ) -> TableCols<T> {
        TableCols {
            page_cols: PageCols::from_slice(page, idx_len, data_len),
            out_mult: aux[0].clone(),
        }
    }
}
