pub struct TableCols<T> {
    pub is_alloc: T,
    pub idx: Vec<T>,
    pub data: Vec<T>,

    pub mult: T,
}

impl<T: Clone> TableCols<T> {
    pub fn from_slice(cols: &[T], idx_len: usize, data_len: usize) -> TableCols<T> {
        TableCols {
            is_alloc: cols[0].clone(),
            idx: cols[1..idx_len + 1].to_vec(),
            data: cols[idx_len + 1..idx_len + data_len + 1].to_vec(),
            mult: cols[idx_len + data_len + 1].clone(),
        }
    }
}
