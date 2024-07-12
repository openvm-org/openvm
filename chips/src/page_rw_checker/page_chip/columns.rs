#[derive(Clone)]
pub struct PageCols<T> {
    pub is_alloc: T, // indicates if row is allocated
    pub idx: Vec<T>,
    pub data: Vec<T>,
}

impl<T> PageCols<T> {
    pub fn cols_numbered(cols: &[usize], idx_len: usize, data_len: usize) -> PageCols<usize> {
        PageCols {
            is_alloc: cols[0],
            idx: cols[1..idx_len + 1].to_vec(),
            data: cols[idx_len + 1..idx_len + data_len + 1].to_vec(),
        }
    }
    pub fn from_slice(cols: &[T], idx_len: usize, data_len: usize) -> PageCols<T>
    where
        T: Clone,
    {
        PageCols {
            is_alloc: cols[0].clone(),
            idx: cols[1..idx_len + 1].to_vec(),
            data: cols[idx_len + 1..idx_len + data_len + 1].to_vec(),
        }
    }
}
