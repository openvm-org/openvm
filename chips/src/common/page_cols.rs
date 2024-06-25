use std::iter;

#[derive(Clone)]
pub struct PageCols<T> {
    pub is_alloc: T, // indicates if row is allocated
    pub idx: Vec<T>,
    pub data: Vec<T>,
}

impl<T: Clone> PageCols<T> {
    pub fn from_slice(cols: &[T], idx_len: usize, data_len: usize) -> PageCols<T> {
        PageCols {
            is_alloc: cols[0].clone(),
            idx: cols[1..idx_len + 1].to_vec(),
            data: cols[idx_len + 1..idx_len + data_len + 1].to_vec(),
        }
    }

    pub fn to_vec(&self) -> Vec<T> {
        iter::once(self.is_alloc.clone())
            .chain(self.idx.clone())
            .chain(self.data.clone())
            .collect()
    }
}
