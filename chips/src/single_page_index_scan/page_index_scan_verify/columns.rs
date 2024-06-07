pub struct PageIndexScanVerifyCols<T> {
    pub is_alloc: T,
    pub idx: Vec<T>,
    pub data: Vec<T>,
}

impl<T: Clone> PageIndexScanVerifyCols<T> {
    pub fn from_slice(slc: &[T], idx_len: usize, data_len: usize) -> Self {
        Self {
            is_alloc: slc[0].clone(),
            idx: slc[1..idx_len + 1].to_vec(),
            data: slc[idx_len + 1..idx_len + data_len + 1].to_vec(),
        }
    }

    pub fn get_width(idx_len: usize, data_len: usize) -> usize {
        1 + idx_len + data_len
    }
}
