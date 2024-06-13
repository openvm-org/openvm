pub struct IntersectorCols<T> {
    pub idx: Vec<T>,
    pub t1_mult: T,
    pub t2_mult: T,
    pub out_mult: T,
    pub is_extra: T,
}

impl<T: Clone> IntersectorCols<T> {
    pub fn from_slice(slc: &[T], idx_len: usize) -> Self {
        Self {
            idx: slc[..idx_len].to_vec(),
            t1_mult: slc[idx_len].clone(),
            t2_mult: slc[idx_len + 1].clone(),
            out_mult: slc[idx_len + 2].clone(),
            is_extra: slc[idx_len + 3].clone(),
        }
    }
}
