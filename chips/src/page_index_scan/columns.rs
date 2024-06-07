use crate::is_less_than_tuple::columns::IsLessThanTupleAuxCols;

pub struct PageIndexScanCols<T> {
    pub is_alloc: T,
    pub idx: Vec<T>,
    pub data: Vec<T>,

    pub x: Vec<T>,

    pub satisfies_pred: T,
    pub is_less_than_tuple_aux: IsLessThanTupleAuxCols<T>,
}

impl<T: Clone> PageIndexScanCols<T> {
    pub fn from_slice(
        slc: &[T],
        idx_len: usize,
        data_len: usize,
        decomp: usize,
        limb_bits: Vec<usize>,
    ) -> Self {
        Self {
            is_alloc: slc[0].clone(),
            idx: slc[1..idx_len + 1].to_vec(),
            data: slc[idx_len + 1..idx_len + data_len + 1].to_vec(),
            x: slc[idx_len + data_len + 1..2 * idx_len + data_len + 1].to_vec(),
            satisfies_pred: slc[2 * idx_len + data_len + 1].clone(),
            is_less_than_tuple_aux: IsLessThanTupleAuxCols::from_slice(
                &slc[2 * idx_len + data_len + 2..],
                limb_bits,
                decomp,
                idx_len,
            ),
        }
    }

    pub fn get_width(
        idx_len: usize,
        data_len: usize,
        limb_bits: Vec<usize>,
        decomp: usize,
    ) -> usize {
        1 + idx_len
            + data_len
            + idx_len
            + 1
            + IsLessThanTupleAuxCols::<T>::get_width(limb_bits, decomp, idx_len)
    }
}
