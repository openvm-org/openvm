use crate::{
    is_equal_vec::columns::IsEqualVecAuxCols, is_less_than_tuple::columns::IsLessThanTupleAuxCols,
};

use super::Comp;

pub enum PageIndexScanInputCols<T> {
    Lt {
        is_alloc: T,
        idx: Vec<T>,
        data: Vec<T>,
        x: Vec<T>,
        satisfies_pred: T,
        send_row: T,
        is_less_than_tuple_aux: IsLessThanTupleAuxCols<T>,
    },

    Eq {
        is_alloc: T,
        idx: Vec<T>,
        data: Vec<T>,
        x: Vec<T>,
        satisfies_pred: T,
        send_row: T,
        is_equal_vec_aux: IsEqualVecAuxCols<T>,
    },

    Gt {
        is_alloc: T,
        idx: Vec<T>,
        data: Vec<T>,
        x: Vec<T>,
        satisfies_pred: T,
        send_row: T,
        is_less_than_tuple_aux: IsLessThanTupleAuxCols<T>,
    },
}

impl<T: Clone> PageIndexScanInputCols<T> {
    pub fn from_slice(
        slc: &[T],
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: Vec<usize>,
        decomp: usize,
        cmp: Comp,
    ) -> Self {
        match cmp {
            Comp::Lt => Self::Lt {
                is_alloc: slc[0].clone(),
                idx: slc[1..idx_len + 1].to_vec(),
                data: slc[idx_len + 1..idx_len + data_len + 1].to_vec(),
                x: slc[idx_len + data_len + 1..2 * idx_len + data_len + 1].to_vec(),
                satisfies_pred: slc[2 * idx_len + data_len + 1].clone(),
                send_row: slc[2 * idx_len + data_len + 2].clone(),
                is_less_than_tuple_aux: IsLessThanTupleAuxCols::from_slice(
                    &slc[2 * idx_len + data_len + 3..],
                    idx_limb_bits,
                    decomp,
                    idx_len,
                ),
            },
            Comp::Eq => Self::Eq {
                is_alloc: slc[0].clone(),
                idx: slc[1..idx_len + 1].to_vec(),
                data: slc[idx_len + 1..idx_len + data_len + 1].to_vec(),
                x: slc[idx_len + data_len + 1..2 * idx_len + data_len + 1].to_vec(),
                satisfies_pred: slc[2 * idx_len + data_len + 1].clone(),
                send_row: slc[2 * idx_len + data_len + 2].clone(),
                is_equal_vec_aux: IsEqualVecAuxCols {
                    prods: slc[2 * idx_len + data_len + 3..3 * idx_len + data_len + 3].to_vec(),
                    invs: slc[3 * idx_len + data_len + 3..].to_vec(),
                },
            },
            Comp::Gt => Self::Gt {
                is_alloc: slc[0].clone(),
                idx: slc[1..idx_len + 1].to_vec(),
                data: slc[idx_len + 1..idx_len + data_len + 1].to_vec(),
                x: slc[idx_len + data_len + 1..2 * idx_len + data_len + 1].to_vec(),
                satisfies_pred: slc[2 * idx_len + data_len + 1].clone(),
                send_row: slc[2 * idx_len + data_len + 2].clone(),
                is_less_than_tuple_aux: IsLessThanTupleAuxCols::from_slice(
                    &slc[2 * idx_len + data_len + 3..],
                    idx_limb_bits,
                    decomp,
                    idx_len,
                ),
            },
        }
    }

    pub fn get_width(
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: Vec<usize>,
        decomp: usize,
        cmp: Comp,
    ) -> usize {
        match cmp {
            Comp::Lt => {
                1 + idx_len
                    + data_len
                    + idx_len
                    + 1
                    + 1
                    + IsLessThanTupleAuxCols::<T>::get_width(idx_limb_bits, decomp, idx_len)
            }
            Comp::Eq => 1 + idx_len + data_len + idx_len + 1 + 1 + 2 * idx_len,
            Comp::Gt => {
                1 + idx_len
                    + data_len
                    + idx_len
                    + 1
                    + 1
                    + IsLessThanTupleAuxCols::<T>::get_width(idx_limb_bits, decomp, idx_len)
            }
        }
    }
}
