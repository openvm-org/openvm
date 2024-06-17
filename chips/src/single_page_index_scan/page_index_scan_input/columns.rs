use crate::{
    common::page_cols::PageCols, is_equal_vec::columns::IsEqualVecAuxCols,
    is_less_than_tuple::columns::IsLessThanTupleAuxCols,
};

use super::Comp;

pub struct StrictCompAuxCols<T> {
    pub is_less_than_tuple_aux: IsLessThanTupleAuxCols<T>,
}

pub struct NonStrictCompAuxCols<T> {
    pub satisfies_strict_comp: T,
    pub satisfies_eq_comp: T,
    pub is_less_than_tuple_aux: IsLessThanTupleAuxCols<T>,
    pub is_equal_vec_aux: IsEqualVecAuxCols<T>,
}

pub struct EqCompAuxCols<T> {
    pub is_equal_vec_aux: IsEqualVecAuxCols<T>,
}

pub enum PageIndexScanInputAuxCols<T> {
    Lt(StrictCompAuxCols<T>),
    Lte(NonStrictCompAuxCols<T>),
    Eq(EqCompAuxCols<T>),
    Gte(NonStrictCompAuxCols<T>),
    Gt(StrictCompAuxCols<T>),
}

pub struct PageIndexScanInputCols<T> {
    pub page_cols: PageCols<T>,
    pub x: Vec<T>,
    pub satisfies_pred: T,
    pub send_row: T,
    pub aux_cols: PageIndexScanInputAuxCols<T>,
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
        let page_cols = PageCols {
            is_alloc: slc[0].clone(),
            idx: slc[1..idx_len + 1].to_vec(),
            data: slc[idx_len + 1..idx_len + data_len + 1].to_vec(),
        };

        let x = slc[idx_len + data_len + 1..2 * idx_len + data_len + 1].to_vec();
        let satisfies_pred = slc[2 * idx_len + data_len + 1].clone();
        let send_row = slc[2 * idx_len + data_len + 2].clone();

        let aux_cols = match cmp {
            Comp::Lt => PageIndexScanInputAuxCols::Lt(StrictCompAuxCols {
                is_less_than_tuple_aux: IsLessThanTupleAuxCols::from_slice(
                    &slc[2 * idx_len + data_len + 3..],
                    idx_limb_bits,
                    decomp,
                    idx_len,
                ),
            }),
            Comp::Lte => {
                let less_than_tuple_aux_width =
                    IsLessThanTupleAuxCols::<T>::get_width(idx_limb_bits.clone(), decomp, idx_len);
                PageIndexScanInputAuxCols::Lte(NonStrictCompAuxCols {
                    satisfies_strict_comp: slc[2 * idx_len + data_len + 3].clone(),
                    satisfies_eq_comp: slc[2 * idx_len + data_len + 4].clone(),
                    is_less_than_tuple_aux: IsLessThanTupleAuxCols::from_slice(
                        &slc[2 * idx_len + data_len + 5
                            ..2 * idx_len + data_len + 5 + less_than_tuple_aux_width],
                        idx_limb_bits,
                        decomp,
                        idx_len,
                    ),
                    is_equal_vec_aux: IsEqualVecAuxCols::from_slice(
                        &slc[2 * idx_len + data_len + 5 + less_than_tuple_aux_width..],
                        idx_len,
                    ),
                })
            }
            Comp::Eq => PageIndexScanInputAuxCols::Eq(EqCompAuxCols {
                is_equal_vec_aux: IsEqualVecAuxCols::from_slice(
                    &slc[2 * idx_len + data_len + 3..],
                    idx_len,
                ),
            }),
            Comp::Gte => {
                let less_than_tuple_aux_width =
                    IsLessThanTupleAuxCols::<T>::get_width(idx_limb_bits.clone(), decomp, idx_len);
                PageIndexScanInputAuxCols::Gte(NonStrictCompAuxCols {
                    satisfies_strict_comp: slc[2 * idx_len + data_len + 3].clone(),
                    satisfies_eq_comp: slc[2 * idx_len + data_len + 4].clone(),
                    is_less_than_tuple_aux: IsLessThanTupleAuxCols::from_slice(
                        &slc[2 * idx_len + data_len + 5
                            ..2 * idx_len + data_len + 5 + less_than_tuple_aux_width],
                        idx_limb_bits,
                        decomp,
                        idx_len,
                    ),
                    is_equal_vec_aux: IsEqualVecAuxCols::from_slice(
                        &slc[2 * idx_len + data_len + 5 + less_than_tuple_aux_width..],
                        idx_len,
                    ),
                })
            }
            Comp::Gt => PageIndexScanInputAuxCols::Gt(StrictCompAuxCols {
                is_less_than_tuple_aux: IsLessThanTupleAuxCols::from_slice(
                    &slc[2 * idx_len + data_len + 3..],
                    idx_limb_bits,
                    decomp,
                    idx_len,
                ),
            }),
        };

        Self {
            page_cols,
            x,
            satisfies_pred,
            send_row,
            aux_cols,
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
            Comp::Lt | Comp::Gt => {
                1 + idx_len
                    + data_len
                    + idx_len
                    + 1
                    + 1
                    + IsLessThanTupleAuxCols::<T>::get_width(idx_limb_bits, decomp, idx_len)
            }
            Comp::Lte | Comp::Gte => {
                1 + idx_len
                    + data_len
                    + idx_len
                    + 1
                    + 1
                    + 1
                    + 1
                    + IsLessThanTupleAuxCols::<T>::get_width(idx_limb_bits, decomp, idx_len)
                    + IsEqualVecAuxCols::<T>::get_width(idx_len)
            }
            Comp::Eq => {
                1 + idx_len
                    + data_len
                    + idx_len
                    + 1
                    + 1
                    + IsEqualVecAuxCols::<T>::get_width(idx_len)
            }
        }
    }
}
