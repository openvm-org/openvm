use crate::{
    common::page_cols::PageCols,
    is_equal_vec::columns::IsEqualVecAuxCols,
    is_less_than_tuple::{columns::IsLessThanTupleAuxCols, IsLessThanTupleAir},
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

pub struct PageIndexScanInputLocalCols<T> {
    pub x: Vec<T>,
    pub satisfies_pred: T,
    pub send_row: T,
    pub aux_cols: PageIndexScanInputAuxCols<T>,
}

impl<T: Clone> PageIndexScanInputLocalCols<T> {
    pub fn from_slice(
        slc: &[T],
        idx_len: usize,
        idx_limb_bits: &[usize],
        decomp: usize,
        cmp: Comp,
    ) -> Self {
        let x = slc[0..idx_len].to_vec();
        let satisfies_pred = slc[idx_len].clone();
        let send_row = slc[idx_len + 1].clone();

        let aux_cols = match cmp {
            Comp::Lt => PageIndexScanInputAuxCols::Lt(StrictCompAuxCols {
                is_less_than_tuple_aux: IsLessThanTupleAuxCols::from_slice(
                    &slc[idx_len + 2..],
                    &IsLessThanTupleAir::new(0, idx_limb_bits.to_vec(), decomp),
                ),
            }),
            Comp::Lte => {
                let less_than_tuple_aux_width = IsLessThanTupleAuxCols::<T>::width(
                    &IsLessThanTupleAir::new(0, idx_limb_bits.to_vec(), decomp),
                );
                PageIndexScanInputAuxCols::Lte(NonStrictCompAuxCols {
                    satisfies_strict_comp: slc[idx_len + 2].clone(),
                    satisfies_eq_comp: slc[idx_len + 3].clone(),
                    is_less_than_tuple_aux: IsLessThanTupleAuxCols::from_slice(
                        &slc[idx_len + 4..idx_len + 4 + less_than_tuple_aux_width],
                        &IsLessThanTupleAir::new(0, idx_limb_bits.to_vec(), decomp),
                    ),
                    is_equal_vec_aux: IsEqualVecAuxCols::from_slice(
                        &slc[idx_len + 4 + less_than_tuple_aux_width..],
                        idx_len,
                    ),
                })
            }
            Comp::Eq => PageIndexScanInputAuxCols::Eq(EqCompAuxCols {
                is_equal_vec_aux: IsEqualVecAuxCols::from_slice(&slc[idx_len + 2..], idx_len),
            }),
            Comp::Gte => {
                let less_than_tuple_aux_width = IsLessThanTupleAuxCols::<T>::width(
                    &IsLessThanTupleAir::new(0, idx_limb_bits.to_vec(), decomp),
                );
                PageIndexScanInputAuxCols::Gte(NonStrictCompAuxCols {
                    satisfies_strict_comp: slc[idx_len + 2].clone(),
                    satisfies_eq_comp: slc[idx_len + 3].clone(),
                    is_less_than_tuple_aux: IsLessThanTupleAuxCols::from_slice(
                        &slc[idx_len + 4..idx_len + 4 + less_than_tuple_aux_width],
                        &IsLessThanTupleAir::new(0, idx_limb_bits.to_vec(), decomp),
                    ),
                    is_equal_vec_aux: IsEqualVecAuxCols::from_slice(
                        &slc[idx_len + 4 + less_than_tuple_aux_width..],
                        idx_len,
                    ),
                })
            }
            Comp::Gt => PageIndexScanInputAuxCols::Gt(StrictCompAuxCols {
                is_less_than_tuple_aux: IsLessThanTupleAuxCols::from_slice(
                    &slc[idx_len + 2..],
                    &IsLessThanTupleAir::new(0, idx_limb_bits.to_vec(), decomp),
                ),
            }),
        };

        Self {
            x,
            satisfies_pred,
            send_row,
            aux_cols,
        }
    }
}

pub struct PageIndexScanInputCols<T> {
    pub page_cols: PageCols<T>,
    pub local_cols: PageIndexScanInputLocalCols<T>,
}

impl<T: Clone> PageIndexScanInputCols<T> {
    pub fn from_partitioned_slice(
        page_slc: &[T],
        aux_slc: &[T],
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: &[usize],
        decomp: usize,
        cmp: Comp,
    ) -> Self {
        let page_cols = PageCols::from_slice(page_slc, idx_len, data_len);
        let local_cols =
            PageIndexScanInputLocalCols::from_slice(aux_slc, idx_len, idx_limb_bits, decomp, cmp);

        Self {
            page_cols,
            local_cols,
        }
    }

    pub fn from_slice(
        slc: &[T],
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: &[usize],
        decomp: usize,
        cmp: Comp,
    ) -> Self {
        Self::from_partitioned_slice(
            &slc[..idx_len + data_len + 1],
            &slc[idx_len + data_len + 1..],
            idx_len,
            data_len,
            idx_limb_bits,
            decomp,
            cmp,
        )
    }

    pub fn get_width(
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: &[usize],
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
                    + IsLessThanTupleAuxCols::<T>::width(&IsLessThanTupleAir::new(
                        0,
                        idx_limb_bits.to_vec(),
                        decomp,
                    ))
            }
            Comp::Lte | Comp::Gte => {
                1 + idx_len
                    + data_len
                    + idx_len
                    + 1
                    + 1
                    + 1
                    + 1
                    + IsLessThanTupleAuxCols::<T>::width(&IsLessThanTupleAir::new(
                        0,
                        idx_limb_bits.to_vec(),
                        decomp,
                    ))
                    + IsEqualVecAuxCols::<T>::width(idx_len)
            }
            Comp::Eq => {
                1 + idx_len + data_len + idx_len + 1 + 1 + IsEqualVecAuxCols::<T>::width(idx_len)
            }
        }
    }
}
