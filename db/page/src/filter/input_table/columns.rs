use afs_primitives::{
    is_equal_vec::columns::IsEqualVecAuxCols,
    is_less_than_tuple::{columns::IsLessThanTupleAuxCols, IsLessThanTupleAir},
};

use crate::common::{
    comp::{
        columns::{EqCompAuxCols, StrictCompAuxCols},
        Comp,
    },
    page_cols::PageCols,
};

pub enum FilterInputTableAuxCols<T> {
    Lt(StrictCompAuxCols<T>),
    Lte(StrictCompAuxCols<T>),
    Eq(EqCompAuxCols<T>),
    Gte(StrictCompAuxCols<T>),
    Gt(StrictCompAuxCols<T>),
}

pub struct FilterInputTableLocalCols<T> {
    pub x: Vec<T>,
    pub satisfies_pred: T,
    pub send_row: T,
    pub aux_cols: FilterInputTableAuxCols<T>,
}

impl<T: Clone> FilterInputTableLocalCols<T> {
    pub fn from_slice(
        slc: &[T],
        start_col: usize,
        end_col: usize,
        limb_bits: &[usize],
        decomp: usize,
        cmp: Comp,
    ) -> Self {
        let select_len = end_col - start_col;
        let x = slc[0..select_len].to_vec();
        let satisfies_pred = slc[select_len].clone();
        let send_row = slc[select_len + 1].clone();

        let aux_cols = match cmp {
            Comp::Lt | Comp::Gte => FilterInputTableAuxCols::Lt(StrictCompAuxCols {
                is_less_than_tuple_aux: IsLessThanTupleAuxCols::from_slice(
                    &slc[select_len + 2..],
                    &IsLessThanTupleAir::new(0, limb_bits.to_vec(), decomp),
                ),
            }),
            Comp::Gt | Comp::Lte => FilterInputTableAuxCols::Gt(StrictCompAuxCols {
                is_less_than_tuple_aux: IsLessThanTupleAuxCols::from_slice(
                    &slc[select_len + 2..],
                    &IsLessThanTupleAir::new(0, limb_bits.to_vec(), decomp),
                ),
            }),
            Comp::Eq => FilterInputTableAuxCols::Eq(EqCompAuxCols {
                is_equal_vec_aux: IsEqualVecAuxCols::from_slice(&slc[select_len + 2..], select_len),
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

pub struct FilterInputCols<T> {
    pub page_cols: PageCols<T>,
    pub local_cols: FilterInputTableLocalCols<T>,
    pub start_col: usize,
    pub end_col: usize,
}

impl<T: Clone> FilterInputCols<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn from_partitioned_slice(
        page_slc: &[T],
        aux_slc: &[T],
        idx_len: usize,
        data_len: usize,
        start_col: usize,
        end_col: usize,
        idx_limb_bits: &[usize],
        decomp: usize,
        cmp: Comp,
    ) -> Self {
        let page_cols = PageCols::from_slice(page_slc, idx_len, data_len);
        let local_cols = FilterInputTableLocalCols::from_slice(
            aux_slc,
            start_col,
            end_col,
            idx_limb_bits,
            decomp,
            cmp,
        );

        Self {
            page_cols,
            local_cols,
            start_col,
            end_col,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_slice(
        slc: &[T],
        idx_len: usize,
        data_len: usize,
        start_col: usize,
        end_col: usize,
        idx_limb_bits: &[usize],
        decomp: usize,
        cmp: Comp,
    ) -> Self {
        let page_width = 1 + idx_len + data_len;
        Self::from_partitioned_slice(
            &slc[..page_width],
            &slc[page_width..],
            idx_len,
            data_len,
            start_col,
            end_col,
            idx_limb_bits,
            decomp,
            cmp,
        )
    }
    pub fn get_width(
        idx_len: usize,
        data_len: usize,
        start_col: usize,
        end_col: usize,
        idx_limb_bits: &[usize],
        decomp: usize,
        cmp: Comp,
    ) -> usize {
        let select_len = end_col - start_col;
        match cmp {
            Comp::Lt | Comp::Lte | Comp::Gt | Comp::Gte => {
                1 + idx_len
                    + data_len
                    + select_len
                    + 1
                    + 1
                    + IsLessThanTupleAuxCols::<T>::width(&IsLessThanTupleAir::new(
                        0,
                        idx_limb_bits.to_vec(),
                        decomp,
                    ))
            }
            Comp::Eq => {
                1 + idx_len
                    + data_len
                    + select_len
                    + 1
                    + 1
                    + IsEqualVecAuxCols::<T>::width(select_len)
            }
        }
    }
}
