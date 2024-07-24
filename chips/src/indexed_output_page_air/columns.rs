use std::iter;

use crate::{common::page_cols::PageCols, is_less_than_tuple::columns::IsLessThanTupleAuxCols};

#[derive(Clone)]
pub struct IndexedOutputPageCols<T> {
    /// The columns for the page itself
    pub page_cols: PageCols<T>,
    /// The auxiliary columns used for ensuring sorting
    pub aux_cols: IndexedOutputPageAuxCols<T>,
}

impl<T: Clone> IndexedOutputPageCols<T> {
    pub fn from_slice(
        slc: &[T],
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        decomp: usize,
    ) -> IndexedOutputPageCols<T> {
        Self::from_partitioned_slice(
            &slc[..1 + idx_len + data_len],
            &slc[1 + idx_len + data_len..],
            idx_len,
            data_len,
            idx_limb_bits,
            decomp,
        )
    }
    pub fn from_partitioned_slice(
        page: &[T],
        other: &[T],
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        decomp: usize,
    ) -> IndexedOutputPageCols<T> {
        IndexedOutputPageCols {
            page_cols: PageCols::from_slice(page, idx_len, data_len),
            aux_cols: IndexedOutputPageAuxCols::from_slice(other, idx_limb_bits, decomp, idx_len),
        }
    }
}

#[derive(Clone)]
pub struct IndexedOutputPageAuxCols<T> {
    pub lt_cols: IsLessThanTupleAuxCols<T>, // auxiliary columns used for lt_out
    pub lt_out: T, // this bit indicates whether the idx in this row is greater than the idx in the previous row
}

impl<T: Clone> IndexedOutputPageAuxCols<T> {
    pub fn from_slice(
        slc: &[T],
        idx_limb_bits: usize,
        decomp: usize,
        idx_len: usize,
    ) -> IndexedOutputPageAuxCols<T> {
        IndexedOutputPageAuxCols {
            lt_cols: IsLessThanTupleAuxCols::from_slice(
                &slc[..slc.len() - 1],
                &vec![idx_limb_bits; idx_len],
                decomp,
            ),
            lt_out: slc[slc.len() - 1].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        self.lt_cols
            .flatten()
            .into_iter()
            .chain(iter::once(self.lt_out.clone()))
            .collect()
    }
}
