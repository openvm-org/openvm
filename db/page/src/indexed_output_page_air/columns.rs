use std::iter;

use afs_primitives::is_less_than_tuple::columns::IsLessThanTupleAuxCols;

use crate::common::page_cols::PageCols;

use super::IndexedOutputPageAir;

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
        indexed_page_air: &IndexedOutputPageAir,
    ) -> IndexedOutputPageCols<T> {
        let idx_len = indexed_page_air.idx_len;
        let data_len = indexed_page_air.data_len;

        Self::from_partitioned_slice(
            &slc[..1 + idx_len + data_len],
            &slc[1 + idx_len + data_len..],
            indexed_page_air,
        )
    }

    pub fn from_partitioned_slice(
        page: &[T],
        other: &[T],
        indexed_page_air: &IndexedOutputPageAir,
    ) -> IndexedOutputPageCols<T> {
        let idx_len = indexed_page_air.idx_len;
        let data_len = indexed_page_air.data_len;

        IndexedOutputPageCols {
            page_cols: PageCols::from_slice(page, idx_len, data_len),
            aux_cols: IndexedOutputPageAuxCols::from_slice(other, indexed_page_air),
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
        indexed_page_air: &IndexedOutputPageAir,
    ) -> IndexedOutputPageAuxCols<T> {
        IndexedOutputPageAuxCols {
            lt_cols: IsLessThanTupleAuxCols::from_slice(
                &slc[..slc.len() - 1],
                &indexed_page_air.lt_air,
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
