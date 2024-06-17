use std::iter;

use crate::{common::page_cols::PageCols, is_less_than_tuple::columns::IsLessThanTupleAuxCols};

pub struct FinalPageCols<T> {
    /// The columns for the page itself
    pub page_cols: PageCols<T>,
    /// The auxiliary columns used for ensuring sorting
    pub aux_cols: FinalPageAuxCols<T>,
}

impl<T: Clone> FinalPageCols<T> {
    pub fn from_slice(
        slc: &[T],
        idx_len: usize,
        data_len: usize,
        limb_bits: usize,
        decomp: usize,
    ) -> FinalPageCols<T> {
        FinalPageCols {
            page_cols: PageCols::from_slice(&slc[..1 + idx_len + data_len], idx_len, data_len),
            aux_cols: FinalPageAuxCols::from_slice(
                &slc[1 + idx_len + data_len..],
                limb_bits,
                decomp,
                idx_len,
            ),
        }
    }
}

#[derive(Clone)]
pub struct FinalPageAuxCols<T> {
    pub lt_cols: IsLessThanTupleAuxCols<T>, // auxiliary columns used for lt_out
    pub lt_out: T, // this bit indicates whether the idx in this row is greater than the idx in the previous row
}

impl<T: Clone> FinalPageAuxCols<T> {
    pub fn from_slice(
        slc: &[T],
        limb_bits: usize,
        decomp: usize,
        tuple_len: usize,
    ) -> FinalPageAuxCols<T> {
        FinalPageAuxCols {
            lt_cols: IsLessThanTupleAuxCols::from_slice(
                &slc[..slc.len() - 1],
                vec![limb_bits; tuple_len],
                decomp,
                tuple_len,
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
