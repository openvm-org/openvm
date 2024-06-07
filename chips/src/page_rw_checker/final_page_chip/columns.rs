use crate::{
    is_less_than_tuple::columns::IsLessThanTupleAuxCols,
    page_rw_checker::page_chip::columns::PageCols,
};

pub struct FinalPageCols<T> {
    pub page_cols: PageCols<T>,
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
                idx_len + 1,
            ),
        }
    }
}

pub struct FinalPageAuxCols<T> {
    pub lt_cols: IsLessThanTupleAuxCols<T>,
    pub lt_out: T,
}

impl<T: Clone> FinalPageAuxCols<T> {
    // TODO: pass here a vector limb_bits
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
}
