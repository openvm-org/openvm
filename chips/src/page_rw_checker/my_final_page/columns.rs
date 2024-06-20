use crate::final_page::{columns::FinalPageCols, FinalPageAir};

pub struct MyFinalPageCols<T> {
    /// The columns for FinalPageAir, which include the page itself
    /// and the extra columns for ensuting sorting
    pub final_page_cols: FinalPageCols<T>,
    /// The multiplicity with which a row is received on the page_bus
    pub rcv_mult: T,
}

impl<T: Clone> MyFinalPageCols<T> {
    pub fn from_slice(slc: &[T], final_air: &FinalPageAir) -> Self {
        Self {
            final_page_cols: FinalPageCols::from_slice(
                &slc[..slc.len() - 1],
                final_air.idx_len,
                final_air.data_len,
                final_air.idx_limb_bits,
                final_air.idx_decomp,
            ),
            rcv_mult: slc[slc.len() - 1].clone(),
        }
    }
}
