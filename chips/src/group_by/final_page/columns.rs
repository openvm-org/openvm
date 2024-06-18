use crate::final_page::{columns::FinalPageCols, FinalPageAir};

// Minimal wrapper around FinalPageCols
pub struct MyFinalPageCols<T> {
    pub final_page_cols: FinalPageCols<T>,
}

impl<T: Clone> MyFinalPageCols<T> {
    pub fn from_slice(slc: &[T], final_air: FinalPageAir) -> Self {
        Self {
            final_page_cols: FinalPageCols::from_slice(
                &slc[..slc.len()],
                final_air.idx_len,
                final_air.data_len,
                final_air.idx_limb_bits,
                final_air.idx_decomp,
            ),
        }
    }
}
