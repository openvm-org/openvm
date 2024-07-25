use crate::indexed_output_page_air::{columns::IndexedOutputPageCols, IndexedOutputPageAir};

// Minimal wrapper around IndexedOutputPageCols
pub struct ReceivingIndexedOutputPageCols<T> {
    pub final_page_cols: IndexedOutputPageCols<T>,
}

impl<T: Clone> ReceivingIndexedOutputPageCols<T> {
    pub fn from_slice(slc: &[T], final_air: &IndexedOutputPageAir) -> Self {
        Self {
            final_page_cols: IndexedOutputPageCols::from_slice(
                &slc[..slc.len()],
                &IndexedOutputPageAir::new(
                    0,
                    final_air.idx_len,
                    final_air.data_len,
                    final_air.idx_limb_bits,
                    final_air.idx_decomp,
                ),
            ),
        }
    }
}
