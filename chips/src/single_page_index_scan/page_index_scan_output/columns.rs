use crate::indexed_output_page_air::{columns::IndexedOutputPageCols, IndexedOutputPageAir};

pub struct PageIndexScanOutputCols<T> {
    pub final_page_cols: IndexedOutputPageCols<T>,
}

impl<T: Clone> PageIndexScanOutputCols<T> {
    pub fn from_slice(slc: &[T], final_page_air: &IndexedOutputPageAir) -> Self {
        Self {
            final_page_cols: IndexedOutputPageCols::from_slice(
                slc,
                final_page_air.idx_len,
                final_page_air.data_len,
                final_page_air.idx_limb_bits,
                final_page_air.idx_decomp,
            ),
        }
    }

    pub fn get_width(final_page_air: &IndexedOutputPageAir) -> usize {
        final_page_air.air_width()
    }
}
