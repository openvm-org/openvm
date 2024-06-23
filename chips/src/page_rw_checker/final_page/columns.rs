use crate::indexed_output_page_air::{columns::IndexedOutputPageCols, IndexedOutputPageAir};

pub struct IndexedPageWriteCols<T> {
    /// The columns for IndexedOutputPageAir, which include the page itself
    /// and the extra columns for ensuring sorting
    pub final_page_cols: IndexedOutputPageCols<T>,
    /// The multiplicity with which a row is received on the page_bus
    pub rcv_mult: T,
}

impl<T: Clone> IndexedPageWriteCols<T> {
    pub fn from_slice(slc: &[T], final_air: &IndexedOutputPageAir) -> Self {
        Self::from_partitioned_slice(
            &slc[0..final_air.page_width()],
            &slc[final_air.page_width()..],
            final_air,
        )
    }
    pub fn from_partitioned_slice(
        page: &[T],
        other: &[T],
        final_air: &IndexedOutputPageAir,
    ) -> Self {
        Self {
            final_page_cols: IndexedOutputPageCols::from_partitioned_slice(
                page,
                &other[..other.len() - 1],
                final_air.idx_len,
                final_air.data_len,
                final_air.idx_limb_bits,
                final_air.idx_decomp,
            ),
            rcv_mult: other[other.len() - 1].clone(),
        }
    }
}
