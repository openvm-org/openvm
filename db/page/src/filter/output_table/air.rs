use p3_air::BaseAir;
use p3_field::Field;

use crate::indexed_output_page_air::IndexedOutputPageAir;

/// Wrapper around [IndexedOutputPageAir] that receives each row of the page with
/// multiplicity `is_alloc`.
pub struct FilterOutputTableAir {
    pub page_bus_index: usize,
    pub inner: IndexedOutputPageAir,
    // pub lt_air: IsLessThanTupleAir,
    // pub idx_len: usize,
    // pub data_len: usize,
    // pub start_col: usize,
    // pub end_col: usize,
    // pub idx_limb_bits: usize,
    // pub idx_decomp: usize,
}

impl FilterOutputTableAir {
    // pub fn new(
    //     page_bus_index: usize,
    //     range_bus_index: usize,
    //     idx_len: usize,
    //     data_len: usize,
    //     start_col: usize,
    //     end_col: usize,
    //     idx_limb_bits: usize,
    //     idx_decomp: usize,
    // ) -> Self {
    //     Self {
    //         page_bus_index,
    //         lt_air: IsLessThanTupleAir::new(
    //             range_bus_index,
    //             vec![idx_limb_bits; idx_len],
    //             idx_decomp,
    //         ),
    //         idx_len,
    //         data_len,
    //         start_col,
    //         end_col,
    //         idx_limb_bits,
    //         idx_decomp,
    //     }
    // }

    pub fn page_width(&self) -> usize {
        // 1 + self.idx_len + self.data_len
        self.inner.page_width()
    }

    pub fn aux_width(&self) -> usize {
        // IsLessThanTupleAuxCols::<usize>::width(&self.lt_air) + 1
        self.inner.aux_width()
    }

    pub fn air_width(&self) -> usize {
        self.page_width() + self.aux_width()
    }
}

impl<F: Field> BaseAir<F> for FilterOutputTableAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.inner)
    }
}
