use crate::indexed_output_page_air::IndexedOutputPageAir;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

<<<<<<< HEAD:chips/src/page_rw_checker/my_final_page/mod.rs
#[derive(Debug, Clone)]
pub(crate) struct MyFinalPageAir {
=======
#[derive(Clone, Debug)]
pub struct IndexedPageWriteAir {
>>>>>>> d74b0541394676b6966e07196adf50328a41d65b:chips/src/page_rw_checker/final_page/mod.rs
    page_bus_index: usize,

    final_air: IndexedOutputPageAir,
}

impl IndexedPageWriteAir {
    pub fn new(
        page_bus_index: usize,
        range_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        idx_decomp: usize,
    ) -> Self {
        Self {
            page_bus_index,
            final_air: IndexedOutputPageAir::new(
                range_bus_index,
                idx_len,
                data_len,
                idx_limb_bits,
                idx_decomp,
            ),
        }
    }

    pub fn page_width(&self) -> usize {
        self.final_air.page_width()
    }

    pub fn aux_width(&self) -> usize {
        self.final_air.aux_width() + 1
    }

    pub fn air_width(&self) -> usize {
        self.page_width() + self.aux_width()
    }
}
