use crate::final_page::FinalPageAir;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Clone, Debug)]
pub(crate) struct MyFinalPageAir {
    page_bus_index: usize,

    final_air: FinalPageAir,
}

impl MyFinalPageAir {
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
            final_air: FinalPageAir::new(
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
