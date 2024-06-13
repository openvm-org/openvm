use crate::final_page::FinalPageAir;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Clone)]
pub struct MyFinalPageAir {
    t1_output_bus_index: usize,
    t2_output_bus_index: usize,

    final_air: FinalPageAir,
}

impl MyFinalPageAir {
    pub fn new(
        range_bus_index: usize,
        t1_output_bus_index: usize,
        t2_output_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        idx_decomp: usize,
    ) -> Self {
        Self {
            t1_output_bus_index,
            t2_output_bus_index,
            final_air: FinalPageAir::new(
                range_bus_index,
                idx_len,
                data_len,
                idx_limb_bits,
                idx_decomp,
            ),
        }
    }

    pub fn air_width(&self) -> usize {
        self.final_air.air_width()
    }
}
