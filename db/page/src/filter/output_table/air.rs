use p3_air::BaseAir;
use p3_field::Field;

use crate::indexed_output_page_air::IndexedOutputPageAir;

/// Wrapper around [IndexedOutputPageAir] that receives each row of the page with
/// multiplicity `is_alloc`.
pub struct FilterOutputTableAir {
    pub page_bus_index: usize,
    pub final_air: IndexedOutputPageAir,
}

impl FilterOutputTableAir {
    pub fn page_width(&self) -> usize {
        self.final_air.page_width()
    }

    pub fn aux_width(&self) -> usize {
        self.final_air.aux_width()
    }

    pub fn air_width(&self) -> usize {
        self.page_width() + self.aux_width()
    }
}

impl<F: Field> BaseAir<F> for FilterOutputTableAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.final_air)
    }
}
