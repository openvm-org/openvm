use crate::is_less_than_tuple::columns::IsLessThanTupleAuxCols;

pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

pub struct FinalPageChip {
    page_bus_index: usize,
    pub sorted_bus_index: usize,

    idx_len: usize,
    data_len: usize,

    idx_limb_bits: usize,
    idx_decomp: usize,
}

impl FinalPageChip {
    pub fn new(
        page_bus_index: usize,
        sorted_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        idx_decomp: usize,
    ) -> Self {
        Self {
            page_bus_index,
            sorted_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
        }
    }

    pub fn page_width(&self) -> usize {
        1 + self.idx_len + self.data_len
    }

    pub fn aux_width(&self) -> usize {
        IsLessThanTupleAuxCols::<usize>::get_width(
            vec![self.idx_limb_bits; 1 + self.idx_len],
            self.idx_decomp,
            1 + self.idx_len,
        ) + 1
    }

    pub fn air_width(&self) -> usize {
        self.page_width() + self.aux_width()
    }
}
