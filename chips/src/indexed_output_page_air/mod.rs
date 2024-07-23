use crate::is_less_than_tuple::columns::IsLessThanTupleAuxCols;

pub mod air;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

/// Air that constraints that the index of the page is strictly ascending.
/// This Air does not receive any data and its only interactions are for range check.
#[derive(Copy, Clone, Debug)]
pub struct IndexedOutputPageAir {
    range_bus_index: usize,

    pub idx_len: usize,
    pub data_len: usize,

    pub idx_limb_bits: usize,
    pub idx_decomp: usize,
}

impl IndexedOutputPageAir {
    pub fn new(
        range_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        idx_decomp: usize,
    ) -> Self {
        Self {
            range_bus_index,
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
            vec![self.idx_limb_bits; self.idx_len],
            self.idx_decomp,
            self.idx_len,
        ) + 1
    }

    pub fn air_width(&self) -> usize {
        self.page_width() + self.aux_width()
    }
}
