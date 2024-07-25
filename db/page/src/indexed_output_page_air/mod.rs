use afs_primitives::is_less_than_tuple::{columns::IsLessThanTupleAuxCols, IsLessThanTupleAir};

pub mod air;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

#[derive(Clone)]
pub struct IndexedOutputPageAir {
    pub lt_air: IsLessThanTupleAir,

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
            lt_air: IsLessThanTupleAir::new(
                range_bus_index,
                vec![idx_limb_bits; idx_len],
                idx_decomp,
            ),
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
        IsLessThanTupleAuxCols::<usize>::width(&self.lt_air) + 1
    }

    pub fn air_width(&self) -> usize {
        self.page_width() + self.aux_width()
    }
}
