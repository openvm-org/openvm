use afs_primitives::is_less_than_tuple::{columns::IsLessThanTupleAuxCols, IsLessThanTupleAir};

pub struct FilterOutputTableAir {
    pub page_bus_index: usize,
    pub lt_air: IsLessThanTupleAir,
    pub idx_len: usize,
    pub data_len: usize,
    pub idx_limb_bits: usize,
    pub idx_decomp: usize,
}

impl FilterOutputTableAir {
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
