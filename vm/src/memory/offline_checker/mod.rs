use afs_chips::is_less_than_tuple::columns::IsLessThanTupleAuxCols;

mod air;
mod bridge;
mod columns;
mod trace;

pub struct OfflineChecker {
    data_len: usize,
    addr_clk_limb_bits: Vec<usize>,
    decomp: usize,
}

impl OfflineChecker {
    pub fn new(
        data_len: usize,
        addr_space_limb_bits: usize,
        pointer_limb_bits: usize,
        clk_limb_bits: usize,
        decomp: usize,
    ) -> Self {
        Self {
            data_len,
            addr_clk_limb_bits: vec![addr_space_limb_bits, pointer_limb_bits, clk_limb_bits],
            decomp,
        }
    }

    pub fn mem_width(&self) -> usize {
        // 1 for addr_space, 1 for pointer, data_len for data
        2 + self.data_len
    }

    pub fn air_width(&self) -> usize {
        10 + self.mem_width()
            + 2 * self.data_len
            + IsLessThanTupleAuxCols::<usize>::get_width(
                self.addr_clk_limb_bits.clone(),
                self.decomp,
                3,
            )
    }
}
