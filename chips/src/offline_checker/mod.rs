use crate::is_equal_vec::columns::IsEqualVecAuxCols;
use crate::is_less_than_tuple::columns::IsLessThanTupleAuxCols;

mod air;
mod bridge;
mod columns;
mod trace;

pub struct OfflineChecker<const WORD_SIZE: usize> {
    addr_clk_limb_bits: Vec<usize>,
    decomp: usize,
    idx_len: usize,
    data_len: usize,
    range_bus: usize,
}

impl<const WORD_SIZE: usize> OfflineChecker<WORD_SIZE> {
    pub fn air_width(&self) -> usize {
        6 + self.idx_len
            + self.data_len
            + IsEqualVecAuxCols::<usize>::get_width(self.idx_len)
            + IsEqualVecAuxCols::<usize>::get_width(self.data_len)
            + IsLessThanTupleAuxCols::<usize>::get_width(
                self.addr_clk_limb_bits.clone(),
                self.decomp,
                3,
            )
    }
}
