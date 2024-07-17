use crate::is_equal_vec::columns::IsEqualVecAuxCols;
use crate::is_less_than_tuple::columns::IsLessThanTupleAuxCols;

mod air;
mod bridge;
mod columns;
mod trace;

pub trait OfflineCheckerOperation<F> {
    fn get_timestamp(&self) -> usize;
    fn get_idx(&self) -> Vec<F>;
    fn get_data(&self) -> Vec<F>;
    fn get_op_type(&self) -> u8;
}

pub struct OfflineChecker<const WORD_SIZE: usize> {
    idx_clk_limb_bits: Vec<usize>,
    decomp: usize,
    idx_len: usize,
    data_len: usize,
    range_bus: usize,
    ops_bus: usize,
}

impl<const WORD_SIZE: usize> OfflineChecker<WORD_SIZE> {
    pub fn air_width(&self) -> usize {
        7 + self.idx_len
            + self.data_len
            + IsEqualVecAuxCols::<usize>::get_width(self.idx_len)
            + IsEqualVecAuxCols::<usize>::get_width(self.data_len)
            + IsLessThanTupleAuxCols::<usize>::get_width(
                self.idx_clk_limb_bits.clone(),
                self.decomp,
                3,
            )
    }
}
