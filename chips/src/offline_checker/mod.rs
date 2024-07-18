use std::marker::PhantomData;

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
    pub fn new(
        idx_clk_limb_bits: Vec<usize>,
        decomp: usize,
        idx_len: usize,
        data_len: usize,
        range_bus: usize,
        ops_bus: usize,
    ) -> Self {
        Self {
            idx_clk_limb_bits,
            decomp,
            idx_len,
            data_len,
            range_bus,
            ops_bus,
        }
    }

    pub fn idx_data_width(&self) -> usize {
        self.idx_len + self.data_len
    }

    pub fn air_width(&self) -> usize {
        7 + self.idx_len
            + self.data_len
            + IsEqualVecAuxCols::<usize>::get_width(self.idx_len)
            + IsEqualVecAuxCols::<usize>::get_width(self.data_len)
            + IsLessThanTupleAuxCols::<usize>::get_width(
                self.idx_clk_limb_bits.clone(),
                self.decomp,
                self.idx_len + 1,
            )
    }
}

pub struct OfflineCheckerChip<const WORD_SIZE: usize, F, Operation: OfflineCheckerOperation<F>> {
    _marker: PhantomData<(F, Operation)>,
    pub air: OfflineChecker<WORD_SIZE>,
}

impl<const WORD_SIZE: usize, F, Operation: OfflineCheckerOperation<F>>
    OfflineCheckerChip<WORD_SIZE, F, Operation>
{
    pub fn new(air: OfflineChecker<WORD_SIZE>) -> Self {
        Self {
            _marker: Default::default(),
            air,
        }
    }
}
