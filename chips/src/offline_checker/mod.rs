use std::marker::PhantomData;

use crate::is_equal_vec::columns::IsEqualVecAuxCols;
use crate::is_less_than_tuple::columns::IsLessThanTupleAuxCols;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub trait GeneralOfflineCheckerOperation<F> {
    fn get_timestamp(&self) -> usize;
    fn get_idx(&self) -> Vec<F>;
    fn get_data(&self) -> Vec<F>;
    fn get_op_type(&self) -> u8;
}

#[derive(Clone)]
pub struct GeneralOfflineChecker {
    pub idx_clk_limb_bits: Vec<usize>,
    pub decomp: usize,
    pub idx_len: usize,
    pub data_len: usize,
    pub range_bus: usize,
    pub ops_bus: usize,
}

impl GeneralOfflineChecker {
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

pub struct GeneralOfflineCheckerChip<F, Operation: GeneralOfflineCheckerOperation<F>> {
    _marker: PhantomData<(F, Operation)>,
    pub air: GeneralOfflineChecker,
}

impl<F, Operation: GeneralOfflineCheckerOperation<F>> GeneralOfflineCheckerChip<F, Operation> {
    pub fn new(air: GeneralOfflineChecker) -> Self {
        Self {
            _marker: Default::default(),
            air,
        }
    }
}
