use std::marker::PhantomData;

use derive_new::new;

use crate::is_equal_vec::columns::IsEqualVecAuxCols;
use crate::is_less_than_tuple::columns::IsLessThanTupleAuxCols;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub trait OfflineCheckerOperation<F> {
    fn get_timestamp(&self) -> usize;
    fn get_idx(&self) -> Vec<F>;
    fn get_data(&self) -> Vec<F>;
    fn get_op_type(&self) -> u8;
}

#[derive(Clone, new)]
pub struct OfflineChecker {
    pub idx_clk_limb_bits: Vec<usize>,
    pub decomp: usize,
    pub idx_len: usize,
    pub data_len: usize,
    pub range_bus: usize,
    pub ops_bus: usize,
}

impl OfflineChecker {
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
            )
    }
}

pub struct OfflineCheckerChip<F, Operation: OfflineCheckerOperation<F>> {
    _marker: PhantomData<(F, Operation)>,
    pub air: OfflineChecker,
}

impl<F, Operation: OfflineCheckerOperation<F>> OfflineCheckerChip<F, Operation> {
    pub fn new(air: OfflineChecker) -> Self {
        Self {
            _marker: Default::default(),
            air,
        }
    }
}
