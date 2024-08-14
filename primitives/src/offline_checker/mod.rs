use std::marker::PhantomData;

use self::columns::OfflineCheckerCols;
use crate::{is_equal_vec::IsEqualVecAir, is_less_than_tuple::IsLessThanTupleAir};

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

#[derive(Clone, Debug)]
pub struct OfflineChecker {
    pub idx_clk_limb_bits: Vec<usize>,
    pub decomp: usize,
    pub idx_len: usize,
    pub data_len: usize,
    pub range_bus: usize,
    pub ops_bus: usize,

    pub is_equal_idx_air: IsEqualVecAir,
    pub lt_tuple_air: IsLessThanTupleAir,
}

impl OfflineChecker {
    pub fn new(
        idx_clk_limb_bits: Vec<usize>,
        decomp: usize,
        idx_len: usize,
        data_len: usize,
        range_bus: usize,
        ops_bus: usize,
    ) -> Self {
        Self {
            idx_clk_limb_bits: idx_clk_limb_bits.clone(),
            decomp,
            idx_len,
            data_len,
            range_bus,
            ops_bus,
            is_equal_idx_air: IsEqualVecAir::new(idx_len),
            lt_tuple_air: IsLessThanTupleAir::new(range_bus, idx_clk_limb_bits, decomp),
        }
    }

    pub fn idx_data_width(&self) -> usize {
        self.idx_len + self.data_len
    }

    pub fn air_width(&self) -> usize {
        OfflineCheckerCols::<usize>::width(self)
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
