use afs_primitives::is_less_than::columns::IsLessThanAuxCols;

use super::air::NewMemoryOfflineChecker;
use crate::memory::manager::MemoryReadWriteOpCols;

pub struct MemoryOfflineCheckerCols<const WORD_SIZE: usize, T> {
    pub op_cols: MemoryReadWriteOpCols<WORD_SIZE, T>,
    pub is_extra: T,

    pub clk_lt: T,
    pub clk_lt_aux: IsLessThanAuxCols<T>,
}

impl<const WORD_SIZE: usize, T: Clone> MemoryOfflineCheckerCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            op_cols: MemoryReadWriteOpCols::from_slice(&slc[0..4 + 2 * WORD_SIZE]),
            is_extra: slc[4 + 2 * WORD_SIZE].clone(),
            clk_lt: slc[5 + 2 * WORD_SIZE].clone(),
            clk_lt_aux: IsLessThanAuxCols::from_slice(&slc[6 + 2 * WORD_SIZE..]),
        }
    }
}

impl<const WORD_SIZE: usize, T> MemoryOfflineCheckerCols<WORD_SIZE, T> {
    pub fn width(oc: &NewMemoryOfflineChecker<WORD_SIZE>) -> usize {
        MemoryReadWriteOpCols::<WORD_SIZE, usize>::width()
            + 2
            + IsLessThanAuxCols::<T>::width(&oc.clk_lt_air)
    }
}
