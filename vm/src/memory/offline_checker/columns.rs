use std::iter;

use afs_primitives::is_less_than::columns::IsLessThanAuxCols;
use derive_new::new;

use super::air::NewMemoryOfflineChecker;
use crate::memory::manager::access::NewMemoryAccessCols;

#[derive(Clone, Debug, PartialEq, Eq, new)]
pub struct MemoryOfflineCheckerCols<const WORD_SIZE: usize, T> {
    pub op_cols: NewMemoryAccessCols<WORD_SIZE, T>,
    pub is_immediate: T,
    pub clk_lt: T,
    pub enabled: T,

    pub is_zero_aux: T,
    pub clk_lt_aux: IsLessThanAuxCols<T>,
}

impl<const WORD_SIZE: usize, T: Clone> MemoryOfflineCheckerCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let op_cols_width = NewMemoryAccessCols::<WORD_SIZE, T>::width();

        Self {
            op_cols: NewMemoryAccessCols::from_slice(&slc[0..op_cols_width]),
            is_immediate: slc[op_cols_width].clone(),
            clk_lt: slc[1 + op_cols_width].clone(),
            enabled: slc[2 + op_cols_width].clone(),
            is_zero_aux: slc[3 + op_cols_width].clone(),
            clk_lt_aux: IsLessThanAuxCols::from_slice(&slc[4 + op_cols_width..]),
        }
    }

    pub fn flatten(self) -> Vec<T> {
        self.op_cols
            .flatten()
            .into_iter()
            .chain(iter::once(self.is_immediate))
            .chain(iter::once(self.clk_lt))
            .chain(iter::once(self.enabled))
            .chain(iter::once(self.is_zero_aux))
            .chain(self.clk_lt_aux.flatten())
            .collect()
    }
}

impl<const WORD_SIZE: usize, T: Copy> MemoryOfflineCheckerCols<WORD_SIZE, T> {
    // TODO[osama]: make sure this is used in all relevant parts
    pub fn data(&self) -> [T; WORD_SIZE] {
        self.op_cols.data_write
    }

    pub fn addr_space(&self) -> T {
        self.op_cols.addr_space
    }

    pub fn pointer(&self) -> T {
        self.op_cols.pointer
    }
}

impl<const WORD_SIZE: usize, T> MemoryOfflineCheckerCols<WORD_SIZE, T> {
    pub fn width(oc: &NewMemoryOfflineChecker<WORD_SIZE>) -> usize {
        NewMemoryAccessCols::<WORD_SIZE, usize>::width()
            + 4
            + IsLessThanAuxCols::<T>::width(&oc.clk_lt_air)
    }
}
