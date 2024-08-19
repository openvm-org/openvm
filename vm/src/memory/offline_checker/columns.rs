use std::iter;

use afs_primitives::is_less_than::{columns::IsLessThanAuxCols, IsLessThanAir};
use derive_new::new;

use super::air::NewMemoryOfflineChecker;
use crate::memory::manager::{access_cell::AccessCell, operation::MemoryOperation};

// TODO[osama]: to be renamed to MemoryAccess
#[derive(new, Default)]
pub struct NewMemoryAccess<const WORD_SIZE: usize, T> {
    pub op: MemoryOperation<WORD_SIZE, T>,
    pub old_cell: AccessCell<WORD_SIZE, T>,
}

#[derive(Clone, Debug, PartialEq, Eq, Default, new)]
pub struct MemoryOfflineCheckerCols<const WORD_SIZE: usize, T> {
    pub io: MemoryOperation<WORD_SIZE, T>,
    pub aux: MemoryOfflineCheckerAuxCols<WORD_SIZE, T>,
}

#[derive(Clone, Debug, PartialEq, Eq, Default, new)]
pub struct MemoryOfflineCheckerAuxCols<const WORD_SIZE: usize, T> {
    // TODO[jpw]: Remove this; read does not need old_data
    pub old_cell: AccessCell<WORD_SIZE, T>,
    pub is_immediate: T,
    pub is_zero_aux: T,
    // TODO[jpw]: IsLessThan should be optimized to AssertLessThan
    pub clk_lt: T,
    pub clk_lt_aux: IsLessThanAuxCols<T>,
}

// Straightforward implementations for from_slice, flatten, width functions for the above structs below

impl<const WORD_SIZE: usize, T: Clone> MemoryOfflineCheckerCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let op_width = MemoryOperation::<WORD_SIZE, T>::width();
        Self {
            io: MemoryOperation::<WORD_SIZE, T>::from_slice(&slc[..op_width]),
            aux: MemoryOfflineCheckerAuxCols::<WORD_SIZE, T>::from_slice(&slc[op_width..]),
        }
    }
}

impl<const WORD_SIZE: usize, T> MemoryOfflineCheckerCols<WORD_SIZE, T> {
    pub fn flatten(self) -> Vec<T> {
        self.io
            .flatten()
            .into_iter()
            .chain(self.aux.flatten())
            .collect()
    }

    pub fn width(oc: &NewMemoryOfflineChecker<WORD_SIZE>) -> usize {
        MemoryOperation::<WORD_SIZE, T>::width()
            + MemoryOfflineCheckerAuxCols::<WORD_SIZE, T>::width(oc)
    }
}

impl<const WORD_SIZE: usize, T: Clone> MemoryOfflineCheckerAuxCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            old_cell: AccessCell::from_slice(&slc[..WORD_SIZE + 1]),
            is_immediate: slc[WORD_SIZE + 1].clone(),
            is_zero_aux: slc[WORD_SIZE + 2].clone(),
            clk_lt: slc[WORD_SIZE + 3].clone(),
            clk_lt_aux: IsLessThanAuxCols::from_slice(&slc[WORD_SIZE + 4..]),
        }
    }
}

impl<const WORD_SIZE: usize, T> MemoryOfflineCheckerAuxCols<WORD_SIZE, T> {
    pub fn flatten(self) -> Vec<T> {
        self.old_cell
            .flatten()
            .into_iter()
            .chain(iter::once(self.is_immediate))
            .chain(iter::once(self.is_zero_aux))
            .chain(iter::once(self.clk_lt))
            .chain(self.clk_lt_aux.flatten())
            .collect()
    }

    pub fn from_iter<I: Iterator<Item = T>>(iter: &mut I, lt_air: &IsLessThanAir) -> Self {
        Self {
            old_cell: AccessCell::from_iter(iter),
            is_immediate: iter.next().unwrap(),
            is_zero_aux: iter.next().unwrap(),
            clk_lt: iter.next().unwrap(),
            clk_lt_aux: IsLessThanAuxCols::from_iter(iter, lt_air),
        }
    }

    pub fn width(oc: &NewMemoryOfflineChecker<WORD_SIZE>) -> usize {
        AccessCell::<WORD_SIZE, T>::width() + 3 + IsLessThanAuxCols::<T>::width(&oc.clk_lt_air)
    }
}

// impl<const WORD_SIZE: usize, T: Copy> MemoryOfflineCheckerCols<WORD_SIZE, T> {
//     // TODO[osama]: make sure this is used in all relevant parts
//     pub fn data(&self) -> [T; WORD_SIZE] {
//         self.io.op_cols.data_write
//     }

//     pub fn addr_space(&self) -> T {
//         self.io.op_cols.addr_space
//     }

//     pub fn pointer(&self) -> T {
//         self.io.op_cols.pointer
//     }
// }
