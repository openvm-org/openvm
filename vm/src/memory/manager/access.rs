use std::{array::from_fn, iter};

use derive_new::new;

// TODO[osama]: to be deleted
// TODO[osama]: to be renamed to MemoryAccess
#[derive(Clone, Debug, PartialEq, Eq, new)]
pub struct NewMemoryAccessCols<const WORD_SIZE: usize, T> {
    pub addr_space: T,
    pub pointer: T,
    pub op_type: T,

    pub data_read: [T; WORD_SIZE],
    pub clk_read: T,
    pub data_write: [T; WORD_SIZE],
    pub clk_write: T,
}

impl<const WORD_SIZE: usize, T: Default> Default for NewMemoryAccessCols<WORD_SIZE, T> {
    fn default() -> Self {
        Self {
            addr_space: T::default(),
            pointer: T::default(),
            op_type: T::default(),
            data_read: from_fn(|_| T::default()),
            clk_read: T::default(),
            data_write: from_fn(|_| T::default()),
            clk_write: T::default(),
        }
    }
}

impl<const WORD_SIZE: usize, T: Clone> NewMemoryAccessCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            addr_space: slc[0].clone(),
            pointer: slc[1].clone(),
            op_type: slc[2].clone(),
            data_read: from_fn(|i| slc[3 + i].clone()),
            clk_read: slc[3 + WORD_SIZE].clone(),
            data_write: from_fn(|i| slc[4 + WORD_SIZE + i].clone()),
            clk_write: slc[4 + 2 * WORD_SIZE].clone(),
        }
    }
}

impl<const WORD_SIZE: usize, T> NewMemoryAccessCols<WORD_SIZE, T> {
    pub fn flatten(self) -> Vec<T> {
        vec![self.addr_space, self.pointer, self.op_type]
            .into_iter()
            .chain(self.data_read)
            .chain(iter::once(self.clk_read))
            .chain(self.data_write)
            .chain(iter::once(self.clk_write))
            .collect()
    }

    pub fn width() -> usize {
        5 + 2 * WORD_SIZE
    }
}
