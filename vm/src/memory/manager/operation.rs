use std::iter;

use derive_new::new;

use super::access_cell::AccessCell;

#[derive(Clone, Debug, PartialEq, Eq, new)]
pub struct MemoryOperation<const WORD_SIZE: usize, T> {
    pub addr_space: T,
    pub pointer: T,
    pub op_type: T,
    pub cell: AccessCell<WORD_SIZE, T>,
    pub enabled: T,
}

impl<const WORD_SIZE: usize, T: Clone> MemoryOperation<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            addr_space: slc[0].clone(),
            pointer: slc[1].clone(),
            op_type: slc[2].clone(),
            cell: AccessCell::from_slice(&slc[3..WORD_SIZE + 4]),
            enabled: slc[WORD_SIZE + 4].clone(),
        }
    }
}

impl<const WORD_SIZE: usize, T> MemoryOperation<WORD_SIZE, T> {
    pub fn flatten(self) -> Vec<T> {
        iter::once(self.addr_space)
            .chain(iter::once(self.pointer))
            .chain(iter::once(self.op_type))
            .chain(self.cell.flatten())
            .chain(iter::once(self.enabled))
            .collect()
    }

    pub fn width() -> usize {
        4 + AccessCell::<WORD_SIZE, T>::width()
    }
}
