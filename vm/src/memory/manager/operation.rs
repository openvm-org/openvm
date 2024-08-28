use std::iter;

use derive_new::new;
use p3_air::AirBuilder;

use super::access_cell::AccessCell;

pub type MemoryReadCols<const WORD_SIZE: usize, T> = MemoryAccessCols<WORD_SIZE, T>;
pub type MemoryWriteCols<const WORD_SIZE: usize, T> = MemoryAccessCols<WORD_SIZE, T>;

#[derive(Clone, Debug, PartialEq, Eq, Default, new)]
pub struct MemoryAccessCols<const WORD_SIZE: usize, T> {
    pub addr_space: T,
    pub pointer: T,
    pub cell: AccessCell<WORD_SIZE, T>,
    pub enabled: T,
}

impl<const WORD_SIZE: usize, T: Clone> MemoryAccessCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let ac_width = AccessCell::<WORD_SIZE, T>::width();

        Self {
            addr_space: slc[0].clone(),
            pointer: slc[1].clone(),
            cell: AccessCell::from_slice(&slc[2..2 + ac_width]),
            enabled: slc[2 + ac_width].clone(),
        }
    }
}

impl<const WORD_SIZE: usize, T> MemoryAccessCols<WORD_SIZE, T> {
    pub fn flatten(self) -> Vec<T> {
        iter::once(self.addr_space)
            .chain(iter::once(self.pointer))
            .chain(self.cell.flatten())
            .chain(iter::once(self.enabled))
            .collect()
    }

    pub fn width() -> usize {
        3 + AccessCell::<WORD_SIZE, T>::width()
    }
}

impl<const WORD_SIZE: usize, T> MemoryAccessCols<WORD_SIZE, T> {
    pub fn into_expr<AB: AirBuilder>(self) -> MemoryAccessCols<WORD_SIZE, AB::Expr>
    where
        T: Into<AB::Expr>,
    {
        MemoryAccessCols::new(
            self.addr_space.into(),
            self.pointer.into(),
            self.cell.into_expr::<AB>(),
            self.enabled.into(),
        )
    }
}
