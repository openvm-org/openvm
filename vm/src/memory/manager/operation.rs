use std::iter;

use derive_new::new;
use p3_air::AirBuilder;

use super::access_cell::AccessCell;

#[derive(Clone, Debug, PartialEq, Eq, Default, new)]
pub struct MemoryOperation<const WORD_SIZE: usize, T> {
    pub addr_space: T,
    pub pointer: T,
    // TODO[jpw]: remove this
    pub op_type: T,
    pub cell: AccessCell<WORD_SIZE, T>,
    pub enabled: T,
}

impl<const WORD_SIZE: usize, T: Clone> MemoryOperation<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let ac_width = AccessCell::<WORD_SIZE, T>::width();

        Self {
            addr_space: slc[0].clone(),
            pointer: slc[1].clone(),
            op_type: slc[2].clone(),
            cell: AccessCell::from_slice(&slc[3..3 + ac_width]),
            enabled: slc[3 + ac_width].clone(),
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

impl<const WORD_SIZE: usize, T> MemoryOperation<WORD_SIZE, T> {
    pub fn into_expr<AB: AirBuilder>(self) -> MemoryOperation<WORD_SIZE, AB::Expr>
    where
        T: Into<AB::Expr>,
    {
        MemoryOperation::new(
            self.addr_space.into(),
            self.pointer.into(),
            self.op_type.into(),
            self.cell.into_expr::<AB>(),
            self.enabled.into(),
        )
    }
}
