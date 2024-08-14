use std::iter;

use afs_primitives::is_less_than_tuple::columns::IsLessThanTupleAuxCols;
use derive_new::new;

use super::air::MemoryAuditAir;
use crate::memory::manager::access_cell::AccessCell;

#[allow(clippy::too_many_arguments)]
#[derive(new)]
pub struct AuditCols<const WORD_SIZE: usize, T> {
    pub addr_space: T,
    pub pointer: T,

    pub initial_cell: AccessCell<WORD_SIZE, T>,
    pub final_cell: AccessCell<WORD_SIZE, T>,

    pub is_extra: T,
    pub addr_lt: T,
    pub addr_lt_aux: IsLessThanTupleAuxCols<T>,
}

impl<const WORD_SIZE: usize, T: Clone> AuditCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T], audit_air: &MemoryAuditAir<WORD_SIZE>) -> Self {
        let ac_width = AccessCell::<WORD_SIZE, T>::width();

        Self {
            addr_space: slc[0].clone(),
            pointer: slc[1].clone(),
            initial_cell: AccessCell::from_slice(&slc[2..2 + ac_width]),
            final_cell: AccessCell::from_slice(&slc[2 + ac_width..2 + 2 * ac_width]),
            is_extra: slc[2 + 2 * ac_width].clone(),
            addr_lt: slc[3 + 2 * ac_width].clone(),
            addr_lt_aux: IsLessThanTupleAuxCols::from_slice(
                &slc[4 + 2 * ac_width..],
                &audit_air.addr_lt_air,
            ),
        }
    }

    pub fn flatten(self) -> Vec<T> {
        iter::once(self.addr_space)
            .chain(iter::once(self.pointer))
            .chain(self.initial_cell.flatten())
            .chain(self.final_cell.flatten())
            .chain(iter::once(self.is_extra))
            .chain(iter::once(self.addr_lt))
            .chain(self.addr_lt_aux.flatten())
            .collect()
    }

    pub fn width(audit_air: &MemoryAuditAir<WORD_SIZE>) -> usize {
        4 + 2 * AccessCell::<WORD_SIZE, T>::width()
            + IsLessThanTupleAuxCols::<T>::width(&audit_air.addr_lt_air)
    }
}
