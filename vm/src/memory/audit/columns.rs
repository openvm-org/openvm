use std::{array::from_fn, iter};

use afs_primitives::is_less_than_tuple::columns::IsLessThanTupleAuxCols;
use derive_new::new;
use p3_air::AirBuilder;

use super::air::MemoryAuditAir;
use crate::memory::manager::access_cell::AccessCell;

#[allow(clippy::too_many_arguments)]
#[derive(new)]
pub struct AuditCols<const WORD_SIZE: usize, T> {
    pub addr_space: T,
    pub pointer: T,

    pub initial_data: [T; WORD_SIZE],
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
            initial_data: from_fn(|i| slc[2 + i].clone()),
            final_cell: AccessCell::from_slice(&slc[2 + WORD_SIZE..2 + WORD_SIZE + ac_width]),
            is_extra: slc[2 + WORD_SIZE + ac_width].clone(),
            addr_lt: slc[3 + WORD_SIZE + ac_width].clone(),
            addr_lt_aux: IsLessThanTupleAuxCols::from_slice(
                &slc[4 + WORD_SIZE + ac_width..],
                &audit_air.addr_lt_air,
            ),
        }
    }

    pub fn flatten(self) -> Vec<T> {
        iter::once(self.addr_space)
            .chain(iter::once(self.pointer))
            .chain(self.initial_data.iter().cloned())
            .chain(self.final_cell.flatten())
            .chain(iter::once(self.is_extra))
            .chain(iter::once(self.addr_lt))
            .chain(self.addr_lt_aux.flatten())
            .collect()
    }

    pub fn width(audit_air: &MemoryAuditAir<WORD_SIZE>) -> usize {
        4 + WORD_SIZE
            + AccessCell::<WORD_SIZE, T>::width()
            + IsLessThanTupleAuxCols::<T>::width(&audit_air.addr_lt_air)
    }
}

impl<const WORD_SIZE: usize, T> AuditCols<WORD_SIZE, T> {
    pub fn into_expr<AB: AirBuilder>(self) -> AuditCols<WORD_SIZE, AB::Expr>
    where
        T: Into<AB::Expr>,
    {
        AuditCols::new(
            self.addr_space.into(),
            self.pointer.into(),
            self.initial_data.map(Into::into),
            self.final_cell.into_expr::<AB>(),
            self.is_extra.into(),
            self.addr_lt.into(),
            self.addr_lt_aux.into_expr::<AB>(),
        )
    }
}
