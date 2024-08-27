use std::iter;

use afs_primitives::is_less_than_tuple::columns::IsLessThanTupleAuxCols;
use derive_new::new;
use p3_air::AirBuilder;

use super::air::MemoryAuditAir;
use crate::memory::manager::access_cell::AccessCell;

#[allow(clippy::too_many_arguments)]
#[derive(new)]
pub struct AuditCols<T> {
    pub addr_space: T,
    pub pointer: T,

    pub initial_data: T,
    pub final_cell: AccessCell<1, T>,

    pub is_extra: T,
    pub addr_lt: T,
    pub addr_lt_aux: IsLessThanTupleAuxCols<T>,
}

impl<T: Clone> AuditCols<T> {
    pub fn from_slice(slc: &[T], audit_air: &MemoryAuditAir) -> Self {
        let ac_width = AccessCell::<1, T>::width();

        Self {
            addr_space: slc[0].clone(),
            pointer: slc[1].clone(),
            initial_data: slc[2].clone(),
            final_cell: AccessCell::from_slice(&slc[3..3 + ac_width]),
            is_extra: slc[3 + ac_width].clone(),
            addr_lt: slc[4 + ac_width].clone(),
            addr_lt_aux: IsLessThanTupleAuxCols::from_slice(
                &slc[5 + ac_width..],
                &audit_air.addr_lt_air,
            ),
        }
    }

    pub fn flatten(self) -> Vec<T> {
        iter::once(self.addr_space)
            .chain(iter::once(self.pointer))
            .chain(iter::once(self.initial_data))
            .chain(self.final_cell.flatten())
            .chain(iter::once(self.is_extra))
            .chain(iter::once(self.addr_lt))
            .chain(self.addr_lt_aux.flatten())
            .collect()
    }

    pub fn width(audit_air: &MemoryAuditAir) -> usize {
        5 + AccessCell::<1, T>::width() + IsLessThanTupleAuxCols::<T>::width(&audit_air.addr_lt_air)
    }
}

impl<T> AuditCols<T> {
    pub fn into_expr<AB: AirBuilder>(self) -> AuditCols<AB::Expr>
    where
        T: Into<AB::Expr>,
    {
        AuditCols::new(
            self.addr_space.into(),
            self.pointer.into(),
            self.initial_data.into(),
            self.final_cell.into_expr::<AB>(),
            self.is_extra.into(),
            self.addr_lt.into(),
            self.addr_lt_aux.into_expr::<AB>(),
        )
    }
}
