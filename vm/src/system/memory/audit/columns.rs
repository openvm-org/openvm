use std::iter;

use afs_primitives::is_less_than_tuple::columns::IsLessThanTupleAuxCols;
use derive_new::new;
use p3_air::AirBuilder;

use super::air::MemoryAuditAir;

#[allow(clippy::too_many_arguments)]
#[derive(new)]
pub struct AuditCols<T> {
    pub addr_space: T,
    pub pointer: T,

    pub initial_data: T,
    pub final_data: T,
    pub final_timestamp: T,

    pub is_extra: T,
    pub addr_lt: T,
    pub addr_lt_aux: IsLessThanTupleAuxCols<T>,
}

impl<T: Clone> AuditCols<T> {
    pub fn from_slice(slc: &[T], audit_air: &MemoryAuditAir) -> Self {
        Self {
            addr_space: slc[0].clone(),
            pointer: slc[1].clone(),
            initial_data: slc[2].clone(),
            final_data: slc[3].clone(),
            final_timestamp: slc[4].clone(),
            is_extra: slc[5].clone(),
            addr_lt: slc[6].clone(),
            addr_lt_aux: IsLessThanTupleAuxCols::from_slice(&slc[7..], &audit_air.addr_lt_air),
        }
    }

    pub fn flatten(self) -> Vec<T> {
        iter::once(self.addr_space)
            .chain(iter::once(self.pointer))
            .chain(iter::once(self.initial_data))
            .chain(iter::once(self.final_data))
            .chain(iter::once(self.final_timestamp))
            .chain(iter::once(self.is_extra))
            .chain(iter::once(self.addr_lt))
            .chain(self.addr_lt_aux.flatten())
            .collect()
    }

    pub fn width(audit_air: &MemoryAuditAir) -> usize {
        7 + IsLessThanTupleAuxCols::<T>::width(&audit_air.addr_lt_air)
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
            self.final_data.into(),
            self.final_timestamp.into(),
            self.is_extra.into(),
            self.addr_lt.into(),
            self.addr_lt_aux.into_expr::<AB>(),
        )
    }
}
