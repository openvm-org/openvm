use std::iter;

use afs_primitives::is_less_than_tuple::columns::IsLessThanTupleAuxCols;
use derive_new::new;

use super::air::MemoryAuditAir;
use crate::memory::manager::NewMemoryAccessCols;

#[allow(clippy::too_many_arguments)]
#[derive(new)]
pub struct AuditCols<const WORD_SIZE: usize, T> {
    pub op_cols: NewMemoryAccessCols<WORD_SIZE, T>,

    pub is_extra: T,
    pub addr_lt: T,
    pub addr_lt_aux: IsLessThanTupleAuxCols<T>,
}

impl<const WORD_SIZE: usize, T: Clone> AuditCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T], audit_air: &MemoryAuditAir<WORD_SIZE>) -> Self {
        let op_cols_width = NewMemoryAccessCols::<WORD_SIZE, T>::width();

        Self {
            op_cols: NewMemoryAccessCols::from_slice(&slc[..op_cols_width]),
            is_extra: slc[op_cols_width].clone(),
            addr_lt: slc[1 + op_cols_width].clone(),
            addr_lt_aux: IsLessThanTupleAuxCols::from_slice(
                &slc[2 + op_cols_width..],
                &audit_air.addr_lt_air,
            ),
        }
    }

    pub fn flatten(self) -> Vec<T> {
        self.op_cols
            .flatten()
            .into_iter()
            .chain(iter::once(self.is_extra))
            .chain(iter::once(self.addr_lt))
            .chain(self.addr_lt_aux.flatten())
            .collect()
    }

    pub fn width(audit_air: &MemoryAuditAir<WORD_SIZE>) -> usize {
        2 + NewMemoryAccessCols::<WORD_SIZE, T>::width()
            + IsLessThanTupleAuxCols::<T>::width(&audit_air.addr_lt_air)
    }
}
