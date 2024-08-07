use std::iter;

use afs_primitives::is_less_than_tuple::columns::IsLessThanTupleAuxCols;
use derive_new::new;

use super::air::AuditAir;
use crate::memory::manager::MemoryReadWriteOpCols;

#[allow(clippy::too_many_arguments)]
#[derive(new)]
pub struct AuditCols<const WORD_SIZE: usize, T> {
    pub op_cols: MemoryReadWriteOpCols<WORD_SIZE, T>,

    pub is_extra: T,
    pub addr_lt: T,
    pub addr_lt_aux: IsLessThanTupleAuxCols<T>,
}

impl<const WORD_SIZE: usize, T: Clone> AuditCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T], audit_air: &AuditAir<WORD_SIZE>) -> Self {
        Self {
            op_cols: MemoryReadWriteOpCols::from_slice(&slc[..4 + 2 * WORD_SIZE]),
            is_extra: slc[4 + 2 * WORD_SIZE].clone(),
            addr_lt: slc[5 + 2 * WORD_SIZE].clone(),
            addr_lt_aux: IsLessThanTupleAuxCols::from_slice(
                &slc[6 + 2 * WORD_SIZE..],
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

    pub fn width(audit_air: &AuditAir<WORD_SIZE>) -> usize {
        6 + 2 * WORD_SIZE + IsLessThanTupleAuxCols::<T>::width(&audit_air.addr_lt_air)
    }
}
