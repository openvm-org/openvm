use openvm_circuit::system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols};
use openvm_circuit_primitives_derive::AlignedBorrow;

use p3_keccak_air::KeccakCols as KeccakPermCols;

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
pub struct KeccakfWrapperCols<T> {
    // SAFETY: inner has to be the first column defined
    pub inner: KeccakPermCols<T>,
    pub request_id: T,
    pub is_enabled: T, 
}

pub const NUM_KECCAKF_WRAPPER_COLS: usize = size_of::<KeccakfWrapperCols<u8>>();