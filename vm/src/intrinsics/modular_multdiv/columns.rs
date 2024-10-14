use afs_derive::AlignedBorrow;
use derive_new::new;

use crate::{
    arch::ExecutionState,
    system::memory::{
        offline_checker::{MemoryHeapReadAuxCols, MemoryHeapWriteAuxCols},
        MemoryHeapDataIoCols,
    },
};

// Note: repr(C) is needed as we assume the memory layout when using aligned_borrow.
#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct ModularMultDivCols<T: Clone, const CARRY_LIMBS: usize, const NUM_LIMBS: usize> {
    pub io: ModularMultDivIoCols<T, NUM_LIMBS>,
    pub aux: ModularMultDivAuxCols<T, CARRY_LIMBS, NUM_LIMBS>,
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct ModularMultDivIoCols<T: Clone, const NUM_LIMBS: usize> {
    pub from_state: ExecutionState<T>,
    pub x: MemoryHeapDataIoCols<T, NUM_LIMBS>,
    pub y: MemoryHeapDataIoCols<T, NUM_LIMBS>,
    pub z: MemoryHeapDataIoCols<T, NUM_LIMBS>,
}

// Note: to save a column we assume that is_div is represented as is_valid - is_mult
//       it is checked in the air
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug, new)]
pub struct ModularMultDivAuxCols<T: Clone, const CARRY_LIMBS: usize, const NUM_LIMBS: usize> {
    // 0 for padding rows.
    pub is_valid: T,
    pub read_x_aux_cols: MemoryHeapReadAuxCols<T, NUM_LIMBS>,
    pub read_y_aux_cols: MemoryHeapReadAuxCols<T, NUM_LIMBS>,
    pub write_z_aux_cols: MemoryHeapWriteAuxCols<T, NUM_LIMBS>,

    pub carries: [T; CARRY_LIMBS],
    pub q: [T; NUM_LIMBS],
    pub is_mult: T,
}
