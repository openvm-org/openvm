use std::mem::size_of;

use afs_derive::AlignedBorrow;
use derive_new::new;

use crate::{
    arch::columns::ExecutionState,
    memory::{
        offline_checker::{MemoryHeapReadAuxCols, MemoryHeapWriteAuxCols},
        MemoryHeapDataIoCols,
    },
};

// Note: repr(C) is needed as we assume the memory layout when using aligned_borrow.
#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct ModularAddSubCols<T: Clone, const NUM_LIMBS: usize> {
    pub io: ModularAddSubIoCols<T, NUM_LIMBS>,
    pub aux: ModularAddSubAuxCols<T, NUM_LIMBS>,
}

impl<T: Clone, const NUM_LIMBS: usize> ModularAddSubCols<T, NUM_LIMBS> {
    pub const fn width() -> usize {
        ModularAddSubIoCols::<T, NUM_LIMBS>::width() + ModularAddSubAuxCols::<T, NUM_LIMBS>::width()
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct ModularAddSubIoCols<T: Clone, const NUM_LIMBS: usize> {
    pub from_state: ExecutionState<T>,
    pub x: MemoryHeapDataIoCols<T, NUM_LIMBS>,
    pub y: MemoryHeapDataIoCols<T, NUM_LIMBS>,
    pub z: MemoryHeapDataIoCols<T, NUM_LIMBS>,
}

impl<T: Clone, const NUM_LIMBS: usize> ModularAddSubIoCols<T, NUM_LIMBS> {
    pub const fn width() -> usize {
        size_of::<ModularAddSubIoCols<u8, NUM_LIMBS>>()
    }
}

// Note: to save a column we assume that is_sub is represented as is_valid - is_add
//       it is checked in the air
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug, new)]
pub struct ModularAddSubAuxCols<T: Clone, const NUM_LIMBS: usize> {
    // 0 for padding rows.
    pub is_valid: T,
    pub read_x_aux_cols: MemoryHeapReadAuxCols<T, NUM_LIMBS>,
    pub read_y_aux_cols: MemoryHeapReadAuxCols<T, NUM_LIMBS>,
    pub write_z_aux_cols: MemoryHeapWriteAuxCols<T, NUM_LIMBS>,

    pub carries: [T; NUM_LIMBS],
    pub q: T,
    pub is_add: T,
}

impl<T: Clone, const NUM_LIMBS: usize> ModularAddSubAuxCols<T, NUM_LIMBS> {
    pub const fn width() -> usize {
        size_of::<ModularAddSubAuxCols<u8, NUM_LIMBS>>()
    }
}
