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
pub struct ModularArithmeticCols<T: Clone, const NUM_LIMBS: usize> {
    pub io: ModularArithmeticIoCols<T, NUM_LIMBS>,
    pub aux: ModularArithmeticAuxCols<T, NUM_LIMBS>,
}

impl<T: Clone, const NUM_LIMBS: usize> ModularArithmeticCols<T, NUM_LIMBS> {
    pub const fn width() -> usize {
        ModularArithmeticIoCols::<T, NUM_LIMBS>::width()
            + ModularArithmeticAuxCols::<T, NUM_LIMBS>::width()
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct ModularArithmeticIoCols<T: Clone, const NUM_LIMBS: usize> {
    pub from_state: ExecutionState<T>,
    pub x: MemoryHeapDataIoCols<T, NUM_LIMBS>,
    pub y: MemoryHeapDataIoCols<T, NUM_LIMBS>,
    pub z: MemoryHeapDataIoCols<T, NUM_LIMBS>,
}

impl<T: Clone, const NUM_LIMBS: usize> ModularArithmeticIoCols<T, NUM_LIMBS> {
    pub const fn width() -> usize {
        size_of::<ModularArithmeticIoCols<u8, NUM_LIMBS>>()
    }
}

// Note: to save a column we assume that is_sub is represented as is_valid - is_add
//       it is checked in the air
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug, new)]
pub struct ModularArithmeticAuxCols<T: Clone, const NUM_LIMBS: usize> {
    // 0 for padding rows.
    pub is_valid: T,
    pub read_x_aux_cols: MemoryHeapReadAuxCols<T, NUM_LIMBS>,
    pub read_y_aux_cols: MemoryHeapReadAuxCols<T, NUM_LIMBS>,
    pub write_z_aux_cols: MemoryHeapWriteAuxCols<T, NUM_LIMBS>,

    pub carries: [T; NUM_LIMBS],
    pub q: T,
    pub is_add: T,
}

impl<T: Clone, const NUM_LIMBS: usize> ModularArithmeticAuxCols<T, NUM_LIMBS> {
    pub const fn width() -> usize {
        size_of::<ModularArithmeticAuxCols<u8, NUM_LIMBS>>()
    }
}
