use std::mem::size_of;

use afs_derive::AlignedBorrow;

use crate::{
    arch::columns::ExecutionState,
    memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct CastFCols<T> {
    pub io: CastFIoCols<T>,
    pub aux: CastFAuxCols<T>,
}

impl<T> CastFCols<T> {
    pub const fn width() -> usize {
        CastFIoCols::<T>::width() + CastFAuxCols::<T>::width()
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug, Default)]
pub struct CastFIoCols<T> {
    pub from_state: ExecutionState<T>,
    pub op_a: T,
    pub op_b: T,
    pub d: T,
    pub e: T,
    pub x: [T; 4],
}

impl<T> CastFIoCols<T> {
    pub const fn width() -> usize {
        size_of::<CastFIoCols<u8>>()
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct CastFAuxCols<T> {
    pub is_valid: T,
    pub write_x_aux_cols: MemoryWriteAuxCols<T, 4>,
    pub read_y_aux_cols: MemoryReadAuxCols<T, 1>,
}

impl<T> CastFAuxCols<T> {
    pub const fn width() -> usize {
        size_of::<CastFAuxCols<u8>>()
    }
}
