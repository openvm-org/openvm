use core::mem::size_of;

use afs_derive::AlignedBorrow;

#[derive(Default, AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct ByteOperationLookupCols<T> {
    pub mult_add: T,
    pub mult_xor: T,
}

#[derive(Default, AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct ByteOperationLookupPreprocessedCols<T> {
    pub x: T,
    pub y: T,
    pub z_add: T,
    pub z_xor: T,
}

pub const NUM_BYTE_OP_LOOKUP_COLS: usize = size_of::<ByteOperationLookupCols<u8>>();
pub const NUM_BYTE_OP_LOOKUP_PREPROCESSED_COLS: usize =
    size_of::<ByteOperationLookupPreprocessedCols<u8>>();
