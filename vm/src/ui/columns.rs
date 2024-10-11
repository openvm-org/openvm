use afs_derive::AlignedBorrow;

use crate::{arch::ExecutionState, memory::offline_checker::MemoryWriteAuxCols};

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct UiCols<T> {
    pub io: UiIoCols<T>,
    pub aux: UiAuxCols<T>,
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug, Default)]
pub struct UiIoCols<T> {
    pub from_state: ExecutionState<T>,
    pub op_a: T,
    pub op_b: T,
    pub x_cols: [T; 2],
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct UiAuxCols<T> {
    pub is_valid: T,
    pub imm_lo_hex: T, // represents the lowest hex of the immediate value
    pub write_x_aux_cols: MemoryWriteAuxCols<T, 4>,
}
