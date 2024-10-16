use afs_derive::AlignedBorrow;

use crate::{
    arch::ExecutionState,
    old::uint_multiplication::MemoryData,
    system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ArithmeticLogicCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub io: ArithmeticLogicIoCols<T, NUM_LIMBS, LIMB_BITS>,
    pub aux: ArithmeticLogicAuxCols<T, NUM_LIMBS, LIMB_BITS>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ArithmeticLogicIoCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub from_state: ExecutionState<T>,
    pub x: MemoryData<T, NUM_LIMBS, LIMB_BITS>,
    pub y: MemoryData<T, NUM_LIMBS, LIMB_BITS>,
    pub z: MemoryData<T, NUM_LIMBS, LIMB_BITS>,
    pub cmp_result: T,
    pub ptr_as: T,
    pub address_as: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ArithmeticLogicAuxCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub is_valid: T,
    pub x_sign: T,
    pub y_sign: T,

    // Opcode flags for different operations
    pub opcode_add_flag: T,
    pub opcode_sub_flag: T,
    pub opcode_sltu_flag: T,
    pub opcode_eq_flag: T,
    pub opcode_xor_flag: T,
    pub opcode_and_flag: T,
    pub opcode_or_flag: T,
    pub opcode_slt_flag: T,

    /// Pointer read auxiliary columns for [z, x, y].
    /// **Note** the ordering, which is designed to match the instruction order.
    pub read_ptr_aux_cols: [MemoryReadAuxCols<T, 1>; 3],
    pub read_x_aux_cols: MemoryReadAuxCols<T, NUM_LIMBS>,
    pub read_y_aux_cols: MemoryReadAuxCols<T, NUM_LIMBS>,
    pub write_z_aux_cols: MemoryWriteAuxCols<T, NUM_LIMBS>,
    pub write_cmp_aux_cols: MemoryWriteAuxCols<T, 1>,
}
