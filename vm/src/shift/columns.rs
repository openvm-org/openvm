use afs_derive::AlignedBorrow;

use crate::{
    arch::ExecutionState,
    memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
    uint_multiplication::MemoryData,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ShiftCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub io: ShiftIoCols<T, NUM_LIMBS, LIMB_BITS>,
    pub aux: ShiftAuxCols<T, NUM_LIMBS, LIMB_BITS>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ShiftIoCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub from_state: ExecutionState<T>,
    pub x: MemoryData<T, NUM_LIMBS, LIMB_BITS>,
    pub y: MemoryData<T, NUM_LIMBS, LIMB_BITS>,
    pub z: MemoryData<T, NUM_LIMBS, LIMB_BITS>,
    pub ptr_as: T,
    pub address_as: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ShiftAuxCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub is_valid: T,

    // Each limb is shifted by bit_shift, where y[0] = bit_shift + LIMB_BITS * bit_quotient and
    // bit_multiplier = 2^bit_shift
    pub bit_shift: T,
    pub bit_multiplier_left: T,
    pub bit_multiplier_right: T,

    // Sign of x for SRA
    pub x_sign: T,

    // Boolean columns that are 1 exactly at the index of the bit/limb shift amount
    pub bit_shift_marker: [T; LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],

    // Part of each x[i] that gets bit shifted to the next limb
    pub bit_shift_carry: [T; NUM_LIMBS],

    // Opcode flags for different operations
    pub opcode_sll_flag: T,
    pub opcode_srl_flag: T,
    pub opcode_sra_flag: T,

    // Pointer read auxiliary columns for [z, x, y]
    pub read_ptr_aux_cols: [MemoryReadAuxCols<T, 1>; 3],
    pub read_x_aux_cols: MemoryReadAuxCols<T, NUM_LIMBS>,
    pub read_y_aux_cols: MemoryReadAuxCols<T, NUM_LIMBS>,
    pub write_z_aux_cols: MemoryWriteAuxCols<T, NUM_LIMBS>,
}
