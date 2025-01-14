use openvm_circuit::{
    arch::ExecutionState,
    system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};
use openvm_circuit_primitives::AlignedBorrow;
use openvm_poseidon2_air::Poseidon2SubCols;
use openvm_stark_backend::p3_field::FieldAlgebra;

use super::NATIVE_POSEIDON2_CHUNK_SIZE;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativePoseidon2Cols<T, const SBOX_REGISTERS: usize> {
    pub inner: Poseidon2SubCols<T, SBOX_REGISTERS>,
    pub memory: NativePoseidon2MemoryCols<T>,
}

#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct NativePoseidon2MemoryCols<T> {
    pub from_state: ExecutionState<T>,
    // 1 if PERMUTE, 2 if COMPRESS, 0 otherwise
    pub opcode_flag: T,

    pub ptr_as: T,
    pub chunk_as: T,

    // rs_ptr[1] if COMPRESS, original value of instruction field c if PERMUTE
    pub c: T,

    pub rs_ptr: [T; 2],
    pub rd_ptr: T,
    pub rs_val: [T; 2],
    pub rd_val: T,
    pub rs_read_aux: [MemoryReadAuxCols<T>; 2],
    pub rd_read_aux: MemoryReadAuxCols<T>,

    pub chunk_read_aux: [MemoryReadAuxCols<T>; 2],
    pub chunk_write_aux: [MemoryWriteAuxCols<T, NATIVE_POSEIDON2_CHUNK_SIZE>; 2],
}

impl<F: FieldAlgebra + Copy> NativePoseidon2MemoryCols<F> {
    pub fn blank() -> Self {
        Self {
            from_state: ExecutionState::default(),
            opcode_flag: F::ZERO,
            ptr_as: F::ZERO,
            chunk_as: F::ZERO,
            c: F::ZERO,
            rs_ptr: [F::ZERO; 2],
            rd_ptr: F::ZERO,
            rs_val: [F::ZERO; 2],
            rd_val: F::ZERO,
            rs_read_aux: [MemoryReadAuxCols::disabled(); 2],
            rd_read_aux: MemoryReadAuxCols::disabled(),
            chunk_read_aux: [MemoryReadAuxCols::disabled(); 2],
            chunk_write_aux: [MemoryWriteAuxCols::disabled(); 2],
        }
    }
}
