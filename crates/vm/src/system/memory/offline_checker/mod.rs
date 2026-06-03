mod bridge;
mod bus;
mod columns;

pub use bridge::*;
pub use bus::*;
pub use columns::*;
use openvm_circuit_primitives_derive::AlignedBytesBorrow;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct MemoryReadAuxRecord {
    pub prev_timestamp: u32,
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone, Copy)]
pub struct MemoryWriteAuxRecord<T, const NUM_LIMBS: usize> {
    pub prev_timestamp: u32,
    pub prev_data: [T; NUM_LIMBS],
}

impl<T: Default + Copy, const NUM_LIMBS: usize> Default for MemoryWriteAuxRecord<T, NUM_LIMBS> {
    fn default() -> Self {
        Self {
            prev_timestamp: 0,
            prev_data: [T::default(); NUM_LIMBS],
        }
    }
}

pub type MemoryWriteBytesAuxRecord<const NUM_LIMBS: usize> = MemoryWriteAuxRecord<u8, NUM_LIMBS>;
pub type MemoryWriteU16AuxRecord<const NUM_LIMBS: usize> = MemoryWriteAuxRecord<u16, NUM_LIMBS>;
