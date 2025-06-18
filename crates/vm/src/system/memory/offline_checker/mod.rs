mod bridge;
mod bus;
mod columns;

pub use bridge::*;
pub use bus::*;
pub use columns::*;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct MemoryReadAuxRecord {
    pub prev_timestamp: u32,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct MemoryWriteAuxRecord<T, const NUM_LIMBS: usize> {
    pub prev_timestamp: u32,
    pub prev_data: [T; NUM_LIMBS],
}

pub type MemoryWriteBytesAuxRecord<const NUM_LIMBS: usize> = MemoryWriteAuxRecord<u8, NUM_LIMBS>;
