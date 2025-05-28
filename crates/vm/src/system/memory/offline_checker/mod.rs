mod bridge;
mod bus;
mod columns;

pub use bridge::*;
pub use bus::*;
pub use columns::*;

#[repr(C)]
#[derive(Debug)]
pub struct MemoryReadAuxRecord {
    pub prev_timestamp: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct MemoryWriteAuxRecord<const NUM_LIMBS: usize> {
    pub prev_timestamp: u32,
    pub prev_data: [u8; NUM_LIMBS],
}
