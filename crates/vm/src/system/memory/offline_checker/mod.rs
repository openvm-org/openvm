mod bridge;
mod bus;
mod columns;

pub use bridge::*;
pub use bus::*;
pub use columns::*;

#[repr(C)]
pub struct MemoryReadAuxRecord {
    pub prev_timestamp: u32,
}

#[repr(C)]
pub struct MemoryWriteAuxRecord<DATA> {
    pub prev_timestamp: u32,
    pub prev_data: DATA,
}
