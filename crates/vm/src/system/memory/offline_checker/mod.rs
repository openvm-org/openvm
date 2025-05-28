mod bridge;
mod bus;
mod columns;

pub use bridge::*;
pub use bus::*;
pub use columns::*;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(C)]
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct MemoryReadAuxRecord {
    pub prev_timestamp: u32,
}
