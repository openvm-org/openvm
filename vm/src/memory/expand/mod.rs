pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub const EXPAND_BUS: usize = 4;

pub struct ExpandAir<const CHUNK: usize> {
    pub height: usize,
}
