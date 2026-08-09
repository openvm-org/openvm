mod air;
pub mod columns;
mod execution;
#[cfg(test)]
pub mod tests;
/// Preflight and CPU trace generation
pub mod trace;

pub use air::*;
pub use columns::*;
pub use trace::*;

pub const NUM_OP_ROWS_PER_INS: usize = 1;

#[derive(derive_new::new, Clone, Copy)]
pub struct Poseidon2PermuteExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}
