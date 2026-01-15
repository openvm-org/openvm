mod air;
mod columns;
mod execution;
#[cfg(test)]
pub mod tests;
/// Preflight and CPU trace generation
mod trace;

pub use air::*;
pub use columns::*;
pub use trace::*;

const NUM_OP_ROWS_PER_INS: usize = 2;

#[derive(derive_new::new, Clone, Copy)]
pub struct KeccakfExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}
