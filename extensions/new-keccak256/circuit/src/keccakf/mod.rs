pub mod execution;
pub mod tests;
pub mod trace;
pub mod columns;
pub struct KeccakfVmExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}