pub mod columns;
pub mod execution;
pub mod tests;
pub mod trace;
pub struct KeccakfVmExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}
