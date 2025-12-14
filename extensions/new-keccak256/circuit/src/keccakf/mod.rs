pub mod columns;
pub mod execution;
pub mod tests;
pub mod trace;

#[derive(derive_new::new, Clone, Copy)]
pub struct KeccakfVmExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}
