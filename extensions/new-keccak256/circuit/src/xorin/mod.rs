pub mod execution;
pub mod tests;
pub mod trace;
pub mod columns;
pub mod air;

const KECCAK_WORD_SIZE: usize = 4;

#[derive(derive_new::new, Clone, Copy)]
pub struct XorinVmExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}

