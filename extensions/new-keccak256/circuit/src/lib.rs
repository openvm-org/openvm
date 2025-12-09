pub mod execution;
pub mod tests;
pub mod trace;

mod extension;

pub use extension::*;

const KECCAK_WORD_SIZE: usize = 4;

#[derive(derive_new::new, Clone, Copy)]
pub struct XorinVmExecutor {

}