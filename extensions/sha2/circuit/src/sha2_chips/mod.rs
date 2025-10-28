mod block_hasher_chip;
mod config;
mod execution;
mod main_chip;
mod trace;

use std::marker::PhantomData;

pub use block_hasher_chip::*;
pub use config::*;
pub use execution::*;
pub use main_chip::*;
pub use trace::*;

#[cfg(test)]
mod test_utils;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub use test_utils::*;

#[derive(derive_new::new, Clone)]
pub struct Sha2VmExecutor<C: Sha2Config> {
    pub offset: usize,
    pub pointer_max_bits: usize,
    _phantom: PhantomData<C>,
}

// Indicates the message type of the interactions on the sha bus
#[repr(u8)]
pub enum MessageType {
    State,
    Message1,
    Message2,
}
