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
use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;
pub use trace::*;

#[derive(derive_new::new, Clone)]
pub struct Sha2VmExecutor<C: Sha2Config> {
    pub offset: usize,
    pub pointer_max_bits: usize,
    _phantom: PhantomData<C>,
}

#[derive(derive_new::new)]
pub struct Sha2VmFiller<C: Sha2Config> {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub pointer_max_bits: usize,
    _phantom: PhantomData<C>,
}
