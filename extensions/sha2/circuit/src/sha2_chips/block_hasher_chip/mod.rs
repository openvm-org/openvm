mod air;
mod columns;

mod config;
mod trace;

use std::marker::PhantomData;

pub use air::*;
pub use columns::*;
pub use config::*;
use openvm_circuit::system::memory::SharedMemoryHelper;
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, var_range::SharedVariableRangeCheckerChip,
};
use openvm_instructions::riscv::RV64_BYTE_BITS;
use openvm_sha2_air::{Sha2BlockHasherFillerHelper, Sha2BlockHasherSubairConfig};
pub(crate) use trace::generate_trace_from_postflight as generate_block_hasher_trace_from_postflight;

pub use super::config::*;

pub struct Sha2BlockHasherChip<F, C: Sha2BlockHasherSubairConfig> {
    pub inner: Sha2BlockHasherFillerHelper<C>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    /// Range checker for digest-row `final_hash` limbs.
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    pub pointer_max_bits: usize,
    pub mem_helper: SharedMemoryHelper<F>,
    _phantom: PhantomData<C>,
}

impl<F, C: Sha2BlockHasherSubairConfig> Sha2BlockHasherChip<F, C> {
    pub fn new(
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
        range_checker_chip: SharedVariableRangeCheckerChip,
        pointer_max_bits: usize,
        mem_helper: SharedMemoryHelper<F>,
    ) -> Self {
        Self {
            inner: Sha2BlockHasherFillerHelper::new(),
            bitwise_lookup_chip,
            range_checker_chip,
            pointer_max_bits,
            mem_helper,
            _phantom: PhantomData,
        }
    }
}
