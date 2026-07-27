mod air;
mod columns;
mod config;
mod trace;

use std::marker::PhantomData;

pub use air::*;
pub use columns::*;
pub use config::*;
use openvm_circuit::system::memory::SharedMemoryHelper;
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
pub(crate) use trace::generate_trace_from_postflight as generate_main_trace_from_postflight;

use crate::Sha2Config;

pub struct Sha2MainChip<F, C: Sha2Config> {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    pub pointer_max_bits: usize,
    pub mem_helper: SharedMemoryHelper<F>,
    _phantom: PhantomData<C>,
}

impl<F, C: Sha2Config> Sha2MainChip<F, C> {
    pub fn new(
        range_checker_chip: SharedVariableRangeCheckerChip,
        pointer_max_bits: usize,
        mem_helper: SharedMemoryHelper<F>,
    ) -> Self {
        Self {
            range_checker_chip,
            pointer_max_bits,
            mem_helper,
            _phantom: PhantomData,
        }
    }
}
