use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::{
    config::StarkGenericConfig,
    prover::{cpu::CpuBackend, types::CommittedTraceData},
};

#[cfg(test)]
pub mod tests;

mod air;
mod bus;
pub mod trace;

pub use air::*;
pub use bus::*;

const EXIT_CODE_FAIL: usize = 1;

// For CPU backend only
pub struct ProgramChip<SC: StarkGenericConfig> {
    /// `i` -> frequency of instruction in `i`th row of trace matrix. This requires filtering
    /// `program.instructions_and_debug_infos` to remove gaps.
    pub(super) filtered_exec_frequencies: Vec<u32>,
    pub(super) cached: Option<CommittedTraceData<CpuBackend<SC>>>,
}

impl<SC: StarkGenericConfig> ProgramChip<SC> {
    pub(super) fn unloaded() -> Self {
        Self {
            filtered_exec_frequencies: Vec::new(),
            cached: None,
        }
    }
}
