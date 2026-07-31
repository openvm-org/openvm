use openvm_cpu_backend::CpuBackend;
use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::{prover::CommittedTraceData, StarkProtocolConfig};

#[cfg(test)]
pub mod tests;

mod air;
mod bus;
pub mod trace;

pub use air::*;
pub use bus::*;

const EXIT_CODE_FAIL: usize = 1;

// For CPU backend only
pub struct ProgramChip<SC: StarkProtocolConfig> {
    pub(super) cached: Option<CommittedTraceData<CpuBackend<SC>>>,
}

impl<SC: StarkProtocolConfig> ProgramChip<SC> {
    pub(super) fn unloaded() -> Self {
        Self { cached: None }
    }
}
