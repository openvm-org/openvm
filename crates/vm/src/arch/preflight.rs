pub use rvr_state::{
    PreflightFieldBlock, PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent,
};

use super::{AddressSpaceHostLayout, VmState, BLOCK_FE_WIDTH};
use crate::system::memory::online::GuestMemory;

/// Append-only memory history produced during serial preflight execution.
///
/// Integer blocks are stored inline in `accesses` and `initial_writes`.
/// Field blocks use dense sidecars so the fixed-size event ABI stays compact.
#[derive(Clone, Debug, Default)]
pub struct PreflightMemoryLog {
    pub accesses: Vec<PreflightMemoryEvent>,
    pub initial_writes: Vec<PreflightInitialWrite>,
    pub field_values: Vec<PreflightFieldBlock>,
    pub field_initial_values: Vec<PreflightFieldBlock>,
}

/// Minimal immutable history consumed by postflight and trace generation.
///
/// Execution state and exit status remain normal VM outputs; they are not
/// duplicated here.
#[derive(Clone, Debug, Default)]
pub struct PreflightHistory {
    pub program: Vec<PreflightProgramEvent>,
    pub memory: PreflightMemoryLog,
}

/// Result of serial preflight execution.
///
/// `history` is the immutable input to postflight. Architectural state and
/// exit status remain ordinary execution outputs and are not duplicated in
/// the history.
pub struct PreflightOutput {
    pub history: PreflightHistory,
    pub state: VmState<GuestMemory>,
    pub exit_code: Option<u32>,
}

impl PreflightOutput {
    /// Preserve sparse-transfer bookkeeping when interpreter writes bypass the
    /// normal online-memory access path.
    pub(crate) fn mark_written_pages(&mut self) {
        let memory = &mut self.state.memory.memory;
        for write in self
            .history
            .memory
            .accesses
            .iter()
            .filter(|event| event.is_write())
        {
            let address_space = write.address_space() as usize;
            let cell_size = memory.config[address_space].layout.size();
            memory.touched_pages[address_space].mark_byte_range(
                write.pointer as usize * cell_size,
                BLOCK_FE_WIDTH * cell_size,
            );
        }
    }
}
