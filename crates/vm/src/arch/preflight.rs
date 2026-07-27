pub use rvr_state::{
    PreflightFieldBlock, PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent,
};

use super::VmState;
use crate::system::memory::online::GuestMemory;

/// Append-only memory history produced during serial preflight execution.
///
/// Integer blocks are stored inline in `accesses` and `initial_writes`.
/// Field blocks use dense sidecars so the fixed-size event ABI stays compact.
#[derive(Debug, Default)]
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
#[derive(Debug, Default)]
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
