use rvr_state::{PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent};

/// Host-side logical execution history used by replay validation tests.
///
/// Production preflight derives the corresponding device buffers directly
/// from checkpoints and residuals.
#[derive(Debug)]
pub struct PreflightEventLog {
    pub program_log: Vec<PreflightProgramEvent>,
    pub memory_log: Vec<PreflightMemoryEvent>,
    pub initial_write_log: Vec<PreflightInitialWrite>,
}
