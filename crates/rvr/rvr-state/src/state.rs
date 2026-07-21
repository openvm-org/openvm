//! RV64 machine state shared with generated C.

/// Number of RV64 general-purpose registers.
pub const NUM_REGS: usize = 32;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionStatus {
    Running = 0,
    Terminated = 1,
    Suspended = 2,
    Trapped = 3,
}

/// RV64 machine state passed to a generated artifact.
///
/// `ModeState` is the single payload selected by the artifact's execution
/// kind: `()` for pure execution, [`crate::InstretTrackingState`] for tracked pure
/// execution, and the corresponding metering state for metered execution.
#[repr(C)]
pub struct RvState<ModeState = ()> {
    pub regs: [u64; NUM_REGS],
    pub pc: u64,
    pub status: u8,
    pub exit_code: u8,
    /// Keeps `memory` naturally aligned and makes the shared Rust/C layout explicit.
    pub padding: [u8; 6],
    pub memory: *mut u8,
    pub mode_state: ModeState,
}

impl<ModeState: Default> RvState<ModeState> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<ModeState: Default> Default for RvState<ModeState> {
    fn default() -> Self {
        Self {
            regs: [0; NUM_REGS],
            pc: 0,
            status: ExecutionStatus::Running as u8,
            exit_code: 0,
            padding: [0; 6],
            memory: std::ptr::null_mut(),
            mode_state: ModeState::default(),
        }
    }
}

impl<ModeState> RvState<ModeState> {
    pub const fn as_void_ptr(&mut self) -> *mut std::ffi::c_void {
        std::ptr::from_mut::<Self>(self).cast::<std::ffi::c_void>()
    }

    pub const fn set_memory(&mut self, memory: *mut u8) {
        self.memory = memory;
    }

    pub const fn execution_status(&self) -> ExecutionStatus {
        match self.status {
            0 => ExecutionStatus::Running,
            1 => ExecutionStatus::Terminated,
            2 => ExecutionStatus::Suspended,
            3 => ExecutionStatus::Trapped,
            _ => ExecutionStatus::Running,
        }
    }

    pub const fn exit_code(&self) -> u8 {
        self.exit_code
    }

    pub const fn is_terminated(&self) -> bool {
        matches!(self.execution_status(), ExecutionStatus::Terminated)
    }

    pub const fn is_suspended(&self) -> bool {
        matches!(self.execution_status(), ExecutionStatus::Suspended)
    }
}
