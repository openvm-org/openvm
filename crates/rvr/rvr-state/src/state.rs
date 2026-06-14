//! RISC-V machine state struct.
//!
//! Layout must match the generated C `RvState` struct exactly.

use crate::{suspender::SuspenderState, tracer::TracerState, xlen::Xlen};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionStatus {
    Running = 0,
    Terminated = 1,
    Suspended = 2,
    Trapped = 3,
}

/// Number of registers for I extension (32 GPRs).
pub const NUM_REGS_I: usize = 32;

/// RISC-V machine state.
///
/// This struct has a C-compatible layout matching the generated C header.
/// The layout is parameterized by:
/// - `X`: Register width (Rv64)
/// - `T`: Tracer state type (ZST when `()`, real struct when tracing)
/// - `S`: Suspender state type (ZST when `()`, real struct when suspending)
/// - `NUM_REGS`: Number of general-purpose registers
#[repr(C)]
pub struct RvState<
    X: Xlen,
    T: TracerState = (),
    S: SuspenderState = (),
    const NUM_REGS: usize = NUM_REGS_I,
> {
    /// General-purpose registers (hot - most frequently accessed).
    pub regs: [X::Reg; NUM_REGS],

    /// Program counter (hot).
    pub pc: X::Reg,

    /// Instructions retired counter (hot).
    pub instret: u64,

    /// Suspender state (ZST when S = (), real struct when suspending).
    pub suspender: S,

    /// Legacy execution-status byte.
    pub has_exited: u8,

    /// Legacy result payload byte.
    pub exit_code: u8,

    /// Guest memory pointer (cold).
    pub memory: *mut u8,

    /// Tracer state (ZST when T = (), real struct when tracing).
    pub tracer: T,
}

impl<X: Xlen, T: TracerState, S: SuspenderState, const NUM_REGS: usize> RvState<X, T, S, NUM_REGS> {
    /// Create a new zeroed state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<X: Xlen, T: TracerState, S: SuspenderState, const NUM_REGS: usize> Default
    for RvState<X, T, S, NUM_REGS>
{
    fn default() -> Self {
        Self {
            regs: [X::from_u64(0); NUM_REGS],
            pc: X::from_u64(0),
            instret: 0,
            suspender: S::default(),
            has_exited: 0,
            exit_code: 0,
            memory: std::ptr::null_mut(),
            tracer: T::default(),
        }
    }
}

impl<X: Xlen, T: TracerState, S: SuspenderState, const NUM_REGS: usize> RvState<X, T, S, NUM_REGS> {
    /// Get state as a void pointer (for FFI).
    pub const fn as_void_ptr(&mut self) -> *mut std::ffi::c_void {
        std::ptr::from_mut::<Self>(self).cast::<std::ffi::c_void>()
    }

    /// Set memory pointer.
    pub const fn set_memory(&mut self, memory: *mut u8) {
        self.memory = memory;
    }

    /// Get the execution status.
    pub const fn execution_status(&self) -> ExecutionStatus {
        match self.has_exited {
            0 => ExecutionStatus::Running,
            1 => ExecutionStatus::Terminated,
            2 => ExecutionStatus::Suspended,
            3 => ExecutionStatus::Trapped,
            _ => ExecutionStatus::Running,
        }
    }

    /// Get the raw result payload byte associated with the execution status.
    pub const fn result_code(&self) -> u8 {
        self.exit_code
    }

    /// Check whether execution status indicates termination.
    pub const fn is_terminated(&self) -> bool {
        matches!(self.execution_status(), ExecutionStatus::Terminated)
    }

    /// Check whether execution status indicates suspension.
    pub const fn is_suspended(&self) -> bool {
        matches!(self.execution_status(), ExecutionStatus::Suspended)
    }
}

/// Type alias for RV64I state (64-bit registers, 32 GPRs, no tracer, no suspender).
pub type Rv64State = RvState<crate::xlen::Rv64, (), (), NUM_REGS_I>;
