//! Instruction-retirement state carried by tracked pure execution.

/// Per-call instruction-retirement tracking.
///
/// Generated blocks carry the remaining count in a register and write
/// `retired` back when execution returns to Rust.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InstretTrackingState {
    /// Instructions retired during this execution call.
    pub retired: u64,
    /// Maximum instructions this execution call may retire.
    pub target: u64,
}

impl InstretTrackingState {
    #[must_use]
    pub const fn unlimited() -> Self {
        Self {
            retired: 0,
            target: u64::MAX,
        }
    }

    #[must_use]
    pub const fn with_limit(target: u64) -> Self {
        Self { retired: 0, target }
    }
}

impl Default for InstretTrackingState {
    fn default() -> Self {
        Self::unlimited()
    }
}
