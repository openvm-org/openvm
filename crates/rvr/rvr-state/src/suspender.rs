//! Suspender state types for cooperative execution.
//!
//! Suspenders allow the VM to pause execution at specific points, typically
//! based on instruction count (instret).

/// Marker trait for FFI-safe suspender state.
pub trait SuspenderState: Default + Copy {
    /// Whether this suspender adds fields to the state struct.
    const HAS_FIELDS: bool;
}

// No suspender - zero-sized type, adds nothing to struct
impl SuspenderState for () {
    const HAS_FIELDS: bool = false;
}

/// Instret-based suspender state - suspends when instret >= target.
///
/// Matches C struct field:
/// ```c
/// uint64_t target_instret;
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct InstretSuspender {
    /// Target instruction count for suspension.
    pub target_instret: u64,
}

impl SuspenderState for InstretSuspender {
    const HAS_FIELDS: bool = true;
}

impl InstretSuspender {
    /// Create a new suspender with the given target.
    #[must_use]
    pub const fn new(target_instret: u64) -> Self {
        Self { target_instret }
    }

    /// Set target instret.
    #[inline]
    pub const fn set_target(&mut self, target: u64) {
        self.target_instret = target;
    }

    /// Disable suspension by setting target to max.
    #[inline]
    pub const fn disable(&mut self) {
        self.target_instret = u64::MAX;
    }
}
