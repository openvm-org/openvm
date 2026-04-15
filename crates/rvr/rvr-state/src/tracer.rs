//! Tracer state marker trait.
//!
//! `RvState` is generic over a `TracerState` type so that different tracers
//! (no-op, metered, preflight) can be embedded inline without changing the
//! struct layout. Concrete tracer types are defined by the consumer
//! (`rvr-openvm`); this trait only exists to express the FFI-safety contract.

/// Marker trait for FFI-safe tracer state.
///
/// Types implementing this trait can be embedded in `RvState` and must:
/// - Have `#[repr(C)]` layout (or be ZST)
/// - Match the corresponding C `Tracer` struct exactly
pub trait TracerState: Default + Copy {
    /// Tracer kind ID for C API (matches `RV_TRACER_KIND`).
    const KIND: u32;
}

// No tracer - zero-sized type, adds nothing to struct
impl TracerState for () {
    const KIND: u32 = 0;
}
