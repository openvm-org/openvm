//! Initialization helpers for the rvr `RvState` scratch struct.
//!
//! `RvState` is a `#[repr(C)]` shim that the rvr-generated C code reads/writes
//! by fixed offsets. It is built fresh per execution and points at VmState's
//! existing memory buffer; we do not own a separate allocation.

use rvr_state::{InstretSuspender, RvState, TracerState};

use super::{
    metered::MeteredTracer,
    metered_cost::{MeteredCostMeter, PureTracer},
};

// ── Generic tracer pointer wrapper ───────────────────────────────────────────

/// A payload type that can sit behind a [`TracerPtr`]. The const `KIND`
/// matches the tracer-kind ABI the rvr-generated C code expects.
pub trait TracerPayload {
    const KIND: u32;
}

/// `#[repr(transparent)]` pointer to a tracer payload. Matches the C
/// `Tracer*` ABI (8 bytes). Default is a null pointer; consumers set the
/// pointer at execute-time via `TracerPtr(&mut payload)`.
///
/// All three concrete tracer wrappers (`PureTracer`, `MeteredCostMeter`,
/// `MeteredTracer`) are type aliases of `TracerPtr<P>` for the matching
/// payload `P`, so the boilerplate `Default`/`TracerState`/`Deref`/`DerefMut`
/// impls live once here rather than three times.
#[repr(transparent)]
pub struct TracerPtr<T>(pub *mut T);

impl<T> Clone for TracerPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for TracerPtr<T> {}

impl<T> Default for TracerPtr<T> {
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl<T: TracerPayload> TracerState for TracerPtr<T> {
    const KIND: u32 = T::KIND;
}

impl<T> std::ops::Deref for TracerPtr<T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.0 }
    }
}

impl<T> std::ops::DerefMut for TracerPtr<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.0 }
    }
}

/// Type alias for RV32 state with PureTracer and instret-based suspension.
pub type PureState = RvState<rvr_state::Rv32, PureTracer, InstretSuspender>;

/// Type alias for RV32 state with MeteredCostMeter and instret-based suspension.
pub type MeteredCostState = RvState<rvr_state::Rv32, MeteredCostMeter, InstretSuspender>;

/// Type alias for RV32 state with MeteredTracer and instret-based suspension.
pub type MeteredState = RvState<rvr_state::Rv32, MeteredTracer, InstretSuspender>;

/// Build a `PureState` whose memory pointer aliases `memory_ptr` and pc starts at `pc`.
///
/// Suspender is disabled by default; the caller may re-arm it via `state.suspender.set_target(_)`.
///
/// # Safety
///
/// `memory_ptr` must be a valid mutable pointer to at least `MEM_SIZE` bytes that stay
/// alive and untouched by anything else for the duration of rvr execution.
pub fn init_rvr_state(memory_ptr: *mut u8, pc: u32) -> PureState {
    let mut state = PureState::new();
    state.set_memory(memory_ptr);
    state.pc = pc;
    state.suspender.disable();
    state
}

/// Build a `MeteredCostState`. See [`init_rvr_state`] for safety.
pub fn init_rvr_state_with_metered_cost(memory_ptr: *mut u8, pc: u32) -> MeteredCostState {
    let mut state = MeteredCostState::new();
    state.set_memory(memory_ptr);
    state.pc = pc;
    state.suspender.disable();
    state
}

/// Build a `MeteredState`. See [`init_rvr_state`] for safety.
pub fn init_rvr_state_with_metered(memory_ptr: *mut u8, pc: u32) -> MeteredState {
    let mut state = MeteredState::new();
    state.set_memory(memory_ptr);
    state.pc = pc;
    state.suspender.disable();
    state
}
