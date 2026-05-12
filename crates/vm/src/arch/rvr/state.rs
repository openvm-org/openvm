//! Initialization helpers for the rvr `RvState` scratch struct.
//!
//! `RvState` is a `#[repr(C)]` shim that the rvr-generated C code reads/writes
//! by fixed offsets. It is built fresh per execution and points at VmState's
//! existing memory buffer; we do not own a separate allocation.

use std::ffi::c_void;

use rvr_state::{InstretSuspender, Rv32State, RvState, NUM_REGS_I};

use super::{
    metered::MeteredTracer,
    metered_cost::{MeteredCostMeter, PureTracer},
};

/// Type alias for RV32 state with PureTracer and instret-based suspension.
pub type PureState = RvState<rvr_state::Rv32, PureTracer, InstretSuspender, NUM_REGS_I>;

/// Type alias for RV32 state with MeteredCostMeter and instret-based suspension.
pub type MeteredCostState =
    RvState<rvr_state::Rv32, MeteredCostMeter, InstretSuspender, NUM_REGS_I>;

/// Type alias for RV32 state with MeteredTracer and instret-based suspension.
pub type MeteredState = RvState<rvr_state::Rv32, MeteredTracer, InstretSuspender, NUM_REGS_I>;

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

/// Get a void pointer to the state for FFI.
pub fn state_as_void_ptr<S>(state: &mut S) -> *mut c_void {
    state as *mut S as *mut c_void
}

/// Copy registers from an `RvState` back to the OpenVM register format.
pub fn extract_registers(state: &Rv32State) -> [(u32, [u8; 4]); 32] {
    let mut result = [(0u32, [0u8; 4]); 32];
    for (i, reg) in state.regs.iter().enumerate() {
        result[i] = (*reg, reg.to_le_bytes());
    }
    result
}
