//! Initialization helpers for the rvr `RvState` scratch struct.
//!
//! `RvState` is a `#[repr(C)]` shim that the rvr-generated C code reads/writes
//! by fixed offsets. It is built fresh per execution and points at VmState's
//! existing memory buffer; we do not own a separate allocation.

use rvr_state::{InstretTrackingState, RvState};

use super::{
    bridge::rv64_memory_ptr, metered::MeteringState, metered_cost::MeteredCostState,
    preflight::PreflightTracerData,
};
use crate::{arch::VmState, system::memory::online::GuestMemory};

pub(crate) type PureRvState = RvState;
pub(crate) type PureWithInstretTrackingRvState = RvState<InstretTrackingState>;
pub(crate) type MeteredRvState = RvState<MeteringState>;
pub(crate) type MeteredCostRvState = RvState<MeteredCostState>;

/// Preflight-specific execution state. The generated C layout keeps these
/// fields flattened after the common `RvState` prefix; the nested Rust field
/// is layout-equivalent under `repr(C)`.
#[repr(C)]
pub struct PreflightModeState {
    pub instret: u64,
    pub target_instret: u64,
    pub tracer: *mut PreflightTracerData,
}

impl Default for PreflightModeState {
    fn default() -> Self {
        Self {
            instret: 0,
            target_instret: u64::MAX,
            tracer: std::ptr::null_mut(),
        }
    }
}

pub type PreflightState = RvState<PreflightModeState>;

/// Build the concrete scratch state selected by the generated artifact.
///
/// The guest-memory buffer must remain alive, unaliased, and unreallocated
/// while the generated code uses the stored pointer.
pub(crate) fn init_rvr_state<ModeState: Default>(
    vm_state: &mut VmState<GuestMemory>,
    pc: u32,
) -> RvState<ModeState> {
    let memory_ptr = rv64_memory_ptr(vm_state);
    let mut state = RvState::new();
    state.set_memory(memory_ptr);
    state.pc = pc as u64;
    state
}
