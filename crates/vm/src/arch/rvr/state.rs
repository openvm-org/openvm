//! Initialization helpers for the rvr `RvState` scratch struct.
//!
//! `RvState` is a `#[repr(C)]` shim that the rvr-generated C code reads/writes
//! by fixed offsets. It is built fresh per execution and points at VmState's
//! existing memory buffer; we do not own a separate allocation.

use rvr_state::{InstretTrackingState, PreflightState, RvState};

use super::{bridge::rv64_memory_ptr, metered::MeteringState, metered_cost::MeteredCostState};
use crate::{arch::VmState, system::memory::online::GuestMemory};

pub(crate) type PureRvState = RvState;
pub(crate) type PureWithInstretTrackingRvState = RvState<InstretTrackingState>;
pub(crate) type MeteredRvState = RvState<MeteringState>;
pub(crate) type MeteredCostRvState = RvState<MeteredCostState>;
pub(crate) type PreflightRvState = RvState<PreflightState>;

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
