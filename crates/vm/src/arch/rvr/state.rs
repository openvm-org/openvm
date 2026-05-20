//! Initialization helpers for the rvr `RvState` scratch struct.
//!
//! `RvState` is a `#[repr(C)]` shim that the rvr-generated C code reads/writes
//! by fixed offsets. It is built fresh per execution and points at VmState's
//! existing memory buffer; we do not own a separate allocation.

use openvm_stark_backend::p3_field::PrimeField32;
use rvr_state::{InstretSuspender, Rv32, RvState, TracerState};

use super::{
    bridge::rv32_memory_ptr,
    metered::MeteredTracer,
    metered_cost::{MeteredCostMeter, PureTracer},
};
use crate::{arch::VmState, system::memory::online::GuestMemory};

pub type PureState = RvState<Rv32, PureTracer, InstretSuspender>;
pub type MeteredCostState = RvState<Rv32, MeteredCostMeter, InstretSuspender>;
pub type MeteredState = RvState<Rv32, MeteredTracer, InstretSuspender>;

/// A payload type that can sit behind a [`TracerPtr`]. The const `KIND`
/// matches the tracer-kind ABI the rvr-generated C code expects.
pub trait TracerPayload {
    const KIND: u32;
}

/// `#[repr(transparent)]` pointer to a tracer payload. Matches the C
/// `Tracer*` ABI (8 bytes). Default is a null pointer; consumers set the
/// pointer at execute-time via `TracerPtr(&mut payload)`.
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
        debug_assert!(
            !self.0.is_null(),
            "TracerPtr dereferenced before payload was assigned"
        );
        unsafe { &*self.0 }
    }
}

impl<T> std::ops::DerefMut for TracerPtr<T> {
    fn deref_mut(&mut self) -> &mut T {
        debug_assert!(
            !self.0.is_null(),
            "TracerPtr dereferenced before payload was assigned"
        );
        unsafe { &mut *self.0 }
    }
}

/// Build a `PureState` whose memory pointer aliases `vm_state`'s
/// `RV32_MEMORY_AS` segment, with pc set to `pc`. Suspender is disabled by
/// default; the caller may re-arm it via `state.suspender.set_target(_)`.
///
/// The stored pointer is dereferenced later by the rvr C engine; the segment
/// must stay alive and be uniquely accessed for the duration of rvr execution.
pub fn init_rvr_state<F: PrimeField32>(
    vm_state: &mut VmState<F, GuestMemory>,
    pc: u32,
) -> PureState {
    let memory_ptr = rv32_memory_ptr(vm_state);
    let mut state = PureState::new();
    state.set_memory(memory_ptr);
    state.pc = pc;
    state.suspender.disable();
    state
}

pub fn init_rvr_state_with_metered_cost<F: PrimeField32>(
    vm_state: &mut VmState<F, GuestMemory>,
    pc: u32,
) -> MeteredCostState {
    let memory_ptr = rv32_memory_ptr(vm_state);
    let mut state = MeteredCostState::new();
    state.set_memory(memory_ptr);
    state.pc = pc;
    state.suspender.disable();
    state
}

pub fn init_rvr_state_with_metered<F: PrimeField32>(
    vm_state: &mut VmState<F, GuestMemory>,
    pc: u32,
) -> MeteredState {
    let memory_ptr = rv32_memory_ptr(vm_state);
    let mut state = MeteredState::new();
    state.set_memory(memory_ptr);
    state.pc = pc;
    state.suspender.disable();
    state
}
