//! OpenVM VmState <-> rvr RvState bridging.

use std::ffi::c_void;

use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_state::{GuardedMemory, InstretSuspender, Rv32State, RvState, NUM_REGS_I};

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

/// Initialize memory from a VmExe into a GuardedMemory buffer.
fn init_memory<F: PrimeField32>(exe: &VmExe<F>, memory: &mut GuardedMemory) {
    for (&(addr_space, addr), &byte) in &exe.init_memory {
        if addr_space == 2 {
            let addr = addr as usize;
            debug_assert!(
                addr < memory.size(),
                "init_memory address {addr:#x} exceeds buffer"
            );
            unsafe { memory.write_u8(addr, byte) };
        }
    }
}

/// Initialize a PureState from a VmExe (no-op tracing, suspension disabled).
pub fn init_rvr_state<F: PrimeField32>(exe: &VmExe<F>, memory: &mut GuardedMemory) -> PureState {
    let mut state = PureState::new();
    init_memory(exe, memory);
    state.set_memory(memory.as_mut_ptr());
    state.pc = exe.pc_start;
    state.suspender.disable();
    state
}

/// Initialize a state with MeteredCostMeter tracer (suspension disabled).
pub fn init_rvr_state_with_metered_cost<F: PrimeField32>(
    exe: &VmExe<F>,
    memory: &mut GuardedMemory,
) -> MeteredCostState {
    let mut state = MeteredCostState::new();
    init_memory(exe, memory);
    state.set_memory(memory.as_mut_ptr());
    state.pc = exe.pc_start;
    state.suspender.disable();
    state
}

/// Initialize a state with MeteredTracer for per-chip metered execution (suspension disabled).
pub fn init_rvr_state_with_metered<F: PrimeField32>(
    exe: &VmExe<F>,
    memory: &mut GuardedMemory,
) -> MeteredState {
    let mut state = MeteredState::new();
    init_memory(exe, memory);
    state.set_memory(memory.as_mut_ptr());
    state.pc = exe.pc_start;
    state.suspender.disable();
    state
}

/// Get a void pointer to the state for FFI.
pub fn state_as_void_ptr<S>(state: &mut S) -> *mut c_void {
    state as *mut S as *mut c_void
}

/// Copy registers from an RvState back to the OpenVM register format.
pub fn extract_registers(state: &Rv32State) -> [(u32, [u8; 4]); 32] {
    let mut result = [(0u32, [0u8; 4]); 32];
    for (i, reg) in state.regs.iter().enumerate() {
        result[i] = (*reg, reg.to_le_bytes());
    }
    result
}
