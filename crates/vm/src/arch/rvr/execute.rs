//! Load .so, bridge state, call rv_execute.
//!
//! Each execution path takes `&mut VmState<F, GuestMemory>` directly: the
//! transient `RvState` scratch struct aliases VmState's memory and registers,
//! and `OpenVmIoState` borrows VmState's `Streams<F>` and rng. There is no
//! separately-owned guest memory or stream conversion.

use std::ffi::c_void;

use openvm_stark_backend::p3_field::PrimeField32;
use rvr_state::{MemoryError, Rv32, RvState, SuspenderState, TracerState, NUM_REGS_I};

use super::{
    bridge::{public_values_slice, read_rv32_registers, rv32_memory_ptr, write_rv32_registers},
    compile::RvrCompiled,
    io::{
        host_deferral_call_lookup, host_deferral_output_lookup, host_hint_buffer, host_hint_input,
        host_hint_random, host_hint_storew, host_hint_stream_set, host_print_str, host_reveal,
        OpenVmHostCallbacks, OpenVmIoState,
    },
    metered::{
        metered_periodic_check, MeteredConfig, MeteredTracerData, RvrMeteredResult,
        SegmentationState, NO_LAST_PAGE,
    },
    metered_cost::{prepare_metered_cost, MeteredCostConfig, MeteredCostData, PureTracerData},
    state::{
        init_rvr_state, init_rvr_state_with_metered, init_rvr_state_with_metered_cost,
        MeteredCostState, PureState, TracerPtr,
    },
};
use crate::{arch::VmState, system::memory::online::GuestMemory};

/// Error during execution.
#[derive(Debug, thiserror::Error)]
pub enum ExecuteError {
    #[error("symbol lookup failed: {0}")]
    SymbolLookup(String),
    #[error("execution returned error code: {0}")]
    ExecutionFailed(i32),
    #[error("guest exited with non-zero exit code: {0}")]
    GuestExit(u8),
    #[error("memory allocation failed: {0}")]
    MemoryAlloc(#[from] MemoryError),
}

/// `suspended` is `false` for unlimited runs.
pub struct RvrPureResult {
    pub state: PureState,
    pub suspended: bool,
}

/// `suspended` is `false` for unlimited runs.
pub struct RvrMeteredCostResult {
    pub state: MeteredCostState,
    pub cost: u64,
    pub suspended: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecuteOutcome {
    Running,
    Terminated,
    Suspended,
}

/// Read-only view of any RV32 rvr state, regardless of tracer variant.
pub trait RvrStateInspect {
    fn pc(&self) -> u32;
    fn instret(&self) -> u64;
    fn regs(&self) -> &[u32; NUM_REGS_I];
    fn is_terminated(&self) -> bool;
    fn is_suspended(&self) -> bool;
    fn result_code(&self) -> u8;
    fn as_void_ptr(&mut self) -> *mut c_void;
}

fn outcome_of<S: RvrStateInspect>(state: &S) -> ExecuteOutcome {
    if state.is_terminated() {
        ExecuteOutcome::Terminated
    } else if state.is_suspended() {
        ExecuteOutcome::Suspended
    } else {
        ExecuteOutcome::Running
    }
}

impl<T: TracerState, S: SuspenderState> RvrStateInspect for RvState<Rv32, T, S> {
    fn pc(&self) -> u32 {
        self.pc
    }
    fn instret(&self) -> u64 {
        self.instret
    }
    fn regs(&self) -> &[u32; NUM_REGS_I] {
        &self.regs
    }
    fn is_terminated(&self) -> bool {
        Self::is_terminated(self)
    }
    fn is_suspended(&self) -> bool {
        Self::is_suspended(self)
    }
    fn result_code(&self) -> u8 {
        Self::result_code(self)
    }
    fn as_void_ptr(&mut self) -> *mut c_void {
        Self::as_void_ptr(self)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

pub fn build_callbacks<F: PrimeField32>(
    io_state: &mut OpenVmIoState<'_, F>,
) -> OpenVmHostCallbacks {
    OpenVmHostCallbacks {
        ctx: io_state as *mut OpenVmIoState<'_, F> as *mut c_void,
        hint_input: host_hint_input::<F>,
        print_str: host_print_str::<F>,
        hint_random: host_hint_random::<F>,
        hint_storew: host_hint_storew::<F>,
        hint_buffer: host_hint_buffer::<F>,
        reveal: host_reveal::<F>,
        hint_stream_set: host_hint_stream_set::<F>,
        deferral_call_lookup: host_deferral_call_lookup::<F>,
        deferral_output_lookup: host_deferral_output_lookup::<F>,
    }
}

fn build_io_state_borrowed<'a, F: PrimeField32>(
    vm_state: &'a mut VmState<F, GuestMemory>,
) -> OpenVmIoState<'a, F> {
    let memory_ptr = rv32_memory_ptr(vm_state);
    let streams = &mut vm_state.streams;
    OpenVmIoState {
        input_stream: &mut streams.input_stream,
        hint_stream: &mut streams.hint_stream,
        rng: &mut vm_state.rng,
        memory_ptr,
        public_values: public_values_slice(&mut vm_state.memory.memory),
        deferrals: &mut streams.deferrals,
    }
}

/// # Safety
///
/// - `compiled` must contain a valid rvr-compiled shared library with the expected ABI
///   (`register_openvm_callbacks` and `rv_execute` symbols).
/// - `state_ptr` must point to a valid, mutable RV32 state struct whose tracer variant matches the
///   one compiled into the shared library.
pub unsafe fn register_and_execute(
    compiled: &RvrCompiled,
    callbacks: &OpenVmHostCallbacks,
    state_ptr: *mut c_void,
) -> Result<(), ExecuteError> {
    type RegisterFn = unsafe extern "C" fn(*const OpenVmHostCallbacks);
    let register_fn: RegisterFn = unsafe {
        let sym = compiled
            .lib
            .get::<RegisterFn>(b"register_openvm_callbacks")
            .map_err(|e| ExecuteError::SymbolLookup(e.to_string()))?;
        *sym
    };
    unsafe { register_fn(callbacks) };

    type ExecuteFn = unsafe extern "C" fn(*mut c_void);
    let execute_fn: ExecuteFn = unsafe {
        let sym = compiled
            .lib
            .get::<ExecuteFn>(b"rv_execute")
            .map_err(|e| ExecuteError::SymbolLookup(e.to_string()))?;
        *sym
    };

    unsafe { execute_fn(state_ptr) };
    Ok(())
}

fn report_failure(prefix: &str, outcome: ExecuteOutcome, pc: u32, instret: u64, exit_code: u8) {
    if std::env::var_os("RVR_OPENVM_DEBUG_EXEC_FAILURE").is_some() {
        eprintln!(
            "[rvr-openvm] {prefix}: outcome={outcome:?}, pc={pc:#x}, instret={instret}, guest_exit_code={exit_code}"
        );
    }
}

fn execution_error(
    prefix: &str,
    outcome: ExecuteOutcome,
    pc: u32,
    instret: u64,
    exit_code: u8,
) -> ExecuteError {
    report_failure(prefix, outcome, pc, instret, exit_code);
    if outcome == ExecuteOutcome::Terminated && exit_code != 0 {
        ExecuteError::GuestExit(exit_code)
    } else {
        ExecuteError::ExecutionFailed(match outcome {
            ExecuteOutcome::Running => 0,
            ExecuteOutcome::Terminated => 1,
            ExecuteOutcome::Suspended => 2,
        })
    }
}

/// Run the FFI execute against `vm_state` + `state`, validate the outcome,
/// and on success write the final pc/regs back into `vm_state`.
///
/// `allow_suspended` permits `Suspended` as a successful outcome (the
/// limit-armed callers pass `true`; unlimited callers pass `false`).
fn run_and_finalize<F, S>(
    compiled: &RvrCompiled,
    vm_state: &mut VmState<F, GuestMemory>,
    state: &mut S,
    allow_suspended: bool,
    failure_prefix: &str,
) -> Result<ExecuteOutcome, ExecuteError>
where
    F: PrimeField32,
    S: RvrStateInspect,
{
    let mut io_state = build_io_state_borrowed(vm_state);
    let callbacks = build_callbacks(&mut io_state);
    unsafe { register_and_execute(compiled, &callbacks, state.as_void_ptr()) }?;

    let outcome = outcome_of(state);
    let success = match outcome {
        ExecuteOutcome::Terminated => state.is_terminated() && state.result_code() == 0,
        ExecuteOutcome::Suspended => allow_suspended,
        ExecuteOutcome::Running => false,
    };

    if success {
        write_rv32_registers(vm_state, state.regs());
        vm_state.set_pc(state.pc());
        Ok(outcome)
    } else {
        Err(execution_error(
            failure_prefix,
            outcome,
            state.pc(),
            state.instret(),
            state.result_code(),
        ))
    }
}

// ── Public execute functions ─────────────────────────────────────────────────

/// Execute a VmExe using a compiled rvr shared library against `vm_state`.
///
/// If `limit` is `Some(n)`, the suspender is armed at `n` instructions and a
/// `Suspended` outcome is accepted as success; otherwise only `Terminated`
/// (with exit-code 0) succeeds.
pub fn execute<F: PrimeField32>(
    compiled: &RvrCompiled,
    vm_state: &mut VmState<F, GuestMemory>,
    limit: Option<u64>,
) -> Result<RvrPureResult, ExecuteError> {
    let pc = vm_state.pc();
    let initial_regs = read_rv32_registers(vm_state);

    let mut tracer_data = PureTracerData;
    let mut state = init_rvr_state(vm_state, pc);
    state.regs = initial_regs;
    state.tracer = TracerPtr(&mut tracer_data);
    if let Some(n) = limit {
        state.suspender.set_target(n);
    }

    let outcome = run_and_finalize(
        compiled,
        vm_state,
        &mut state,
        limit.is_some(),
        "execution failed",
    )?;
    Ok(RvrPureResult {
        state,
        suspended: outcome == ExecuteOutcome::Suspended,
    })
}

/// Execute a VmExe with metered cost tracking. If `limit` is `Some(n)`, the
/// suspender is armed at `n` instructions and `Suspended` counts as success.
pub fn execute_metered_cost<F: PrimeField32>(
    compiled: &RvrCompiled,
    vm_state: &mut VmState<F, GuestMemory>,
    metered_cost_config: MeteredCostConfig,
    limit: Option<u64>,
) -> Result<RvrMeteredCostResult, ExecuteError> {
    let pc = vm_state.pc();
    let initial_regs = read_rv32_registers(vm_state);

    let mut tracer_data = MeteredCostData::default();
    let mut state = init_rvr_state_with_metered_cost(vm_state, pc);
    state.regs = initial_regs;
    state.tracer = TracerPtr(&mut tracer_data);

    let widths_u64 = prepare_metered_cost(&metered_cost_config);
    state.tracer.chip_widths = widths_u64.as_ptr();
    state.tracer.cost = 0;
    if let Some(n) = limit {
        state.suspender.set_target(n);
    }

    let outcome = run_and_finalize(
        compiled,
        vm_state,
        &mut state,
        limit.is_some(),
        "metered-cost execution failed",
    )?;
    let cost = state.tracer.cost;
    Ok(RvrMeteredCostResult {
        state,
        cost,
        suspended: outcome == ExecuteOutcome::Suspended,
    })
}

/// Execute a VmExe with per-chip metered execution and segmentation.
pub fn execute_metered<F: PrimeField32>(
    compiled: &RvrCompiled,
    vm_state: &mut VmState<F, GuestMemory>,
    trace_config: MeteredConfig,
) -> Result<RvrMeteredResult, ExecuteError> {
    let pc = vm_state.pc();
    let initial_regs = read_rv32_registers(vm_state);

    let mut tracer_data = MeteredTracerData::default();
    let mut state = init_rvr_state_with_metered(vm_state, pc);
    state.regs = initial_regs;
    state.tracer = TracerPtr(&mut tracer_data);

    let mut seg_state = SegmentationState::new(trace_config);
    state.tracer.trace_heights = seg_state.trace_heights_ptr();
    state.tracer.mem_page_buf = seg_state.mem_page_buf_ptr();
    state.tracer.pv_page_buf = seg_state.pv_page_buf_ptr();
    state.tracer.deferral_page_buf = seg_state.deferral_page_buf_ptr();
    state.tracer.mem_page_buf_len = 0;
    state.tracer.pv_page_buf_len = 0;
    state.tracer.deferral_page_buf_len = 0;
    state.tracer.last_mem_page = NO_LAST_PAGE;
    state.tracer.check_counter = seg_state.config().segment_check_insns as u32;
    state.tracer.on_check = Some(metered_periodic_check);
    state.tracer.seg_state = &mut seg_state as *mut SegmentationState as *mut c_void;

    run_and_finalize(
        compiled,
        vm_state,
        &mut state,
        false,
        "metered execution failed",
    )?;

    seg_state.on_termination(
        state.tracer.mem_page_buf_len,
        state.tracer.pv_page_buf_len,
        state.tracer.deferral_page_buf_len,
        state.tracer.check_counter,
    );
    Ok(seg_state.into_result(state))
}
