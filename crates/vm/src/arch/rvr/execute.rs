//! Load .so, bridge state, call rv_execute.
//!
//! Each execution path takes `&mut VmState<F, GuestMemory>` directly: the
//! transient `RvState` scratch struct aliases VmState's memory and registers,
//! and `OpenVmIoState` borrows VmState's `Streams<F>` and rng. There is no
//! separately-owned guest memory or stream conversion.

use std::ffi::c_void;

use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_lift::{ExtensionError, ExtensionRegistry};
use rvr_state::{ExecutionStatus, MemoryError, Rv32, RvState, SuspenderState, TracerState};

use super::{
    bridge::{public_values_slice, read_rv32_registers, rv32_memory_ptr, write_rv32_registers},
    compile::RvrCompiled,
    io::{
        host_hint_buffer, host_hint_input, host_hint_random, host_hint_storew,
        host_hint_stream_set, host_print_str, host_reveal, OpenVmHostCallbacks, OpenVmIoState,
    },
    metered::{metered_periodic_check, MeteredTracerData, SegmentationState, NO_LAST_PAGE},
    metered_cost::{
        prepare_metered_cost, MeteredCostConfig, MeteredCostData, PureTracerData,
        RvrMeteredCostResult,
    },
    pure::RvrPureResult,
    state::{
        init_rvr_state, init_rvr_state_with_metered, init_rvr_state_with_metered_cost, TracerPtr,
    },
};
use crate::{arch::VmState, system::memory::online::GuestMemory};

type RegisterFn = unsafe extern "C" fn(*const OpenVmHostCallbacks);
type ExecuteFn = unsafe extern "C" fn(*mut c_void);

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
    #[error("extension host callback registration failed: {0}")]
    ExtensionRegistration(#[from] ExtensionError),
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
/// `compiled` must contain a valid rvr-compiled shared library exporting the
/// `register_openvm_callbacks` symbol with the expected ABI.
pub unsafe fn register_openvm_callbacks(
    compiled: &RvrCompiled,
    callbacks: &OpenVmHostCallbacks,
) -> Result<(), ExecuteError> {
    let register_fn: RegisterFn = unsafe {
        let sym = compiled
            .lib
            .get::<RegisterFn>(b"register_openvm_callbacks")
            .map_err(|e| ExecuteError::SymbolLookup(e.to_string()))?;
        *sym
    };
    unsafe { register_fn(callbacks) };
    Ok(())
}

/// # Safety
///
/// - `compiled` must contain a valid rvr-compiled shared library exporting the `rv_execute` symbol.
/// - `state_ptr` must point to a valid, mutable RV32 state struct whose tracer variant matches the
///   one compiled into the shared library.
pub unsafe fn rv_execute(
    compiled: &RvrCompiled,
    state_ptr: *mut c_void,
) -> Result<(), ExecuteError> {
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

/// Run the FFI execute against `vm_state` + `state`, validate the outcome,
/// and on success write the final pc/regs back into `vm_state`.
///
/// `allow_suspended` permits `Suspended` as a successful outcome (the
/// limit-armed callers pass `true`; unlimited callers pass `false`).
fn run_and_finalize<F, T, S>(
    compiled: &RvrCompiled,
    extensions: &ExtensionRegistry<F>,
    vm_state: &mut VmState<F, GuestMemory>,
    state: &mut RvState<Rv32, T, S>,
    allow_suspended: bool,
) -> Result<ExecutionStatus, ExecuteError>
where
    F: PrimeField32,
    T: TracerState,
    S: SuspenderState,
{
    let mut io_state = build_io_state_borrowed(vm_state);
    let callbacks = build_callbacks(&mut io_state);
    unsafe {
        register_openvm_callbacks(compiled, &callbacks)?;
        extensions.register_host_callbacks(&compiled.lib)?;
        rv_execute(compiled, state.as_void_ptr())?;
    }

    let status = state.execution_status();
    let exit_code = state.result_code();
    match status {
        ExecutionStatus::Terminated if exit_code == 0 => {
            write_rv32_registers(vm_state, &state.regs);
            vm_state.set_pc(state.pc);
            Ok(status)
        }
        ExecutionStatus::Suspended if allow_suspended => {
            write_rv32_registers(vm_state, &state.regs);
            vm_state.set_pc(state.pc);
            Ok(status)
        }
        _ => Err(if status == ExecutionStatus::Terminated {
            ExecuteError::GuestExit(exit_code)
        } else {
            ExecuteError::ExecutionFailed(status as i32)
        }),
    }
}

// ── Public execute functions ─────────────────────────────────────────────────

/// Execute a VmExe using a compiled rvr shared library against `vm_state`.
///
/// If `num_insns` is `Some(n)`, the suspender is armed at `n` instructions and
/// a `Suspended` outcome is accepted as success; otherwise only `Terminated`
/// (with exit-code 0) succeeds.
pub fn execute<F: PrimeField32>(
    compiled: &RvrCompiled,
    extensions: &ExtensionRegistry<F>,
    vm_state: &mut VmState<F, GuestMemory>,
    num_insns: Option<u64>,
) -> Result<RvrPureResult, ExecuteError> {
    let pc = vm_state.pc();
    let initial_regs = read_rv32_registers(vm_state);

    let mut tracer_data = PureTracerData;
    let mut state = init_rvr_state(vm_state, pc);
    state.regs = initial_regs;
    state.tracer = TracerPtr(&mut tracer_data);
    if let Some(n) = num_insns {
        state.suspender.set_target(n);
    }

    let status = run_and_finalize(
        compiled,
        extensions,
        vm_state,
        &mut state,
        num_insns.is_some(),
    )
    .inspect_err(|e| eprintln!("rvr pure execution failed: {e}"))?;
    Ok(RvrPureResult {
        state,
        suspended: status == ExecutionStatus::Suspended,
    })
}

/// Execute a VmExe with metered cost tracking.
pub fn execute_metered_cost<F: PrimeField32>(
    compiled: &RvrCompiled,
    extensions: &ExtensionRegistry<F>,
    vm_state: &mut VmState<F, GuestMemory>,
    metered_cost_config: MeteredCostConfig,
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

    run_and_finalize(compiled, extensions, vm_state, &mut state, false)
        .inspect_err(|e| eprintln!("rvr metered-cost execution failed: {e}"))?;
    let cost = state.tracer.cost;
    Ok(RvrMeteredCostResult { state, cost })
}

/// Execute a VmExe with per-chip metered execution and segmentation.
pub fn execute_metered<F: PrimeField32>(
    compiled: &RvrCompiled,
    extensions: &ExtensionRegistry<F>,
    vm_state: &mut VmState<F, GuestMemory>,
    mut seg_state: SegmentationState,
) -> Result<SegmentationState, ExecuteError> {
    let pc = vm_state.pc();
    let initial_regs = read_rv32_registers(vm_state);

    let mut tracer_data = MeteredTracerData::default();
    let mut state = init_rvr_state_with_metered(vm_state, pc);
    state.regs = initial_regs;
    state.tracer = TracerPtr(&mut tracer_data);

    state.tracer.trace_heights = seg_state.trace_heights_ptr();
    state.tracer.mem_page_buf = seg_state.mem_page_buf_ptr();
    state.tracer.pv_page_buf = seg_state.pv_page_buf_ptr();
    state.tracer.deferral_page_buf = seg_state.deferral_page_buf_ptr();
    state.tracer.mem_page_buf_len = 0;
    state.tracer.pv_page_buf_len = 0;
    state.tracer.deferral_page_buf_len = 0;
    state.tracer.last_mem_page = NO_LAST_PAGE;
    state.tracer.check_counter = seg_state.segmentation_ctx.segment_check_insns as u32;
    state.tracer.on_check = Some(metered_periodic_check);
    state.tracer.seg_state = &mut seg_state as *mut SegmentationState as *mut c_void;

    run_and_finalize(compiled, extensions, vm_state, &mut state, false)
        .inspect_err(|e| eprintln!("rvr metered execution failed: {e}"))?;

    seg_state.on_termination(
        state.tracer.mem_page_buf_len,
        state.tracer.pv_page_buf_len,
        state.tracer.deferral_page_buf_len,
        state.tracer.check_counter,
    );
    Ok(seg_state)
}
