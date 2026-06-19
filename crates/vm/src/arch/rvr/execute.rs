//! Load .so, bridge state, call rv_execute.
//!
//! Each execution path takes `&mut VmState<F, GuestMemory>` directly: the
//! transient `RvState` scratch struct aliases VmState's memory and registers,
//! and `OpenVmIoState` borrows VmState's `Streams<F>` and rng. There is no
//! separately-owned guest memory or stream conversion.

use std::ffi::c_void;

use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_lift::{ExtensionError, ExtensionRegistry};
use rvr_state::{ExecutionStatus, MemoryError, Rv64, RvState, SuspenderState, TracerState};

use super::{
    bridge::{
        deferral_memory_ptr, public_values_slice, read_rv64_registers, rv64_memory_ptr,
        write_rv64_registers,
    },
    compile::RvrCompiled,
    io::OpenVmIoState,
    metered::{
        metered_periodic_check, MeteredTracerData, RvrMeteredResult, SegmentationState,
        NO_LAST_PAGE,
    },
    metered_cost::{MeteredCostData, PureTracerData, RvrMeteredCostResult},
    pure::RvrPureResult,
    state::{
        init_rvr_state, init_rvr_state_with_metered, init_rvr_state_with_metered_cost, TracerPtr,
    },
};
use crate::{arch::VmState, system::memory::online::GuestMemory};

type RegisterIoCtxFn = unsafe extern "C" fn(*mut c_void);
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
    #[error("invalid metered context: {0}")]
    InvalidMeteredContext(String),
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn build_io_state_borrowed<'a, F: PrimeField32>(
    vm_state: &'a mut VmState<F, GuestMemory>,
) -> OpenVmIoState<'a, F> {
    let memory_ptr = rv64_memory_ptr(vm_state);
    let (deferral_memory, deferral_memory_len) =
        deferral_memory_ptr::<F>(&mut vm_state.memory.memory);
    let streams = &mut vm_state.streams;
    OpenVmIoState {
        input_stream: &mut streams.input_stream,
        hint_stream: &mut streams.hint_stream,
        rng: &mut vm_state.rng,
        memory_ptr,
        public_values: public_values_slice(&mut vm_state.memory.memory),
        deferral_memory,
        deferral_memory_len,
        deferrals: &mut streams.deferrals,
    }
}

/// # Safety
///
/// `compiled` must contain a valid rvr-compiled shared library exporting the
/// `register_openvm_io_ctx` symbol with the expected ABI. `io_state` must
/// remain valid for the lifetime of the subsequent `rv_execute` call.
unsafe fn register_openvm_io_ctx<F: PrimeField32>(
    compiled: &RvrCompiled,
    io_state: &mut OpenVmIoState<'_, F>,
) -> Result<(), ExecuteError> {
    let register_fn: RegisterIoCtxFn = unsafe {
        let sym = compiled
            .lib
            .get::<RegisterIoCtxFn>(b"register_openvm_io_ctx")
            .map_err(|e| ExecuteError::SymbolLookup(e.to_string()))?;
        *sym
    };
    unsafe { register_fn(io_state as *mut OpenVmIoState<'_, F> as *mut c_void) };
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
    state: &mut RvState<Rv64, T, S>,
    allow_suspended: bool,
) -> Result<ExecutionStatus, ExecuteError>
where
    F: PrimeField32,
    T: TracerState,
    S: SuspenderState,
{
    let mut io_state = build_io_state_borrowed(vm_state);
    unsafe {
        register_openvm_io_ctx(compiled, &mut io_state)?;
        extensions.register_host_callbacks(&compiled.lib)?;
        rv_execute(compiled, state.as_void_ptr())?;
    }

    let status = state.execution_status();
    let exit_code = state.result_code();
    match status {
        ExecutionStatus::Terminated if exit_code == 0 => {
            write_rv64_registers(vm_state, &state.regs);
            vm_state.set_pc(
                u32::try_from(state.pc).expect("PC must be within u32 range after C bounds check"),
            );
            Ok(status)
        }
        ExecutionStatus::Suspended if allow_suspended => {
            write_rv64_registers(vm_state, &state.regs);
            vm_state.set_pc(
                u32::try_from(state.pc).expect("PC must be within u32 range after C bounds check"),
            );
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
    let initial_regs = read_rv64_registers(vm_state);

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
    .inspect_err(|error| tracing::warn!(%error, "rvr pure execution failed"))?;
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
    widths: &[u64],
) -> Result<RvrMeteredCostResult, ExecuteError> {
    let pc = vm_state.pc();
    let initial_regs = read_rv64_registers(vm_state);

    let mut tracer_data = MeteredCostData::default();
    let mut state = init_rvr_state_with_metered_cost(vm_state, pc);
    state.regs = initial_regs;
    state.tracer = TracerPtr(&mut tracer_data);

    state.tracer.chip_widths = widths.as_ptr();
    state.tracer.cost = 0;

    run_and_finalize(compiled, extensions, vm_state, &mut state, false)
        .inspect_err(|error| tracing::warn!(%error, "rvr metered-cost execution failed"))?;
    let cost = state.tracer.cost;
    Ok(RvrMeteredCostResult { state, cost })
}

/// Execute a VmExe with per-chip metered execution and segmentation.
pub fn execute_metered<F: PrimeField32>(
    compiled: &RvrCompiled,
    extensions: &ExtensionRegistry<F>,
    vm_state: &mut VmState<F, GuestMemory>,
    seg_state: SegmentationState,
) -> Result<SegmentationState, ExecuteError> {
    execute_metered_impl(compiled, extensions, vm_state, seg_state, false).map(|result| {
        debug_assert!(!result.suspended);
        result.seg_state
    })
}

/// Execute a VmExe with per-chip metered execution until termination or a segment boundary.
pub fn execute_metered_segment_boundary<F: PrimeField32>(
    compiled: &RvrCompiled,
    extensions: &ExtensionRegistry<F>,
    vm_state: &mut VmState<F, GuestMemory>,
    seg_state: SegmentationState,
) -> Result<RvrMeteredResult, ExecuteError> {
    execute_metered_impl(compiled, extensions, vm_state, seg_state, true)
}

fn execute_metered_impl<F: PrimeField32>(
    compiled: &RvrCompiled,
    extensions: &ExtensionRegistry<F>,
    vm_state: &mut VmState<F, GuestMemory>,
    mut seg_state: SegmentationState,
    allow_suspended: bool,
) -> Result<RvrMeteredResult, ExecuteError> {
    debug_assert!(
        !allow_suspended || seg_state.suspend_on_segment(),
        "segment-boundary rvr execution requires MeteredCtx::suspend_on_segment"
    );

    let pc = vm_state.pc();
    let initial_regs = read_rv64_registers(vm_state);

    let mut tracer_data = MeteredTracerData::default();
    let mut state = init_rvr_state_with_metered(vm_state, pc);
    state.regs = initial_regs;
    state.tracer = TracerPtr(&mut tracer_data);

    let check_counter =
        u32::try_from(seg_state.segmentation_ctx.instrets_until_check).map_err(|_| {
            ExecuteError::InvalidMeteredContext(format!(
                "instrets_until_check {} exceeds rvr tracer u32 counter",
                seg_state.segmentation_ctx.instrets_until_check
            ))
        })?;
    let _ = u32::try_from(seg_state.segmentation_ctx.segment_check_insns()).map_err(|_| {
        ExecuteError::InvalidMeteredContext(format!(
            "segment_check_insns {} exceeds rvr tracer u32 counter",
            seg_state.segmentation_ctx.segment_check_insns()
        ))
    })?;

    state.tracer.trace_heights = seg_state.trace_heights_ptr();
    state.tracer.mem_page_buf = seg_state.mem_page_buf_ptr();
    state.tracer.pv_page_buf = seg_state.pv_page_buf_ptr();
    state.tracer.deferral_page_buf = seg_state.deferral_page_buf_ptr();
    state.tracer.mem_page_buf_len = 0;
    state.tracer.pv_page_buf_len = 0;
    state.tracer.deferral_page_buf_len = 0;
    state.tracer.last_mem_page = NO_LAST_PAGE;
    state.tracer.check_counter = check_counter;
    state.tracer.on_check = metered_periodic_check;
    state.tracer.seg_state = &mut seg_state as *mut SegmentationState as *mut c_void;

    let status = run_and_finalize(compiled, extensions, vm_state, &mut state, allow_suspended)
        .inspect_err(|error| tracing::warn!(%error, "rvr metered execution failed"))?;

    debug_assert!(matches!(
        status,
        ExecutionStatus::Terminated | ExecutionStatus::Suspended
    ));
    let terminated = status == ExecutionStatus::Terminated;
    if terminated {
        seg_state.on_termination(
            state.tracer.mem_page_buf_len,
            state.tracer.pv_page_buf_len,
            state.tracer.deferral_page_buf_len,
            state.tracer.check_counter,
        );
    } else {
        // The segment-boundary suspender exits before executing the triggering block.
        // The periodic check already flushed page buffers and initialized the next
        // segment; carry the bumped countdown forward for resume.
        seg_state.segmentation_ctx.instrets_until_check = state.tracer.check_counter as u64;
    }
    Ok(RvrMeteredResult {
        seg_state,
        suspended: !terminated,
        exit_code: terminated.then_some(state.result_code() as u32),
    })
}
