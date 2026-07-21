//! Load .so, bridge state, call rv_execute.
//!
//! Each execution path takes `&mut VmState<GuestMemory>` directly: the
//! transient `RvState` scratch struct aliases VmState's memory and registers,
//! and `OpenVmIoState` borrows VmState's `Streams` and rng. There is no
//! separately-owned guest memory or stream conversion.

use std::ffi::c_void;

use rvr_openvm::RvrExecutionKind;
use rvr_openvm_lift::{ExtensionError, RvrRuntimeExtension};
use rvr_state::{ExecutionStatus, InstretTrackingState, RvState};

use super::{
    bridge::{
        deferral_memory_ptr, public_values_slice, read_rv64_registers, rv64_memory_ptr,
        write_rv64_registers,
    },
    compile::RvrCompiled,
    io::{host_hint_stream_set, OpenVmIoState},
    metered::{metered_periodic_check, RvrMeteredExecutionOutcome, SegmentationState},
    metered_cost::RvrMeteredCostResult,
    state::{
        init_rvr_state, MeteredCostRvState, MeteredRvState, PureRvState,
        PureWithInstretTrackingRvState,
    },
};
use crate::{arch::VmState, system::memory::online::GuestMemory};

type RegisterIoCtxFn = unsafe extern "C" fn(*mut c_void);
type RegisterHintStreamSetFn =
    unsafe extern "C" fn(unsafe extern "C" fn(*mut c_void, *const u8, u64));
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
    #[error("extension host callback registration failed: {0}")]
    ExtensionRegistration(#[from] ExtensionError),
    #[error("invalid metered context: {0}")]
    InvalidMeteredContext(String),
    #[error("RVR execution kind mismatch: expected {expected}, found {found:?}")]
    ExecutionKindMismatch {
        expected: &'static str,
        found: RvrExecutionKind,
    },
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn build_io_state_borrowed(vm_state: &mut VmState<GuestMemory>) -> OpenVmIoState<'_> {
    let memory_ptr = rv64_memory_ptr(vm_state);
    let (deferral_memory, deferral_memory_len_bytes) =
        deferral_memory_ptr(&mut vm_state.memory.memory);
    let streams = &mut vm_state.streams;
    OpenVmIoState {
        input_stream: &mut streams.input_stream,
        hint_stream: &mut streams.hint_stream,
        rng: &mut vm_state.rng,
        memory_ptr,
        public_values: public_values_slice(&mut vm_state.memory.memory),
        deferral_memory,
        deferral_memory_len_bytes,
        deferrals: &mut streams.deferrals,
    }
}

fn require_execution_kind(
    compiled: &RvrCompiled,
    expected: &'static str,
    allowed: &[RvrExecutionKind],
) -> Result<(), ExecuteError> {
    let found = compiled.execution_kind();
    if allowed.contains(&found) {
        Ok(())
    } else {
        Err(ExecuteError::ExecutionKindMismatch { expected, found })
    }
}

fn require_num_airs(
    compiled: &RvrCompiled,
    actual: usize,
    actual_name: &str,
) -> Result<(), ExecuteError> {
    let expected = compiled.num_airs().ok_or_else(|| {
        ExecuteError::InvalidMeteredContext(
            "compiled metered artifact does not declare its AIR count".to_string(),
        )
    })?;
    if actual == expected as usize {
        Ok(())
    } else {
        Err(ExecuteError::InvalidMeteredContext(format!(
            "{actual_name} has {actual} entries, but the compiled artifact expects {expected} AIRs"
        )))
    }
}

/// # Safety
///
/// `compiled` must contain a valid rvr-compiled shared library exporting both
/// `register_openvm_io_ctx` and `register_hint_stream_set_fn` with the expected
/// ABIs. `io_state` must remain valid for the lifetime of the subsequent
/// `rv_execute` call.
unsafe fn register_openvm_io_ctx(
    compiled: &RvrCompiled,
    io_state: &mut OpenVmIoState<'_>,
) -> Result<(), ExecuteError> {
    let register_fn: RegisterIoCtxFn = unsafe {
        let sym = compiled
            .lib
            .get::<RegisterIoCtxFn>(b"register_openvm_io_ctx")
            .map_err(|e| ExecuteError::SymbolLookup(e.to_string()))?;
        *sym
    };
    unsafe { register_fn(io_state as *mut OpenVmIoState<'_> as *mut c_void) };

    let register_hint_fn: RegisterHintStreamSetFn = unsafe {
        let sym = compiled
            .lib
            .get::<RegisterHintStreamSetFn>(b"register_hint_stream_set_fn")
            .map_err(|e| ExecuteError::SymbolLookup(e.to_string()))?;
        *sym
    };
    unsafe { register_hint_fn(host_hint_stream_set) };

    Ok(())
}

/// # Safety
///
/// - `compiled` must contain a valid rvr-compiled shared library exporting the `rv_execute` symbol.
/// - `state_ptr` must point to the `RvState` layout selected by the compiled artifact.
unsafe fn rv_execute(compiled: &RvrCompiled, state_ptr: *mut c_void) -> Result<(), ExecuteError> {
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
/// `allow_suspended` permits `Suspended` as a successful outcome for callers
/// that stop at execution boundaries.
fn run_and_finalize<ModeState>(
    compiled: &RvrCompiled,
    runtime_hooks: &[Box<dyn RvrRuntimeExtension>],
    vm_state: &mut VmState<GuestMemory>,
    state: &mut RvState<ModeState>,
    allow_suspended: bool,
) -> Result<ExecutionStatus, ExecuteError> {
    let mut io_state = build_io_state_borrowed(vm_state);
    unsafe {
        register_openvm_io_ctx(compiled, &mut io_state)?;
        for hook in runtime_hooks {
            hook.register_host_callbacks(&compiled.lib)?;
        }
        rv_execute(compiled, state.as_void_ptr())?;
    }

    let status = state.execution_status();
    let exit_code = state.exit_code();
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

pub(super) struct TrackedExecutionResult {
    pub retired: u64,
    pub status: ExecutionStatus,
}

/// Execute an untracked pure artifact until termination.
pub(super) fn execute_pure(
    compiled: &RvrCompiled,
    runtime_hooks: &[Box<dyn RvrRuntimeExtension>],
    vm_state: &mut VmState<GuestMemory>,
) -> Result<(), ExecuteError> {
    require_execution_kind(compiled, "Pure", &[RvrExecutionKind::Pure])?;
    let pc = vm_state.pc();
    let initial_regs = read_rv64_registers(vm_state);
    let mut state: PureRvState = init_rvr_state(vm_state, pc);
    state.regs = initial_regs;
    run_and_finalize(compiled, runtime_hooks, vm_state, &mut state, false)
        .inspect_err(|error| tracing::warn!(%error, "rvr pure execution failed"))?;
    Ok(())
}

/// Execute an instret-tracking pure artifact until termination.
pub(super) fn execute_pure_with_instret_tracking(
    compiled: &RvrCompiled,
    runtime_hooks: &[Box<dyn RvrRuntimeExtension>],
    vm_state: &mut VmState<GuestMemory>,
) -> Result<u64, ExecuteError> {
    let result = execute_pure_with_instret_tracking_impl(
        compiled,
        runtime_hooks,
        vm_state,
        InstretTrackingState::unlimited(),
        false,
    )?;
    debug_assert_eq!(result.status, ExecutionStatus::Terminated);
    Ok(result.retired)
}

/// Execute an instret-tracking pure artifact up to an instruction boundary.
pub(super) fn execute_pure_with_instret_limit(
    compiled: &RvrCompiled,
    runtime_hooks: &[Box<dyn RvrRuntimeExtension>],
    vm_state: &mut VmState<GuestMemory>,
    num_insns: u64,
) -> Result<TrackedExecutionResult, ExecuteError> {
    execute_pure_with_instret_tracking_impl(
        compiled,
        runtime_hooks,
        vm_state,
        InstretTrackingState::with_limit(num_insns),
        true,
    )
}

fn execute_pure_with_instret_tracking_impl(
    compiled: &RvrCompiled,
    runtime_hooks: &[Box<dyn RvrRuntimeExtension>],
    vm_state: &mut VmState<GuestMemory>,
    tracking: InstretTrackingState,
    allow_suspended: bool,
) -> Result<TrackedExecutionResult, ExecuteError> {
    require_execution_kind(
        compiled,
        "PureWithInstretTracking",
        &[RvrExecutionKind::PureWithInstretTracking],
    )?;
    let pc = vm_state.pc();
    let mut state: PureWithInstretTrackingRvState = init_rvr_state(vm_state, pc);
    state.regs = read_rv64_registers(vm_state);
    state.mode_state = tracking;
    let status = run_and_finalize(
        compiled,
        runtime_hooks,
        vm_state,
        &mut state,
        allow_suspended,
    )
    .inspect_err(|error| tracing::warn!(%error, "rvr tracked pure execution failed"))?;
    Ok(TrackedExecutionResult {
        retired: state.mode_state.retired,
        status,
    })
}

/// Execute a VmExe with metered cost tracking.
pub(super) fn execute_metered_cost(
    compiled: &RvrCompiled,
    runtime_hooks: &[Box<dyn RvrRuntimeExtension>],
    vm_state: &mut VmState<GuestMemory>,
    widths: &[u64],
) -> Result<RvrMeteredCostResult, ExecuteError> {
    require_execution_kind(compiled, "MeteredCost", &[RvrExecutionKind::MeteredCost])?;
    require_num_airs(compiled, widths.len(), "chip-width table")?;
    let pc = vm_state.pc();
    let initial_regs = read_rv64_registers(vm_state);

    let mut state: MeteredCostRvState = init_rvr_state(vm_state, pc);
    state.regs = initial_regs;
    state.mode_state.chip_widths = widths.as_ptr();

    run_and_finalize(compiled, runtime_hooks, vm_state, &mut state, false)
        .inspect_err(|error| tracing::warn!(%error, "rvr metered-cost execution failed"))?;
    Ok(RvrMeteredCostResult {
        instret: state.mode_state.instret,
        cost: state.mode_state.cost,
    })
}

/// Execute a VmExe with per-chip metered execution and segmentation.
pub(super) fn execute_metered(
    compiled: &RvrCompiled,
    runtime_hooks: &[Box<dyn RvrRuntimeExtension>],
    vm_state: &mut VmState<GuestMemory>,
    seg_state: SegmentationState,
) -> Result<SegmentationState, ExecuteError> {
    require_execution_kind(compiled, "Metered", &[RvrExecutionKind::Metered])?;
    match execute_metered_impl(compiled, runtime_hooks, vm_state, seg_state, false)? {
        RvrMeteredExecutionOutcome::Terminated(state) => Ok(state),
        RvrMeteredExecutionOutcome::Suspended(_) => {
            unreachable!("unbounded metered execution cannot suspend")
        }
    }
}

/// Execute a VmExe with per-chip metered execution until termination or a segment boundary.
pub(super) fn execute_metered_segment_boundary(
    compiled: &RvrCompiled,
    runtime_hooks: &[Box<dyn RvrRuntimeExtension>],
    vm_state: &mut VmState<GuestMemory>,
    seg_state: SegmentationState,
) -> Result<RvrMeteredExecutionOutcome, ExecuteError> {
    require_execution_kind(
        compiled,
        "MeteredSegment",
        &[RvrExecutionKind::MeteredSegment],
    )?;
    execute_metered_impl(compiled, runtime_hooks, vm_state, seg_state, true)
}

fn execute_metered_impl(
    compiled: &RvrCompiled,
    runtime_hooks: &[Box<dyn RvrRuntimeExtension>],
    vm_state: &mut VmState<GuestMemory>,
    mut seg_state: SegmentationState,
    allow_suspended: bool,
) -> Result<RvrMeteredExecutionOutcome, ExecuteError> {
    debug_assert!(
        !allow_suspended || seg_state.suspend_on_segment(),
        "segment-boundary rvr execution requires MeteredCtx::suspend_on_segment"
    );
    require_num_airs(
        compiled,
        seg_state.ctx.trace_heights.len(),
        "trace-height storage",
    )?;

    let pc = vm_state.pc();
    let initial_regs = read_rv64_registers(vm_state);

    let mut state: MeteredRvState = init_rvr_state(vm_state, pc);
    state.regs = initial_regs;

    let check_counter = u32::try_from(seg_state.ctx.segmentation_ctx.instrets_until_check)
        .map_err(|_| {
            ExecuteError::InvalidMeteredContext(format!(
                "instrets_until_check {} exceeds rvr metering u32 counter",
                seg_state.ctx.segmentation_ctx.instrets_until_check
            ))
        })?;
    state.mode_state.trace_heights = seg_state.trace_heights_ptr();
    state.mode_state.mem_page_buf = seg_state.mem_page_buf_ptr();
    state.mode_state.pv_page_buf = seg_state.pv_page_buf_ptr();
    state.mode_state.deferral_page_buf = seg_state.deferral_page_buf_ptr();
    state.mode_state.check_counter = check_counter;
    state.mode_state.on_check = metered_periodic_check;
    state.mode_state.seg_state = &mut seg_state;

    let status = run_and_finalize(
        compiled,
        runtime_hooks,
        vm_state,
        &mut state,
        allow_suspended,
    )
    .inspect_err(|error| tracing::warn!(%error, "rvr metered execution failed"))?;

    debug_assert!(matches!(
        status,
        ExecutionStatus::Terminated | ExecutionStatus::Suspended
    ));
    let terminated = status == ExecutionStatus::Terminated;
    if terminated {
        seg_state.on_termination(
            state.mode_state.mem_page_buf_len,
            state.mode_state.pv_page_buf_len,
            state.mode_state.deferral_page_buf_len,
            state.mode_state.check_counter,
        );
    } else {
        // The segment boundary exits before executing the triggering block.
        // The periodic check already flushed page buffers and initialized the next
        // segment; carry the bumped countdown forward for resume.
        seg_state.ctx.segmentation_ctx.instrets_until_check = state.mode_state.check_counter as u64;
    }
    Ok(if terminated {
        RvrMeteredExecutionOutcome::Terminated(seg_state)
    } else {
        RvrMeteredExecutionOutcome::Suspended(seg_state)
    })
}
