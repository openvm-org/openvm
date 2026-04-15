//! Load .so, bridge state, call rv_execute.

use std::collections::VecDeque;
use std::ffi::c_void;

use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rvr_state::GuardedMemory;

use crate::compile::RvrCompiled;
use crate::io::{
    convert_input_stream, host_deferral_call_lookup, host_deferral_output_lookup, host_hint_buffer,
    host_hint_input, host_hint_random, host_hint_storew, host_hint_stream_set, host_print_str,
    host_reveal, DeferralData, OpenVmHostCallbacks, OpenVmIoState,
};
use crate::metered::{
    metered_periodic_check, MeteredConfig, MeteredTracer, MeteredTracerData, RvrMeteredResult,
    SegmentationState, NO_LAST_PAGE,
};
use crate::metered_cost::{
    prepare_metered_cost, MeteredCostConfig, MeteredCostData, MeteredCostMeter, PureTracer,
    PureTracerData,
};
use crate::state::{
    init_rvr_state, init_rvr_state_with_metered, init_rvr_state_with_metered_cost,
    MeteredCostState, MeteredState, PureState,
};

/// Result of executing via rvr.
pub struct RvrExecutionResult {
    pub state: PureState,
    pub memory: GuardedMemory,
    pub public_values: Vec<u8>,
}

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
    MemoryAlloc(#[from] rvr_state::MemoryError),
}

/// Result of executing with an instruction limit via rvr.
pub struct RvrLimitedResult {
    pub state: PureState,
    pub memory: GuardedMemory,
    pub instret: u64,
    pub suspended: bool,
}

/// Result of metered cost execution via rvr.
pub struct RvrMeteredCostResult {
    pub state: MeteredCostState,
    pub memory: GuardedMemory,
    pub cost: u64,
    pub instret: u64,
    pub suspended: bool,
}

/// Result of metered cost execution with an instruction limit via rvr.
pub struct RvrMeteredCostLimitedResult {
    pub state: MeteredCostState,
    pub memory: GuardedMemory,
    pub cost: u64,
    pub instret: u64,
    pub suspended: bool,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecuteOutcome {
    Running,
    Terminated,
    Suspended,
}

pub fn build_io_state(
    input_stream: VecDeque<Vec<u8>>,
    memory: *mut u8,
    deferral: DeferralData,
) -> OpenVmIoState {
    OpenVmIoState {
        input_stream,
        hint_stream: Vec::new(),
        hint_pos: 0,
        public_values: Vec::new(),
        memory,
        rng: StdRng::seed_from_u64(0),
        deferral,
    }
}

pub fn build_callbacks(io_state: &mut OpenVmIoState) -> OpenVmHostCallbacks {
    OpenVmHostCallbacks {
        ctx: io_state as *mut OpenVmIoState as *mut c_void,
        hint_input: host_hint_input,
        print_str: host_print_str,
        hint_random: host_hint_random,
        hint_storew: host_hint_storew,
        hint_buffer: host_hint_buffer,
        reveal: host_reveal,
        hint_stream_set: host_hint_stream_set,
        deferral_call_lookup: host_deferral_call_lookup,
        deferral_output_lookup: host_deferral_output_lookup,
    }
}

/// # Safety
///
/// - `compiled` must contain a valid rvr-compiled shared library with the
///   expected ABI (`register_openvm_callbacks` and `rv_execute` symbols).
/// - `state_ptr` must point to a valid, mutable RV32 state struct whose
///   tracer variant matches the one compiled into the shared library.
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

fn outcome_from_pure_state(state: &PureState) -> ExecuteOutcome {
    if state.is_terminated() {
        ExecuteOutcome::Terminated
    } else if state.is_suspended() {
        ExecuteOutcome::Suspended
    } else {
        ExecuteOutcome::Running
    }
}

fn outcome_from_metered_cost_state(state: &MeteredCostState) -> ExecuteOutcome {
    if state.is_terminated() {
        ExecuteOutcome::Terminated
    } else if state.is_suspended() {
        ExecuteOutcome::Suspended
    } else {
        ExecuteOutcome::Running
    }
}

fn outcome_from_metered_state(state: &MeteredState) -> ExecuteOutcome {
    if state.is_terminated() {
        ExecuteOutcome::Terminated
    } else if state.is_suspended() {
        ExecuteOutcome::Suspended
    } else {
        ExecuteOutcome::Running
    }
}

// ── Public execute functions ─────────────────────────────────────────────────

/// Execute a VmExe using a compiled rvr shared library.
pub fn execute<F: PrimeField32>(
    compiled: &RvrCompiled,
    exe: &VmExe<F>,
    input_stream: VecDeque<Vec<F>>,
    deferral: DeferralData,
) -> Result<RvrExecutionResult, ExecuteError> {
    let mut memory = GuardedMemory::new(openvm_platform::memory::MEM_SIZE)?;
    let mut tracer_data = PureTracerData;
    let mut state = init_rvr_state(exe, &mut memory);
    state.tracer = PureTracer(&mut tracer_data);

    let mut io_state = build_io_state(
        convert_input_stream(&input_stream),
        memory.as_ptr(),
        deferral,
    );
    let callbacks = build_callbacks(&mut io_state);
    // SAFETY: state pointer is valid and matches the tracer variant.
    unsafe { register_and_execute(compiled, &callbacks, state.as_void_ptr()) }?;
    let outcome = outcome_from_pure_state(&state);

    if outcome == ExecuteOutcome::Terminated && state.is_terminated() && state.result_code() == 0 {
        Ok(RvrExecutionResult {
            state,
            memory,
            public_values: io_state.public_values,
        })
    } else {
        Err(execution_error(
            "execution failed",
            outcome,
            state.pc,
            state.instret,
            state.result_code(),
        ))
    }
}

/// Execute a VmExe with metered cost tracking, tracking per-chip trace costs.
pub fn execute_metered_cost<F: PrimeField32>(
    compiled: &RvrCompiled,
    exe: &VmExe<F>,
    input_stream: VecDeque<Vec<F>>,
    metered_cost_config: MeteredCostConfig,
    deferral: DeferralData,
) -> Result<RvrMeteredCostResult, ExecuteError> {
    let mut memory = GuardedMemory::new(openvm_platform::memory::MEM_SIZE)?;
    let mut tracer_data = MeteredCostData::default();
    let mut state = init_rvr_state_with_metered_cost(exe, &mut memory);
    state.tracer = MeteredCostMeter(&mut tracer_data);

    let widths_u64 = prepare_metered_cost(&metered_cost_config);

    state.tracer.chip_widths = widths_u64.as_ptr();
    state.tracer.cost = 0;

    let mut io_state = build_io_state(
        convert_input_stream(&input_stream),
        memory.as_ptr(),
        deferral,
    );
    let callbacks = build_callbacks(&mut io_state);
    // SAFETY: state pointer is valid and matches the tracer variant.
    unsafe { register_and_execute(compiled, &callbacks, state.as_void_ptr()) }?;
    let outcome = outcome_from_metered_cost_state(&state);

    let cost = state.tracer.cost;
    let instret = state.instret;

    if outcome == ExecuteOutcome::Terminated && state.is_terminated() && state.result_code() == 0 {
        Ok(RvrMeteredCostResult {
            state,
            memory,
            cost,
            instret,
            suspended: false,
        })
    } else {
        Err(execution_error(
            "metered-cost execution failed",
            outcome,
            state.pc,
            instret,
            state.result_code(),
        ))
    }
}

/// Execute a VmExe with an instruction limit.
///
/// Suspends via target_instret when instret reaches the limit.
pub fn execute_with_limit<F: PrimeField32>(
    compiled: &RvrCompiled,
    exe: &VmExe<F>,
    input_stream: VecDeque<Vec<F>>,
    instruction_limit: u64,
    deferral: DeferralData,
) -> Result<RvrLimitedResult, ExecuteError> {
    let mut memory = GuardedMemory::new(openvm_platform::memory::MEM_SIZE)?;
    let mut tracer_data = PureTracerData;
    let mut state = init_rvr_state(exe, &mut memory);
    state.tracer = PureTracer(&mut tracer_data);
    state.suspender.set_target(instruction_limit);

    let mut io_state = build_io_state(
        convert_input_stream(&input_stream),
        memory.as_ptr(),
        deferral,
    );
    let callbacks = build_callbacks(&mut io_state);
    // SAFETY: state pointer is valid and matches the tracer variant.
    unsafe { register_and_execute(compiled, &callbacks, state.as_void_ptr()) }?;
    let outcome = outcome_from_pure_state(&state);

    let instret = state.instret;

    match outcome {
        ExecuteOutcome::Suspended => Ok(RvrLimitedResult {
            state,
            memory,
            instret,
            suspended: true,
        }),
        ExecuteOutcome::Terminated if state.is_terminated() && state.result_code() == 0 => {
            Ok(RvrLimitedResult {
                state,
                memory,
                instret,
                suspended: false,
            })
        }
        _ => Err(execution_error(
            "limited execution failed",
            outcome,
            state.pc,
            instret,
            state.result_code(),
        )),
    }
}

/// Execute a VmExe with metered cost and an instruction limit.
pub fn execute_metered_cost_with_limit<F: PrimeField32>(
    compiled: &RvrCompiled,
    exe: &VmExe<F>,
    input_stream: VecDeque<Vec<F>>,
    metered_cost_config: MeteredCostConfig,
    instruction_limit: u64,
    deferral: DeferralData,
) -> Result<RvrMeteredCostLimitedResult, ExecuteError> {
    let mut memory = GuardedMemory::new(openvm_platform::memory::MEM_SIZE)?;
    let mut tracer_data = MeteredCostData::default();
    let mut state = init_rvr_state_with_metered_cost(exe, &mut memory);
    state.tracer = MeteredCostMeter(&mut tracer_data);

    let widths_u64 = prepare_metered_cost(&metered_cost_config);

    state.tracer.chip_widths = widths_u64.as_ptr();
    state.tracer.cost = 0;
    state.suspender.set_target(instruction_limit);

    let mut io_state = build_io_state(
        convert_input_stream(&input_stream),
        memory.as_ptr(),
        deferral,
    );
    let callbacks = build_callbacks(&mut io_state);
    // SAFETY: state pointer is valid and matches the tracer variant.
    unsafe { register_and_execute(compiled, &callbacks, state.as_void_ptr()) }?;
    let outcome = outcome_from_metered_cost_state(&state);

    let cost = state.tracer.cost;
    let instret = state.instret;

    match outcome {
        ExecuteOutcome::Suspended => Ok(RvrMeteredCostLimitedResult {
            state,
            memory,
            cost,
            instret,
            suspended: true,
        }),
        ExecuteOutcome::Terminated if state.is_terminated() && state.result_code() == 0 => {
            Ok(RvrMeteredCostLimitedResult {
                state,
                memory,
                cost,
                instret,
                suspended: false,
            })
        }
        _ => Err(execution_error(
            "limited metered-cost execution failed",
            outcome,
            state.pc,
            instret,
            state.result_code(),
        )),
    }
}

/// Execute a VmExe with per-chip metered execution and segmentation.
pub fn execute_metered<F: PrimeField32>(
    compiled: &RvrCompiled,
    exe: &VmExe<F>,
    input_stream: VecDeque<Vec<F>>,
    trace_config: MeteredConfig,
    deferral: DeferralData,
) -> Result<RvrMeteredResult, ExecuteError> {
    let mut memory = GuardedMemory::new(openvm_platform::memory::MEM_SIZE)?;
    let mut tracer_data = MeteredTracerData::default();
    let mut state = init_rvr_state_with_metered(exe, &mut memory);
    state.tracer = MeteredTracer(&mut tracer_data);

    let mut seg_state = SegmentationState::new(trace_config);

    // Wire up the C tracer fields
    state.tracer.trace_heights = seg_state.trace_heights_ptr();
    state.tracer.mem_page_buf = seg_state.mem_page_buf_ptr();
    state.tracer.pv_page_buf = seg_state.pv_page_buf_ptr();
    state.tracer.deferral_page_buf = seg_state.deferral_page_buf_ptr();
    state.tracer.mem_page_buf_len = 0;
    state.tracer.pv_page_buf_len = 0;
    state.tracer.deferral_page_buf_len = 0;
    state.tracer.last_mem_page = NO_LAST_PAGE;

    // Wire up inline callback for periodic segmentation checks.
    state.tracer.check_counter = seg_state.config().segment_check_insns as u32;
    state.tracer.on_check = Some(metered_periodic_check);
    state.tracer.seg_state = &mut seg_state as *mut SegmentationState as *mut c_void;

    let mut io_state = build_io_state(
        convert_input_stream(&input_stream),
        memory.as_ptr(),
        deferral,
    );
    let callbacks = build_callbacks(&mut io_state);
    // SAFETY: state pointer is valid and matches the tracer variant.
    unsafe { register_and_execute(compiled, &callbacks, state.as_void_ptr()) }?;
    let outcome = outcome_from_metered_state(&state);

    if outcome == ExecuteOutcome::Terminated && state.is_terminated() && state.result_code() == 0 {
        seg_state.on_termination(
            state.tracer.mem_page_buf_len,
            state.tracer.pv_page_buf_len,
            state.tracer.deferral_page_buf_len,
            state.tracer.check_counter,
        );
    } else {
        return Err(execution_error(
            "metered execution failed",
            outcome,
            state.pc,
            state.instret,
            state.result_code(),
        ));
    }

    Ok(seg_state.into_result())
}
