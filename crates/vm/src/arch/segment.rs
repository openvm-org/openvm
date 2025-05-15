use backtrace::Backtrace;
use openvm_instructions::{
    exe::FnBounds,
    instruction::{DebugInfo, Instruction},
};
use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig},
    keygen::types::LinearConstraint,
    p3_commit::PolynomialSpace,
    p3_field::PrimeField32,
    prover::types::{CommittedTraceData, ProofInput},
    utils::metrics_span,
    Chip,
};

use super::{
    execution_control::{
        E1ExecutionControl, ExecutionControl, MeteredExecutionControl, TracegenExecutionControl,
    },
    ExecutionError, GenerationError, SystemConfig, VmChipComplex, VmComplexTraceHeights, VmConfig,
};
#[cfg(feature = "bench-metrics")]
use crate::metrics::VmMetrics;
use crate::{
    arch::{instructions::*, InstructionExecutor},
    system::connector::DEFAULT_SUSPEND_EXIT_CODE,
};

pub struct VmSegmentExecutor<F, VC, Ctrl>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    Ctrl: ExecutionControl<F, VC>,
{
    pub chip_complex: VmChipComplex<F, VC::Executor, VC::Periphery>,
    /// Execution control for determining segmentation and stopping conditions
    pub ctrl: Ctrl,

    pub trace_height_constraints: Vec<LinearConstraint>,

    /// Air names for debug purposes only.
    pub(crate) air_names: Vec<String>,
    /// Metrics collected for this execution segment alone.
    #[cfg(feature = "bench-metrics")]
    pub metrics: VmMetrics,
}

impl<F, VC, Ctrl> VmSegmentExecutor<F, VC, Ctrl>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    Ctrl: ExecutionControl<F, VC>,
{
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(
        chip_complex: VmChipComplex<F, VC::Executor, VC::Periphery>,
        trace_height_constraints: Vec<LinearConstraint>,
        #[allow(unused_variables)] fn_bounds: FnBounds,
        ctrl: Ctrl,
    ) -> Self {
        let air_names = chip_complex.air_names();

        Self {
            chip_complex,
            ctrl,
            air_names,
            trace_height_constraints,
            #[cfg(feature = "bench-metrics")]
            metrics: VmMetrics {
                fn_bounds,
                ..Default::default()
            },
        }
    }

    pub fn system_config(&self) -> &SystemConfig {
        self.chip_complex.config()
    }

    pub fn set_override_trace_heights(&mut self, overridden_heights: VmComplexTraceHeights) {
        self.chip_complex
            .set_override_system_trace_heights(overridden_heights.system);
        self.chip_complex
            .set_override_inventory_trace_heights(overridden_heights.inventory);
    }

    /// Stopping is triggered by should_stop() or if VM is terminated
    pub fn execute_from_pc_with_ctx(
        &mut self,
        pc: u32,
        memory: Option<Mem>,
        ctx: Ctrl::Ctx,
    ) -> Result<ExecutionSegmentState<Mem, Ctrl::Ctx>, ExecutionError> {
        let mut prev_backtrace: Option<Backtrace> = None;

        // Call the pre-execution hook
        self.ctrl.on_segment_start(pc, &mut self.chip_complex);

        let mut state = ExecutionSegmentState::new_with_pc_and_ctx(pc, ctx);
        loop {
            // Fetch, decode and execute single instruction
            let terminated_exit_code = self.execute_instruction(&mut state, &mut prev_backtrace)?;

            if let Some(exit_code) = terminated_exit_code {
                state.exit_code = exit_code;
                state.is_terminated = true;
                self.ctrl
                    .on_terminate(state.pc, &mut self.chip_complex, exit_code);
                break;
            }
            if self.should_stop() {
                state.exit_code = DEFAULT_SUSPEND_EXIT_CODE;
                self.ctrl.on_segment_end(state.pc, &mut self.chip_complex);
                break;
            }
        }

        Ok(state)
    }

    /// Executes a single instruction and updates VM state
    // TODO(ayush): clean this up, separate to smaller functions
    fn execute_instruction(
        &mut self,
        state: &mut ExecutionSegmentState<Mem, Ctrl::Ctx>,
        prev_backtrace: &mut Option<Backtrace>,
    ) -> Result<Option<u32>, ExecutionError> {
        let pc = state.pc;
        let timestamp = self.chip_complex.memory_controller().timestamp();

        // Process an instruction and update VM state
        let (instruction, debug_info) = self.chip_complex.base.program_chip.get_instruction(pc)?;

        tracing::trace!("pc: {pc:#x} | time: {timestamp} | {:?}", instruction);

        let &Instruction { opcode, c, .. } = instruction;

        // Handle termination instruction
        if opcode == SystemOpcode::TERMINATE.global_opcode() {
            return Ok(Some(c.as_canonical_u32()));
        }

        // Extract debug info components
        #[allow(unused_variables)]
        let (dsl_instr, trace) = debug_info.as_ref().map_or(
            (None, None),
            |DebugInfo {
                 dsl_instruction,
                 trace,
             }| (Some(dsl_instruction.clone()), trace.as_ref()),
        );

        // Handle phantom instructions
        if opcode == SystemOpcode::PHANTOM.global_opcode() {
            let discriminant = c.as_canonical_u32() as u16;
            if let Some(phantom) = SysPhantom::from_repr(discriminant) {
                tracing::trace!("pc: {pc:#x} | system phantom: {phantom:?}");

                if phantom == SysPhantom::DebugPanic {
                    if let Some(mut backtrace) = prev_backtrace.take() {
                        backtrace.resolve();
                        eprintln!("openvm program failure; backtrace:\n{:?}", backtrace);
                    } else {
                        eprintln!("openvm program failure; no backtrace");
                    }
                    return Err(ExecutionError::Fail { pc });
                }

                #[cfg(feature = "bench-metrics")]
                {
                    let dsl_str = dsl_instr.clone().unwrap_or_else(|| "Default".to_string());
                    match phantom {
                        SysPhantom::CtStart => self.metrics.cycle_tracker.start(dsl_str),
                        SysPhantom::CtEnd => self.metrics.cycle_tracker.end(dsl_str),
                        _ => {}
                    }
                }
            }
        }

        // TODO(ayush): move to vm state?
        *prev_backtrace = trace.cloned();

        // Execute the instruction using the control implementation
        // TODO(AG): maybe avoid cloning the instruction?
        self.control
            .execute_instruction(state, &instruction.clone(), &mut self.chip_complex)?;

        // Update metrics if enabled
        #[cfg(feature = "bench-metrics")]
        {
            self.update_instruction_metrics(pc, opcode, dsl_instr);
        }

        Ok(None)
    }

    /// Returns bool of whether to switch to next segment or not.
    fn should_stop(&mut self) -> bool {
        if !self.system_config().continuation_enabled {
            return false;
        }

        // Check with the execution control policy
        self.ctrl.should_stop(&self.chip_complex)
    }

    // TODO(ayush): this is not relevant for e1/e2 execution
    /// Generate ProofInput to prove the segment. Should be called after ::execute
    pub fn generate_proof_input<SC: StarkGenericConfig>(
        #[allow(unused_mut)] mut self,
        cached_program: Option<CommittedTraceData<SC>>,
    ) -> Result<ProofInput<SC>, GenerationError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        metrics_span("trace_gen_time_ms", || {
            self.chip_complex.generate_proof_input(
                cached_program,
                &self.trace_height_constraints,
                #[cfg(feature = "bench-metrics")]
                &mut self.metrics,
            )
        })
    }

    #[cfg(feature = "bench-metrics")]
    #[allow(unused_variables)]
    pub fn update_instruction_metrics(
        &mut self,
        pc: u32,
        opcode: VmOpcode,
        dsl_instr: Option<String>,
    ) {
        self.metrics.cycle_count += 1;

        if self.system_config().profiling {
            let executor = self.chip_complex.inventory.get_executor(opcode).unwrap();
            let opcode_name = executor.get_opcode_name(opcode.as_usize());
            self.metrics.update_trace_cells(
                &self.air_names,
                self.chip_complex.current_trace_cells(),
                opcode_name,
                dsl_instr,
            );

            #[cfg(feature = "function-span")]
            self.metrics.update_current_fn(pc);
        }
    }
}

// E1 execution
pub type E1Ctx = ();
pub type E1VmSegmentExecutor<F, VC> = VmSegmentExecutor<F, VC, E1ExecutionControl>;

// E2 (metered) execution
#[derive(Default)]
pub struct MeteredCtx {
    pub trace_heights: Vec<usize>,
    pub total_trace_cells: usize,
    pub total_interactions: usize,
}

impl MeteredCtx {
    pub fn new_with_len(len: usize) -> Self {
        Self {
            trace_heights: vec![0; len],
            total_trace_cells: 0,
            total_interactions: 0,
        }
    }
}

pub type MeteredVmSegmentExecutor<'a, F, VC> =
    VmSegmentExecutor<F, VC, MeteredExecutionControl<'a>>;

// E3 (tracegen) execution
pub type TracegenCtx = ();
pub type TracegenVmSegmentExecutor<F, VC> = VmSegmentExecutor<F, VC, TracegenExecutionControl>;

pub struct ExecutionSegmentState<Mem, Ctx> {
    pub memory: Option<Mem>,
    pub pc: u32,
    // TODO(ayush): do we need both exit_code and is_terminated?
    pub exit_code: u32,
    pub is_terminated: bool,
    pub ctx: Ctx,
}

impl<Ctx> ExecutionSegmentState<Ctx> {
    pub fn new_with_pc_and_ctx(pc: u32, ctx: Ctx) -> Self {
        Self {
            memory: None,
            pc,
            ctx,
            exit_code: 0,
            is_terminated: false,
        }
    }
}
