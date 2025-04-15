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
    execution_control::{ExecutionControl, TracegenExecutionControl},
    ExecutionError, GenerationError, SystemConfig, VmChipComplex, VmComplexTraceHeights, VmConfig,
};
#[cfg(feature = "bench-metrics")]
use crate::metrics::VmMetrics;
use crate::{
    arch::{instructions::*, ExecutionState, InstructionExecutor},
    system::memory::{online::GuestMemory, paged_vec::PAGE_SIZE, AddressMap, MemoryImage},
};

/// Represents the state of the VM during execution
pub struct VmExecutionState<'a, Mem, Ctx>
where
    Mem: GuestMemory,
{
    /// Program counter - current instruction address
    pub pc: u32,
    /// Whether execution has terminated
    // TODO: see if it can be removed
    pub terminated: bool,
    /// Guest memory interface
    pub memory: &'a Mem,
    /// Host-specific execution context
    pub ctx: Ctx,
}

impl<'a, Mem, Ctx> VmExecutionState<'a, Mem, Ctx>
where
    Mem: GuestMemory,
{
    /// Creates a new VM execution state with the given parameters
    pub fn new(pc: u32, memory: &'a Mem, ctx: Ctx) -> Self {
        Self {
            pc,
            terminated: false,
            memory,
            ctx,
        }
    }
}

pub struct VmSegmentExecutor<F, VC, Mem, Ctx, Ctrl>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    Mem: GuestMemory,
    Ctrl: ExecutionControl<Mem = Mem, Ctx = Ctx>,
{
    pub chip_complex: VmChipComplex<F, VC::Executor, VC::Periphery>,
    /// Execution control for determining segmentation and stopping conditions
    pub control: Ctrl,

    pub trace_height_constraints: Vec<LinearConstraint>,

    /// Metrics collected for this execution segment alone.
    #[cfg(feature = "bench-metrics")]
    pub metrics: VmMetrics,
}

pub struct TracegenCtx {
    pub timestamp: u32,
}

impl TracegenCtx {
    pub fn new(timestamp: u32) -> Self {
        Self { timestamp }
    }
}

pub type TracegenVmExecutionState<'a> = VmExecutionState<'a, AddressMap<PAGE_SIZE>, TracegenCtx>;
pub type TracegenVmSegmentExecutor<F: PrimeField32, VC: VmConfig<F>> =
    VmSegmentExecutor<F, VC, AddressMap<PAGE_SIZE>, TracegenCtx, TracegenExecutionControl>;

impl<F, VC, Mem, Ctx, Ctrl> VmSegmentExecutor<F, VC, Mem, Ctx, Ctrl>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    Mem: GuestMemory,
    Ctrl: ExecutionControl<Mem = Mem, Ctx = Ctx>,
{
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(
        chip_complex: VmChipComplex<F, VC::Executor, VC::Periphery>,
        control: Ctrl,
        trace_height_constraints: Vec<LinearConstraint>,
        #[allow(unused_variables)] fn_bounds: FnBounds,
    ) -> Self {
        Self {
            chip_complex,
            control,
            trace_height_constraints,
            #[cfg(feature = "bench-metrics")]
            metrics: VmMetrics {
                fn_bounds,
                ..Default::default()
            },
        }
    }

    /// Stopping is triggered by should_stop() or if VM is terminated
    pub fn execute_from_state<'a>(
        &mut self,
        vm_state: &mut VmExecutionState<'a, Mem, Ctx>,
    ) -> Result<(), ExecutionError> {
        let mut prev_backtrace: Option<Backtrace> = None;

        // Call the pre-execution hook
        self.control.on_segment_start(vm_state);

        while !vm_state.terminated && !self.should_stop(vm_state) {
            // Fetch, decode and execute single instruction
            self.execute_instruction(vm_state, &mut prev_backtrace)?;
        }

        // Call the post-execution hook
        self.control.on_segment_end(vm_state);

        Ok(())
    }

    /// Executes a single instruction and updates VM state
    // TODO: clean this up, separate to smaller functions
    fn execute_instruction<'a>(
        &mut self,
        vm_state: &mut VmExecutionState<'a, Mem, Ctx>,
        prev_backtrace: &mut Option<Backtrace>,
    ) -> Result<(), ExecutionError> {
        let pc = vm_state.pc;
        let timestamp = self.chip_complex.memory_controller().timestamp();

        // Process an instruction and update VM state
        let (instruction, debug_info) = self.chip_complex.base.program_chip.get_instruction(pc)?;

        tracing::trace!("pc: {pc:#x} | time: {timestamp} | {:?}", instruction);

        // Extract debug info components
        #[allow(unused_variables)]
        let (dsl_instr, trace) = debug_info.as_ref().map_or(
            (None, None),
            |DebugInfo {
                 dsl_instruction,
                 trace,
             }| (Some(dsl_instruction), trace.as_ref()),
        );

        let &Instruction { opcode, c, .. } = instruction;

        // Handle termination instruction
        if opcode == SystemOpcode::TERMINATE.global_opcode() {
            self.chip_complex.connector_chip_mut().end(
                ExecutionState::new(pc, timestamp),
                Some(c.as_canonical_u32()),
            );
            vm_state.terminated = true;
            return Ok(());
        }

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
                    let dsl_str = dsl_instr.cloned().unwrap_or_else(|| "Default".to_string());
                    match phantom {
                        SysPhantom::CtStart => self.metrics.cycle_tracker.start(dsl_str),
                        SysPhantom::CtEnd => self.metrics.cycle_tracker.end(dsl_str),
                        _ => {}
                    }
                }
            }
        }

        // TODO: move to vm state?
        *prev_backtrace = trace.cloned();

        // Execute the instruction using the control implementation
        self.control.execute_instruction(vm_state, instruction)?;

        // Update metrics if enabled
        #[cfg(feature = "bench-metrics")]
        {
            let dsl_instr_clone = dsl_instr.cloned();
            self.update_instruction_metrics(pc, opcode, dsl_instr_clone);
        }

        Ok(())
    }

    /// Returns bool of whether to switch to next segment or not.
    fn should_stop<'a>(&mut self, vm_state: &VmExecutionState<'a, Mem, Ctx>) -> bool {
        if !self.system_config().continuation_enabled {
            return false;
        }

        // Check with the execution control policy
        self.control.should_stop(vm_state)
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

    // TODO: not sure what to do of these
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
            let air_names = self.chip_complex.air_names();
            self.metrics.update_trace_cells(
                &air_names,
                self.chip_complex.current_trace_cells(),
                opcode_name,
                dsl_instr,
            );

            #[cfg(feature = "function-span")]
            self.metrics.update_current_fn(pc);
        }
    }
}

// TODO: Only use AddressMap/PagedVec
//       Add metric counter for gas metering, figure out what else to add
//       Proper benchmarking, profiling - close feedback loop
//
// /// Prepared instruction type for efficient execution
// pub type PInstruction = Instruction<u32>;

// // TODO: move to instruction.rs
// /// Trait for instruction execution
// pub trait InsExecutor<Mem, Ctx, F>
// where
//     Mem: GuestMemory,
// {
//     fn execute(
//         &mut self,
//         state: &mut VmExecutionState<Mem, Ctx>,
//         // TODO: Change to PInstruction
//         instruction: &Instruction<F>,
//     ) -> Result<(), ExecutionError>
//     where
//         F: PrimeField32;
// }
//
// /// Trait for context with timestamp tracking
// pub trait Temporal {
//     fn timestamp(&self) -> u32;
//     fn timestamp_mut(&mut self) -> &mut u32;
// }

// /// Metered instruction executor wrapper for gas metering
// pub struct Metered<E>
// where
//     E: InsExecutor<Mem, Ctx>,
//     Mem: GuestMemory,
//     Ctx: Temporal,
// {
//     inner: E,
//     trace_height: usize,
//     weight: u32,
// }

// impl<E, Mem, Ctx> InsExecutor<Mem, Ctx> for Metered<E>
// where
//     E: InsExecutor<Mem, Ctx>,
//     Mem: GuestMemory,
//     Ctx: Temporal,
// {
//     fn execute(
//         &mut self,
//         state: &mut VmExecutionState<Mem, Ctx>,
//         instruction: &PInstruction,
//     ) -> Result<(), ExecutionError> {
//         // Execute the inner implementation
//         let result = self.inner.execute(state, instruction);

//         // Add gas costs based on weight and trace height
//         // This could be more complex based on your gas model
//         *state.ctx.timestamp_mut() += self.weight;

//         result
//     }
// }

// /// Trait for recording instruction execution into trace buffers
// pub trait RecordingExecutor<Mem, Ctx>: InsExecutor<Mem, Ctx>
// where
//     Mem: GuestMemory,
//     Ctx: Temporal,
// {
//     /// Calculate buffer size needed for recording based on instruction count
//     fn buffer_size(&self, ins_counter: usize) -> usize;

//     /// Set the buffer for recording execution trace
//     fn set_buffer(&mut self, buffer: &mut [u8]);

//     /// Get the current position in the buffer
//     fn buffer_position(&self) -> usize;
// }
