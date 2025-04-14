// TODO: Only use AddressMap/PagedVec
//       Add metric counter for gas metering, figure out what else to add
//       Proper benchmarking, profiling - close feedback loop
use backtrace::Backtrace;
use openvm_instructions::{
    exe::FnBounds,
    instruction::{DebugInfo, Instruction},
    program::Program,
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
    ExecutionError, GenerationError, Streams, SystemConfig, VmChipComplex, VmComplexTraceHeights,
    VmConfig,
};
#[cfg(feature = "bench-metrics")]
use crate::metrics::VmMetrics;
use crate::{
    arch::{instructions::*, ExecutionState, InstructionExecutor},
    system::memory::{online::GuestMemory, MemoryImage},
};

/// Check segment every 100 instructions.
const SEGMENT_CHECK_INTERVAL: usize = 100;

/// Prepared instruction type for efficient execution
pub type PInstruction = Instruction<u32>;

/// Represents the state of the VM during execution
pub struct VmExecutionState<Mem, Ctx>
where
    Mem: GuestMemory,
{
    /// Current timestamp representing clock cycles
    pub timestamp: u32,
    /// Program counter - current instruction address
    pub pc: u32,
    /// Whether execution has terminated
    // TODO: see if it can be removed
    pub terminated: bool,
    /// Guest memory interface
    pub memory: Mem,
    /// Host-specific execution context
    pub ctx: Ctx,
}

// TODO: replace execution::InstructionExecutor with this
/// Trait for instruction execution
pub trait InsExecutor<Mem, Ctx> {
    fn execute(
        &mut self,
        state: &mut VmExecutionState<Mem, Ctx>,
        instruction: &PInstruction,
    ) -> Result<(), ExecutionError>
    where
        Mem: GuestMemory;
}

pub struct VmSegmentExecutor<F, VC, Mem, Ctx>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    Mem: GuestMemory,
{
    pub chip_complex: VmChipComplex<F, VC::Executor, VC::Periphery>,
    pub vm_state: VmExecutionState<Mem, Ctx>,
    /// Memory image after segment was executed. Not used in trace generation.
    pub final_memory: Option<MemoryImage>,

    pub trace_height_constraints: Vec<LinearConstraint>,

    /// Air names for debug purposes only.
    pub(crate) air_names: Vec<String>,
    /// Metrics collected for this execution segment alone.
    #[cfg(feature = "bench-metrics")]
    pub metrics: VmMetrics,

    /// Counter for segment checking
    since_last_segment_check: usize,
}

impl<F, VC, Mem, Ctx> VmSegmentExecutor<F, VC, Mem, Ctx>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    Mem: GuestMemory,
{
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(
        config: &VC,
        program: Program<F>,
        init_streams: Streams<F>,
        initial_memory: Option<MemoryImage>,
        vm_state: VmExecutionState<Mem, Ctx>,
        trace_height_constraints: Vec<LinearConstraint>,
        #[allow(unused_variables)] fn_bounds: FnBounds,
    ) -> Self {
        let mut chip_complex = config.create_chip_complex().unwrap();
        chip_complex.set_streams(init_streams);
        let program = if !config.system().profiling {
            program.strip_debug_infos()
        } else {
            program
        };
        chip_complex.set_program(program);

        if let Some(initial_memory) = initial_memory {
            chip_complex.set_initial_memory(initial_memory);
        }
        let air_names = chip_complex.air_names();

        Self {
            chip_complex,
            vm_state,
            final_memory: None,
            air_names,
            trace_height_constraints,
            #[cfg(feature = "bench-metrics")]
            metrics: VmMetrics {
                fn_bounds,
                ..Default::default()
            },
            since_last_segment_check: 0,
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

    /// Stopping is triggered by should_segment() or if VM is terminated
    pub fn execute_segment(&mut self) -> Result<(), ExecutionError> {
        let mut prev_backtrace: Option<Backtrace> = None;

        self.chip_complex
            .connector_chip_mut()
            .begin(ExecutionState::new(
                self.vm_state.pc,
                self.vm_state.timestamp,
            ));

        loop {
            // Execute single instruction
            self.execute_instruction(&mut prev_backtrace)?;

            // Check if we should break after executing the instruction
            if self.vm_state.terminated {
                break;
            }

            // Check for segmentation
            if self.should_segment() {
                // End the current segment with connector chip
                self.chip_complex.connector_chip_mut().end(
                    ExecutionState::new(self.vm_state.pc, self.vm_state.timestamp),
                    None,
                );
                break;
            }
        }

        self.final_memory = Some(
            self.chip_complex
                .base
                .memory_controller
                .memory_image()
                .clone(),
        );

        Ok(())
    }

    /// Executes a single instruction and updates VM state
    fn execute_instruction(
        &mut self,
        prev_backtrace: &mut Option<Backtrace>,
    ) -> Result<(), ExecutionError> {
        let pc = self.vm_state.pc;
        let timestamp = self.vm_state.timestamp;

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
            self.vm_state.terminated = true;
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

        // Get executor for this opcode
        let executor = match self.chip_complex.inventory.get_mut_executor(&opcode) {
            Some(executor) => executor,
            None => return Err(ExecutionError::DisabledOperation { pc, opcode }),
        };

        // Execute the instruction
        let next_state = executor.execute(
            &mut self.chip_complex.base.memory_controller,
            instruction,
            ExecutionState::new(pc, timestamp),
        )?;

        assert!(next_state.timestamp > timestamp);
        self.vm_state.pc = next_state.pc;
        self.vm_state.timestamp = next_state.timestamp;

        // Update metrics if enabled
        #[cfg(feature = "bench-metrics")]
        {
            let dsl_instr_clone = dsl_instr.cloned();
            self.update_instruction_metrics(pc, opcode, dsl_instr_clone);
        }

        Ok(())
    }

    /// Returns bool of whether to switch to next segment or not.
    fn should_segment(&mut self) -> bool {
        if !self.system_config().continuation_enabled {
            return false;
        }

        // Avoid checking segment too often.
        if self.since_last_segment_check < SEGMENT_CHECK_INTERVAL {
            self.since_last_segment_check += 1;
            return false;
        }

        // Reset counter after checking
        self.since_last_segment_check = 0;

        let segmentation_strategy = &self.system_config().segmentation_strategy;
        segmentation_strategy.should_segment(
            &self.air_names,
            &self
                .chip_complex
                .dynamic_trace_heights()
                .collect::<Vec<_>>(),
            &self.chip_complex.current_trace_cells(),
        )
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

    pub fn current_trace_cells(&self) -> Vec<usize> {
        self.chip_complex.current_trace_cells()
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
                self.current_trace_cells(),
                opcode_name,
                dsl_instr,
            );

            #[cfg(feature = "function-span")]
            self.metrics.update_current_fn(pc);
        }
    }
}

// struct Metered<E>
// where
//   E: InsExecutor {
//     inner: E,
//     num_rows: usize
//     // weight
// }
// /// Trait for recording instruction execution into trace buffers
// pub trait RecordingIExecutor: InsExecutor<impl, impl> {
//     fn buffer_size(&self, ins_counter: usize) -> usize;
// }

// /// Trait for context with timestamp tracking
// pub trait Temporal {
//     fn timestamp(&self) -> u32;
//     fn timestamp_mut(&mut self) -> &mut u32;
// }

// /// Execution control trait for determining segmentation and stopping conditions
// pub trait ExecutionControl {
//     fn should_stop(&self, state: &VmExecutionState<impl, impl>) -> bool;

//     fn should_segment(&self, state: &VmExecutionState<impl, impl>) -> bool;
// }
