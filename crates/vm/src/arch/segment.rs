use backtrace::Backtrace;
use openvm_instructions::{exe::FnBounds, instruction::DebugInfo, program::Program};
use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    p3_field::PrimeField32,
    prover::types::{CommittedTraceData, ProofInput},
    utils::metrics_span,
    Chip,
};

use super::{
    AnyEnum, ExecutionError, Streams, SystemConfig, VmChipComplex, VmComplexTraceHeights, VmConfig,
};
#[cfg(feature = "bench-metrics")]
use crate::metrics::VmMetrics;
use crate::{
    arch::{instructions::*, ExecutionState, InstructionExecutor},
    system::{
        memory::{Equipartition, CHUNK},
        poseidon2::Poseidon2PeripheryChip,
    },
};

/// Check segment every 100 instructions.
const SEGMENT_CHECK_INTERVAL: usize = 100;

pub struct ExecutionSegment<F, VC>
where
    F: PrimeField32,
    VC: VmConfig<F>,
{
    pub chip_complex: VmChipComplex<F, VC::Executor, VC::Periphery>,
    pub final_memory: Option<Equipartition<F, CHUNK>>,
    pub since_last_segment_check: usize,

    /// Air names for debug purposes only.
    pub(crate) air_names: Vec<String>,
    /// Metrics collected for this execution segment alone.
    #[cfg(feature = "bench-metrics")]
    pub(crate) metrics: VmMetrics,
}

pub struct ExecutionSegmentState {
    pub pc: u32,
    pub is_terminated: bool,
}

impl<F: PrimeField32, VC: VmConfig<F>> ExecutionSegment<F, VC> {
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(
        config: &VC,
        program: Program<F>,
        init_streams: Streams<F>,
        initial_memory: Option<Equipartition<F, CHUNK>>,
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
            chip_complex
                .memory_controller()
                .borrow_mut()
                .set_initial_memory(initial_memory);
        }
        let air_names = chip_complex.air_names();

        Self {
            chip_complex,
            final_memory: None,
            air_names,
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

    /// Stopping is triggered by should_segment()
    pub fn execute_from_pc(
        &mut self,
        mut pc: u32,
    ) -> Result<ExecutionSegmentState, ExecutionError> {
        let mut timestamp = self.chip_complex.memory_controller().borrow().timestamp();
        let mut prev_backtrace: Option<Backtrace> = None;

        self.chip_complex
            .connector_chip_mut()
            .begin(ExecutionState::new(pc, timestamp));

        let mut did_terminate = false;

        loop {
            let (instruction, debug_info) =
                self.chip_complex.program_chip_mut().get_instruction(pc)?;
            tracing::trace!("pc: {pc:#x} | time: {timestamp} | {:?}", instruction);

            #[allow(unused_variables)]
            let (dsl_instr, trace) = debug_info.map_or(
                (None, None),
                |DebugInfo {
                     dsl_instruction,
                     trace,
                 }| (Some(dsl_instruction), trace),
            );

            let opcode = instruction.opcode;
            if opcode == VmOpcode::with_default_offset(SystemOpcode::TERMINATE) {
                did_terminate = true;
                self.chip_complex.connector_chip_mut().end(
                    ExecutionState::new(pc, timestamp),
                    Some(instruction.c.as_canonical_u32()),
                );
                break;
            }

            // Some phantom instruction handling is more convenient to do here than in PhantomChip.
            if opcode == VmOpcode::with_default_offset(SystemOpcode::PHANTOM) {
                // Note: the discriminant is the lower 16 bits of the c operand.
                let discriminant = instruction.c.as_canonical_u32() as u16;
                let phantom = SysPhantom::from_repr(discriminant);
                tracing::trace!("pc: {pc:#x} | system phantom: {phantom:?}");
                match phantom {
                    Some(SysPhantom::DebugPanic) => {
                        if let Some(mut backtrace) = prev_backtrace {
                            backtrace.resolve();
                            eprintln!("openvm program failure; backtrace:\n{:?}", backtrace);
                        } else {
                            eprintln!("openvm program failure; no backtrace");
                        }
                        return Err(ExecutionError::Fail { pc });
                    }
                    Some(SysPhantom::CtStart) =>
                    {
                        #[cfg(feature = "bench-metrics")]
                        self.metrics
                            .cycle_tracker
                            .start(dsl_instr.clone().unwrap_or("Default".to_string()))
                    }
                    Some(SysPhantom::CtEnd) =>
                    {
                        #[cfg(feature = "bench-metrics")]
                        self.metrics
                            .cycle_tracker
                            .end(dsl_instr.clone().unwrap_or("Default".to_string()))
                    }
                    _ => {}
                }
            }
            prev_backtrace = trace;

            if let Some(executor) = self.chip_complex.inventory.get_mut_executor(&opcode) {
                let next_state = InstructionExecutor::execute(
                    executor,
                    instruction,
                    ExecutionState::new(pc, timestamp),
                )?;
                assert!(next_state.timestamp > timestamp);
                pc = next_state.pc;
                timestamp = next_state.timestamp;
            } else {
                return Err(ExecutionError::DisabledOperation { pc, opcode });
            };

            #[cfg(feature = "bench-metrics")]
            self.update_instruction_metrics(pc, opcode, dsl_instr);

            if self.should_segment() {
                self.chip_complex
                    .connector_chip_mut()
                    .end(ExecutionState::new(pc, timestamp), None);
                break;
            }
        }
        // Finalize memory.
        {
            // Need some partial borrows, so code is ugly:
            let mut memory_controller = self.chip_complex.base.memory_controller.borrow_mut();
            self.final_memory = if self.system_config().continuation_enabled {
                let chip = self
                    .chip_complex
                    .inventory
                    .periphery
                    .get_mut(
                        VmChipComplex::<F, VC::Executor, VC::Periphery>::POSEIDON2_PERIPHERY_IDX,
                    )
                    .expect("Poseidon2 chip required for persistent memory");
                let hasher: &mut Poseidon2PeripheryChip<F> = chip
                    .as_any_kind_mut()
                    .downcast_mut()
                    .expect("Poseidon2 chip required for persistent memory");
                memory_controller.finalize(Some(hasher))
            } else {
                memory_controller.finalize(None::<&mut Poseidon2PeripheryChip<F>>)
            };
        }
        #[cfg(feature = "bench-metrics")]
        self.finalize_metrics();

        Ok(ExecutionSegmentState {
            pc,
            is_terminated: did_terminate,
        })
    }

    /// Generate ProofInput to prove the segment. Should be called after ::execute
    pub fn generate_proof_input<SC: StarkGenericConfig>(
        self,
        cached_program: Option<CommittedTraceData<SC>>,
    ) -> ProofInput<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        metrics_span("trace_gen_time_ms", || {
            self.chip_complex.generate_proof_input(cached_program)
        })
    }

    /// Returns bool of whether to switch to next segment or not. This is called every clock cycle inside of Core trace generation.
    ///
    /// Default config: switch if any runtime chip height exceeds 1<<20 - 100
    fn should_segment(&mut self) -> bool {
        // Avoid checking segment too often.
        if self.since_last_segment_check != SEGMENT_CHECK_INTERVAL {
            self.since_last_segment_check += 1;
            return false;
        }
        self.since_last_segment_check = 0;
        let heights = self.chip_complex.dynamic_trace_heights();
        for (i, height) in heights.enumerate() {
            if height > self.system_config().max_segment_len {
                tracing::info!(
                    "Should segment because chip {} has height {}",
                    self.air_names[i],
                    height
                );
                return true;
            }
        }

        false
    }

    pub fn current_trace_cells(&self) -> Vec<usize> {
        self.chip_complex.current_trace_cells()
    }
    /// Gets current trace heights for each chip.
    /// Includes constant trace heights.
    pub fn current_trace_heights(&self) -> Vec<usize> {
        self.chip_complex.current_trace_heights()
    }
}
