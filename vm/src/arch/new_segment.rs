use ax_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    prover::types::{CommittedTraceData, ProofInput},
    Chip,
};
#[cfg(feature = "function-span")]
use axvm_instructions::exe::FnBound;
use axvm_instructions::{exe::FnBounds, instruction::DebugInfo, program::Program};
use backtrace::Backtrace;
use p3_field::PrimeField32;

use super::{AnyEnum, ExecutionError, Streams, SystemConfig, VmChipComplex, VmGenericConfig};
#[cfg(feature = "bench-metrics")]
use crate::metrics::VmMetrics;
use crate::{
    arch::{instructions::*, ExecutionState, InstructionExecutor},
    intrinsics::hashes::poseidon2::Poseidon2Chip,
    metrics::cycle_tracker::CycleTracker,
    system::memory::{Equipartition, CHUNK},
};

/// Check segment every 100 instructions.
const SEGMENT_CHECK_INTERVAL: usize = 100;

pub struct ExecutionSegment<F, VmConfig>
where
    F: PrimeField32,
    VmConfig: VmGenericConfig<F>,
{
    pub chip_complex: VmChipComplex<F, VmConfig::Executor, VmConfig::Periphery>,

    pub final_memory: Option<Equipartition<F, CHUNK>>,

    /// Metric collection tools. Only collected when `config.collect_metrics` is true.
    pub cycle_tracker: CycleTracker,
    #[cfg(feature = "bench-metrics")]
    pub(crate) collected_metrics: VmMetrics,

    #[allow(dead_code)]
    pub(crate) fn_bounds: FnBounds,

    pub air_names: Vec<String>,
    pub since_last_segment_check: usize,
}

pub struct ExecutionSegmentState {
    pub pc: u32,
    pub is_terminated: bool,
}

impl<F: PrimeField32, VmConfig: VmGenericConfig<F>> ExecutionSegment<F, VmConfig> {
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(
        config: &VmConfig,
        program: Program<F>,
        init_streams: Streams<F>,
        initial_memory: Option<Equipartition<F, CHUNK>>,
        fn_bounds: FnBounds,
    ) -> Self {
        let mut chip_complex = config.create_chip_complex().unwrap();
        chip_complex.set_streams(init_streams);
        let program = if config.system().collect_metrics {
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
            cycle_tracker: CycleTracker::new(),
            #[cfg(feature = "bench-metrics")]
            collected_metrics: Default::default(),
            fn_bounds,
            air_names,
            since_last_segment_check: 0,
        }
    }

    pub fn system_config(&self) -> &SystemConfig {
        self.chip_complex.config()
    }

    /// Stopping is triggered by should_segment()
    pub fn execute_from_pc(
        &mut self,
        mut pc: u32,
    ) -> Result<ExecutionSegmentState, ExecutionError> {
        let mut timestamp = self.chip_complex.memory_controller().borrow().timestamp();

        #[cfg(feature = "bench-metrics")]
        let collect_metrics = self.system_config().collect_metrics;
        // The backtrace for the previous instruction, if any.
        let mut prev_backtrace: Option<Backtrace> = None;

        // Cycle span by function if function start/end addresses are available
        #[cfg(feature = "function-span")]
        let mut current_fn = FnBound::default();

        self.chip_complex
            .connector_chip_mut()
            .begin(ExecutionState::new(pc, timestamp));

        let mut did_terminate = false;

        loop {
            let (instruction, debug_info) =
                self.chip_complex.program_chip_mut().get_instruction(pc)?;
            tracing::trace!("pc: {pc:#x} | time: {timestamp} | {:?}", instruction);

            let (dsl_instr, trace) = debug_info.map_or(
                (None, None),
                |DebugInfo {
                     dsl_instruction,
                     trace,
                 }| (Some(dsl_instruction), trace),
            );

            let opcode = instruction.opcode;
            #[cfg(feature = "bench-metrics")]
            let prev_trace_cells = if collect_metrics {
                self.current_trace_cells()
            } else {
                vec![]
            };

            if opcode == SystemOpcode::TERMINATE.with_default_offset() {
                did_terminate = true;
                self.chip_complex.connector_chip_mut().end(
                    ExecutionState::new(pc, timestamp),
                    Some(instruction.c.as_canonical_u32()),
                );
                break;
            }

            // Some phantom instruction handling is more convenient to do here than in PhantomChip.
            if opcode == SystemOpcode::PHANTOM as usize {
                // Note: the discriminant is the lower 16 bits of the c operand.
                let discriminant = instruction.c.as_canonical_u32() as u16;
                let phantom = SysPhantom::from_repr(discriminant);
                tracing::trace!("pc: {pc:#x} | system phantom: {phantom:?}");
                match phantom {
                    Some(SysPhantom::DebugPanic) => {
                        if let Some(mut backtrace) = prev_backtrace {
                            backtrace.resolve();
                            eprintln!("axvm program failure; backtrace:\n{:?}", backtrace);
                        } else {
                            eprintln!("axvm program failure; no backtrace");
                        }
                        return Err(ExecutionError::Fail { pc });
                    }
                    Some(SysPhantom::CtStart) => {
                        // hack to remove "CT-" prefix
                        #[cfg(not(feature = "function-span"))]
                        self.cycle_tracker.start(
                            dsl_instr.clone().unwrap_or("CT-Default".to_string())[3..].to_string(),
                        )
                    }
                    Some(SysPhantom::CtEnd) => {
                        // hack to remove "CT-" prefix
                        #[cfg(not(feature = "function-span"))]
                        self.cycle_tracker.end(
                            dsl_instr.clone().unwrap_or("CT-Default".to_string())[3..].to_string(),
                        )
                    }
                    _ => {}
                }
            }
            prev_backtrace = trace;

            #[cfg(feature = "function-span")]
            if !self.fn_bounds.is_empty() && (pc < current_fn.start || pc > current_fn.end) {
                current_fn = self
                    .fn_bounds
                    .range(..=pc)
                    .next_back()
                    .map(|(_, func)| (*func).clone())
                    .unwrap();
                if pc == current_fn.start {
                    self.cycle_tracker.start(current_fn.name.clone());
                } else {
                    self.cycle_tracker.force_end();
                }
            };

            #[cfg(feature = "bench-metrics")]
            let mut opcode_name = None;
            if let Some(executor) = self.chip_complex.inventory.get_mut_executor(&opcode) {
                let next_state = InstructionExecutor::execute(
                    executor,
                    instruction,
                    ExecutionState::new(pc, timestamp),
                )?;
                assert!(next_state.timestamp > timestamp);
                #[cfg(feature = "bench-metrics")]
                if collect_metrics {
                    opcode_name = Some(executor.get_opcode_name(opcode));
                }
                pc = next_state.pc;
                timestamp = next_state.timestamp;
            } else {
                return Err(ExecutionError::DisabledOperation { pc, opcode });
            };

            #[cfg(feature = "bench-metrics")]
            if collect_metrics {
                let now_trace_cells = self.current_trace_cells();

                let opcode_name = opcode_name.unwrap_or(opcode.to_string());
                let key = (dsl_instr.clone(), opcode_name.clone());
                self.cycle_tracker.increment_opcode(&key);
                *self.collected_metrics.counts.entry(key).or_insert(0) += 1;

                for (air_name, now_value, &prev_value) in
                    itertools::izip!(&self.air_names, now_trace_cells, &prev_trace_cells)
                {
                    if prev_value != now_value {
                        let key = (dsl_instr.clone(), opcode_name.clone(), air_name.to_owned());
                        self.cycle_tracker
                            .increment_cells_used(&key, now_value - prev_value);
                        *self.collected_metrics.trace_cells.entry(key).or_insert(0) +=
                            now_value - prev_value;
                    }
                }
            }
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
                    .get_mut(VmChipComplex::<F, VmConfig::Executor, VmConfig::Periphery>::POSEIDON2_PERIPHERY_IDX)
                    .expect("Poseidon2 chip required for persistent memory");
                let hasher: &mut Poseidon2Chip<F> = chip
                    .as_any_kind_mut()
                    .downcast_mut()
                    .expect("Poseidon2 chip required for persistent memory");
                memory_controller.finalize(Some(hasher))
            } else {
                memory_controller.finalize(None::<&mut Poseidon2Chip<F>>)
            };
        }
        #[cfg(feature = "bench-metrics")]
        if collect_metrics {
            self.collected_metrics.chip_heights =
                itertools::izip!(self.air_names.clone(), self.current_trace_heights()).collect();

            self.collected_metrics.emit();
            if did_terminate {
                metrics::counter!("total_cells_used")
                    .absolute(self.current_trace_cells().into_iter().sum::<usize>() as u64);
            }
        }

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
        VmConfig::Executor: Chip<SC>,
        VmConfig::Periphery: Chip<SC>,
    {
        self.chip_complex.generate_proof_input(cached_program)
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
