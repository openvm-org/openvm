use std::{ops::DerefMut, sync::Arc};

use ax_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    prover::types::{CommittedTraceData, ProofInput},
};
use axvm_instructions::{instruction::DebugInfo, program::Program};
use backtrace::Backtrace;
use itertools::izip;
use p3_field::PrimeField32;
use parking_lot::Mutex;

use super::{AxVmExecutor, Streams, VmChipSet, VmConfig};
use crate::{
    arch::{instructions::*, AxVmChip, ExecutionState, InstructionExecutor},
    intrinsics::hashes::poseidon2::Poseidon2Chip,
    metrics::{cycle_tracker::CycleTracker, VmMetrics},
    system::{
        memory::{Equipartition, CHUNK},
        program::ExecutionError,
    },
};

/// Check segment every 100 instructions.
const SEGMENT_CHECK_INTERVAL: usize = 100;

pub struct ExecutionSegment<F: PrimeField32> {
    pub config: VmConfig,
    pub chip_set: VmChipSet<F>,

    pub final_memory: Option<Equipartition<F, CHUNK>>,

    pub cycle_tracker: CycleTracker,
    /// Collected metrics for this segment alone.
    /// Only collected when `config.collect_metrics` is true.
    pub(crate) collected_metrics: VmMetrics,
    pub air_names: Vec<String>,
    pub const_height_air_ids: Vec<usize>,
    pub since_last_segment_check: usize,
}

pub struct ExecutionSegmentState {
    pub pc: u32,
    pub is_terminated: bool,
}

macro_rules! find_chip {
    ($chip_set:expr, $chip_type:path) => {{
        let mut found_chip = None;
        for chip in &$chip_set.chips {
            if let $chip_type(c) = chip {
                assert!(
                    found_chip.is_none(),
                    concat!("Multiple ", stringify!($chip_type), " chips found")
                );
                found_chip = Some(c.clone());
            }
        }
        found_chip.unwrap()
    }};
    ($chip_set:expr, $chip_type:path, $chip_type2:path) => {{
        let mut found_chip = None;
        for chip in &$chip_set.chips {
            if let $chip_type($chip_type2(c)) = chip {
                assert!(
                    found_chip.is_none(),
                    concat!("Multiple ", stringify!($chip_type), " chips found")
                );
                found_chip = Some(c.clone());
            }
        }
        found_chip.unwrap()
    }};
}

impl<F: PrimeField32> ExecutionSegment<F> {
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(
        config: VmConfig,
        program: Program<F>,
        streams: Arc<Mutex<Streams<F>>>,
        initial_memory: Option<Equipartition<F, CHUNK>>,
    ) -> Self {
        let mut chip_set = config.create_chip_set();
        chip_set.set_streams(streams);
        chip_set.set_program(program);

        if let Some(initial_memory) = initial_memory {
            chip_set
                .memory_controller
                .borrow_mut()
                .set_initial_memory(initial_memory);
        }
        let air_names = chip_set.air_names();
        let const_height_air_ids = chip_set.const_height_air_ids();

        Self {
            config,
            chip_set,
            final_memory: None,
            collected_metrics: Default::default(),
            cycle_tracker: CycleTracker::new(),
            air_names,
            const_height_air_ids,
            since_last_segment_check: 0,
        }
    }

    /// Stopping is triggered by should_segment()
    pub fn execute_from_pc(
        &mut self,
        mut pc: u32,
    ) -> Result<ExecutionSegmentState, ExecutionError> {
        let mut timestamp = self.chip_set.memory_controller.borrow().timestamp();

        let collect_metrics = self.config.collect_metrics;
        // The backtrace for the previous instruction, if any.
        let mut prev_backtrace: Option<Backtrace> = None;

        self.chip_set
            .connector_chip
            .begin(ExecutionState::new(pc, timestamp));

        let mut did_terminate = false;

        loop {
            let (instruction, debug_info) = self.chip_set.program_chip.get_instruction(pc)?;
            tracing::trace!("pc: {pc:#x} | time: {timestamp} | {:?}", instruction);

            let (dsl_instr, trace) = debug_info.map_or(
                (None, None),
                |DebugInfo {
                     dsl_instruction,
                     trace,
                 }| (Some(dsl_instruction), trace),
            );

            let opcode = instruction.opcode;
            let prev_trace_cells = if collect_metrics {
                self.current_trace_cells()
            } else {
                vec![]
            };

            if opcode == SystemOpcode::TERMINATE.with_default_offset() {
                did_terminate = true;
                self.chip_set.connector_chip.end(
                    ExecutionState::new(pc, timestamp),
                    Some(instruction.c.as_canonical_u32()),
                );
                break;
            }

            // Some phantom instruction handling is more convenient to do here than in PhantomChip. FIXME[jpw]
            if opcode == SystemOpcode::PHANTOM as usize {
                // Note: the discriminant is the lower 16 bits of the c operand.
                let discriminant = instruction.c.as_canonical_u32() as u16;
                let phantom = PhantomInstruction::from_repr(discriminant)
                    .ok_or(ExecutionError::InvalidPhantomInstruction(pc, discriminant))?;
                tracing::trace!("pc: {pc:#x} | phantom: {phantom:?}");
                match phantom {
                    PhantomInstruction::DebugPanic => {
                        if let Some(mut backtrace) = prev_backtrace {
                            backtrace.resolve();
                            eprintln!("axvm program failure; backtrace:\n{:?}", backtrace);
                        } else {
                            eprintln!("axvm program failure; no backtrace");
                        }
                        return Err(ExecutionError::Fail(pc));
                    }
                    PhantomInstruction::CtStart => {
                        // hack to remove "CT-" prefix
                        self.cycle_tracker.start(
                            dsl_instr.clone().unwrap_or("CT-Default".to_string())[3..].to_string(),
                        )
                    }
                    PhantomInstruction::CtEnd => {
                        // hack to remove "CT-" prefix
                        self.cycle_tracker.end(
                            dsl_instr.clone().unwrap_or("CT-Default".to_string())[3..].to_string(),
                        )
                    }
                    _ => {}
                }
            }
            prev_backtrace = trace;

            let mut opcode_name = None;
            if let Some(executor) = self.chip_set.executors.get_mut(&opcode) {
                let next_state = InstructionExecutor::execute(
                    executor,
                    instruction,
                    ExecutionState::new(pc, timestamp),
                )?;
                assert!(next_state.timestamp > timestamp);
                if collect_metrics {
                    opcode_name = Some(executor.get_opcode_name(opcode));
                }
                pc = next_state.pc;
                timestamp = next_state.timestamp;
            } else {
                return Err(ExecutionError::DisabledOperation(pc, opcode));
            };

            if collect_metrics {
                let now_trace_cells = self.current_trace_cells();

                let opcode_name = opcode_name.unwrap_or(opcode.to_string());
                let key = (dsl_instr.clone(), opcode_name.clone());
                #[cfg(feature = "bench-metrics")]
                self.cycle_tracker.increment_opcode(&key);
                *self.collected_metrics.counts.entry(key).or_insert(0) += 1;

                for (air_name, now_value, &prev_value) in
                    izip!(&self.air_names, now_trace_cells, &prev_trace_cells)
                {
                    if prev_value != now_value {
                        let key = (dsl_instr.clone(), opcode_name.clone(), air_name.to_owned());
                        #[cfg(feature = "bench-metrics")]
                        self.cycle_tracker
                            .increment_cells_used(&key, now_value - prev_value);
                        *self.collected_metrics.trace_cells.entry(key).or_insert(0) +=
                            now_value - prev_value;
                    }
                }
            }
            if self.should_segment() {
                self.chip_set
                    .connector_chip
                    .end(ExecutionState::new(pc, timestamp), None);
                break;
            }
        }
        // Finalize memory.
        {
            let mut memory_controller = self.chip_set.memory_controller.borrow_mut();
            self.final_memory = if self.config.continuation_enabled {
                let poseidon_chip =
                    find_chip!(self.chip_set, AxVmChip::Executor, AxVmExecutor::Poseidon2);
                let mut hasher = poseidon_chip.borrow_mut();
                memory_controller.finalize(Some(hasher.deref_mut()))
            } else {
                memory_controller.finalize(None::<&mut Poseidon2Chip<F>>)
            };
        }
        if collect_metrics {
            self.collected_metrics.chip_heights =
                izip!(self.air_names.clone(), self.current_trace_heights()).collect();
            #[cfg(feature = "bench-metrics")]
            {
                self.collected_metrics.emit();
                if did_terminate {
                    metrics::counter!("total_cells_used")
                        .absolute(self.current_trace_cells().into_iter().sum::<usize>() as u64);
                }
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
    {
        self.chip_set.generate_proof_input(cached_program)
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
        let heights = self.current_trace_heights();
        let mut const_height_idx = 0;
        for (i, (air_name, height)) in izip!(&self.air_names, heights).enumerate() {
            if const_height_idx >= self.const_height_air_ids.len()
                || self.const_height_air_ids[const_height_idx] != i
            {
                if height > self.config.max_segment_len {
                    tracing::info!(
                        "Should segment because chip {} has height {}",
                        air_name,
                        height
                    );
                    return true;
                }
                const_height_idx += 1;
            }
        }

        false
    }

    pub fn current_trace_cells(&self) -> Vec<usize> {
        self.chip_set.current_trace_cells()
    }
    pub fn current_trace_heights(&self) -> Vec<usize> {
        self.chip_set.current_trace_heights()
    }
}
