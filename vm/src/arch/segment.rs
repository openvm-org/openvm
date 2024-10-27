use std::{collections::BTreeMap, ops::DerefMut, sync::Arc};

use ax_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    prover::types::{CommittedTraceData, ProofInput},
    ChipUsageGetter,
};
use axvm_instructions::{instruction::DebugInfo, program::Program};
use backtrace::Backtrace;
use itertools::zip_eq;
use p3_field::PrimeField32;
use parking_lot::Mutex;

use super::{PersistenceType, Streams, VmChipSet, VmConfig};
use crate::{
    arch::{instructions::*, AxVmChip, ExecutionState, InstructionExecutor},
    intrinsics::hashes::poseidon2::Poseidon2Chip,
    metrics::{cycle_tracker::CycleTracker, VmMetrics},
    system::{
        memory::{Equipartition, CHUNK},
        program::ExecutionError,
    },
};

pub struct ExecutionSegment<F: PrimeField32> {
    pub config: VmConfig,
    pub chip_set: VmChipSet<F>,

    // The streams should be mutated in serial without thread-safety,
    // but the `VmCoreChip` trait requires thread-safety.
    pub streams: Arc<Mutex<Streams<F>>>,

    pub final_memory: Option<Equipartition<F, CHUNK>>,

    pub cycle_tracker: CycleTracker,
    /// Collected metrics for this segment alone.
    /// Only collected when `config.collect_metrics` is true.
    pub(crate) collected_metrics: VmMetrics,
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
}

impl<F: PrimeField32> ExecutionSegment<F> {
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(
        config: VmConfig,
        program: Program<F>,
        streams: Arc<Mutex<Streams<F>>>,
        initial_memory: Option<Equipartition<F, CHUNK>>,
    ) -> Self {
        let mut chip_set = config.create_chip_set(streams.clone());
        chip_set.program_chip.set_program(program);

        if let Some(initial_memory) = initial_memory {
            chip_set
                .memory_controller
                .borrow_mut()
                .set_initial_memory(initial_memory);
        }

        Self {
            config,
            chip_set,
            streams,
            final_memory: None,
            collected_metrics: Default::default(),
            cycle_tracker: CycleTracker::new(),
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
                BTreeMap::new()
            };

            if opcode == CommonOpcode::TERMINATE.with_default_offset() {
                did_terminate = true;
                self.chip_set.connector_chip.end(
                    ExecutionState::new(pc, timestamp),
                    Some(instruction.c.as_canonical_u32()),
                );
                if collect_metrics {
                    self.update_chip_metrics();
                    #[cfg(feature = "bench-metrics")]
                    metrics::counter!("total_cells_used")
                        .absolute(self.current_trace_cells().into_values().sum::<usize>() as u64);
                }
                break;
            }

            // Some phantom instruction handling is more convenient to do here than in PhantomChip. FIXME[jpw]
            if opcode == CommonOpcode::PHANTOM as usize {
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
                        self.update_chip_metrics();
                        self.cycle_tracker.start(instruction.debug.clone())
                    }
                    PhantomInstruction::CtEnd => {
                        self.update_chip_metrics();
                        self.cycle_tracker.end(instruction.debug.clone())
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

                for (air_name, now_value) in &now_trace_cells {
                    let prev_value = prev_trace_cells.get(air_name).unwrap_or(&0);
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

        if collect_metrics {
            self.update_chip_metrics();
            #[cfg(feature = "bench-metrics")]
            self.collected_metrics.emit();
        }

        // Finalize memory.
        let mut memory_controller = self.chip_set.memory_controller.borrow_mut();
        self.final_memory = match self.config.memory_config.persistence_type {
            PersistenceType::Persistent => {
                let poseidon_chip = find_chip!(self.chip_set, AxVmChip::Poseidon2);
                let mut hasher = poseidon_chip.borrow_mut();

                memory_controller.finalize(Some(hasher.deref_mut()))
            }
            PersistenceType::Volatile => memory_controller.finalize(None::<&mut Poseidon2Chip<F>>),
        };

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
        self.chip_set
            .memory_controller
            .borrow()
            .current_trace_heights()
            .iter()
            .any(|&h| h > self.config.max_segment_len)
            || self
                .chip_set
                .chips
                .iter()
                .any(|chip| chip.current_trace_height() > self.config.max_segment_len)
    }

    fn current_trace_cells(&self) -> BTreeMap<String, usize> {
        self.chip_set.current_trace_cells()
    }

    pub(crate) fn update_chip_metrics(&mut self) {
        self.collected_metrics.chip_heights = self.chip_heights();
    }

    fn chip_heights(&self) -> BTreeMap<String, usize> {
        let mut metrics = BTreeMap::new();
        // TODO: more systematic handling of system chips: Program, Memory, Connector
        metrics.insert(
            "ProgramChip".into(),
            self.chip_set.program_chip.true_program_length,
        );
        for (air_name, height) in zip_eq(
            self.chip_set.memory_controller.borrow().air_names(),
            self.chip_set
                .memory_controller
                .borrow()
                .current_trace_heights(),
        ) {
            metrics.insert(format!("Memory {air_name}"), height);
        }
        for chip in self.chip_set.chips.iter() {
            let chip_name: &'static str = chip.into();
            metrics.insert(chip_name.into(), chip.current_trace_height());
        }
        metrics
    }
}
