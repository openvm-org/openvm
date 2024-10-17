use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    mem,
    ops::DerefMut,
    rc::Rc,
    sync::Arc,
};

use afs_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    prover::types::AirProofInput,
    Chip, ChipUsageGetter,
};
use backtrace::Backtrace;
use itertools::{izip, zip_eq};
use p3_field::PrimeField32;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use super::{cycle_tracker::CycleTracker, VirtualMachineState, VmConfig, VmMetrics};
use crate::{
    arch::{instructions::*, AxVmChip, ExecutionState, InstructionExecutor},
    intrinsics::hashes::poseidon2::Poseidon2Chip,
    kernels::core::{CoreChip, Streams},
    system::{
        program::{DebugInfo, ExecutionError, Program},
        vm::{chip_set::VmChipSet, config::PersistenceType},
    },
};

pub struct ExecutionSegment<F: PrimeField32> {
    pub config: VmConfig,
    pub chip_set: VmChipSet<F>,
    /// Shortcut to the core chip.
    pub core_chip: Rc<RefCell<CoreChip<F>>>,

    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,

    pub cycle_tracker: CycleTracker,
    /// Collected metrics for this segment alone.
    /// Only collected when `config.collect_metrics` is true.
    pub(crate) collected_metrics: VmMetrics,
}

pub struct SegmentResult<SC: StarkGenericConfig> {
    pub air_proof_inputs: Vec<AirProofInput<SC>>,
    pub metrics: VmMetrics,
}

impl<SC: StarkGenericConfig> SegmentResult<SC> {
    pub fn max_log_degree(&self) -> usize {
        self.air_proof_inputs
            .iter()
            .flat_map(|air_proof_input| {
                air_proof_input
                    .raw
                    .common_main
                    .as_ref()
                    .map(|trace| trace.height())
            })
            .map(log2_strict_usize)
            .max()
            .unwrap()
    }
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
    pub fn new(config: VmConfig, program: Program<F>, state: VirtualMachineState<F>) -> Self {
        let mut chip_set = config.create_chip_set();
        chip_set.program_chip.set_program(program);

        let core_chip = find_chip!(chip_set, AxVmChip::Core);
        core_chip.borrow_mut().set_start_state(state.state);

        Self {
            config,
            chip_set,
            core_chip,
            input_stream: state.input_stream,
            hint_stream: state.hint_stream.clone(),
            collected_metrics: Default::default(),
            cycle_tracker: CycleTracker::new(),
        }
    }

    /// Stopping is triggered by should_segment()
    pub fn execute(&mut self) -> Result<(), ExecutionError> {
        let mut timestamp = self.core_chip.borrow().state.timestamp;
        let mut pc = self.core_chip.borrow().state.pc;

        let mut collect_metrics = self.config.collect_metrics;
        // The backtrace for the previous instruction, if any.
        let mut prev_backtrace: Option<Backtrace> = None;

        self.core_chip.borrow_mut().streams = Streams {
            input_stream: self.input_stream.clone(),
            hint_stream: self.hint_stream.clone(),
        };

        self.chip_set
            .connector_chip
            .begin(ExecutionState::new(pc, timestamp));

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
            let prev_trace_cells = self.current_trace_cells();

            // runtime only instruction handling
            // FIXME: assumes CoreOpcode has offset 0:
            if opcode == CoreOpcode::FAIL as usize {
                if let Some(mut backtrace) = prev_backtrace {
                    backtrace.resolve();
                    eprintln!("eDSL program failure; backtrace:\n{:?}", backtrace);
                } else {
                    eprintln!("eDSL program failure; no backtrace");
                }
                return Err(ExecutionError::Fail(pc));
            }
            if opcode == CoreOpcode::CT_START as usize {
                self.update_chip_metrics();
                self.cycle_tracker.start(instruction.debug.clone())
            }
            if opcode == CoreOpcode::CT_END as usize {
                self.update_chip_metrics();
                self.cycle_tracker.end(instruction.debug.clone())
            }
            prev_backtrace = trace;

            let mut opcode_name = None;
            if self.chip_set.executors.contains_key(&opcode) {
                let executor = self.chip_set.executors.get_mut(&opcode).unwrap();
                match InstructionExecutor::execute(
                    executor,
                    instruction,
                    ExecutionState::new(pc, timestamp),
                ) {
                    Ok(next_state) => {
                        pc = next_state.pc;
                        timestamp = next_state.timestamp;
                    }
                    Err(e) => return Err(e),
                }
                if collect_metrics {
                    opcode_name = Some(executor.get_opcode_name(opcode));
                }
            } else {
                return Err(ExecutionError::DisabledOperation(pc, opcode));
            }

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
                if opcode == CoreOpcode::TERMINATE as usize {
                    self.update_chip_metrics();
                    // Due to row padding, the padded rows will all have opcode TERMINATE, so stop metric collection after the first one
                    collect_metrics = false;
                    #[cfg(feature = "bench-metrics")]
                    metrics::counter!("total_cells_used")
                        .absolute(now_trace_cells.into_values().sum::<usize>() as u64);
                }
            }
            if opcode == CoreOpcode::TERMINATE as usize {
                break;
            }
            if self.should_segment() {
                panic!("continuations not supported");
            }
        }

        self.chip_set
            .connector_chip
            .end(ExecutionState::new(pc, timestamp));

        let streams = mem::take(&mut self.core_chip.borrow_mut().streams);
        self.hint_stream = streams.hint_stream;
        self.input_stream = streams.input_stream;

        if collect_metrics {
            self.update_chip_metrics();
        }

        Ok(())
    }

    /// Compile the AIRs and trace generation outputs for the chips used in this segment
    /// Should be called after ::execute
    pub fn produce_result<SC: StarkGenericConfig>(self) -> SegmentResult<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let Self {
            config,
            chip_set,
            core_chip,
            ..
        } = self;
        // Drop all strong references to chips other than self.chips, which will be consumed next.
        drop(chip_set.executors);
        drop(core_chip);

        // Finalize memory.
        match config.memory_config.persistence_type {
            PersistenceType::Persistent => {
                let poseidon_chip = find_chip!(chip_set, AxVmChip::Poseidon2);
                let mut hasher = poseidon_chip.borrow_mut();

                chip_set
                    .memory_controller
                    .borrow_mut()
                    .finalize(Some(hasher.deref_mut()));
            }
            PersistenceType::Volatile => {
                chip_set
                    .memory_controller
                    .borrow_mut()
                    .finalize(None::<&mut Poseidon2Chip<F>>);
            }
        }

        let mut result = SegmentResult {
            air_proof_inputs: vec![chip_set.program_chip.into()],
            metrics: self.collected_metrics,
        };

        for chip in chip_set.chips {
            if chip.current_trace_height() != 0 {
                result
                    .air_proof_inputs
                    .push(chip.generate_air_proof_input());
            }
        }
        // System chips required by architecture: memory and connector
        // REVISIT: range checker is also system chip because memory requests from it, so range checker must generate trace last.
        {
            // memory
            let memory_controller = Rc::try_unwrap(chip_set.memory_controller)
                .expect("other chips still hold a reference to memory chip")
                .into_inner();
            let range_checker = memory_controller.range_checker.clone();
            let heights = memory_controller.current_trace_heights();
            let air_proof_inputs = memory_controller.generate_air_proof_inputs();
            for (height, air_proof_input) in izip!(heights, air_proof_inputs) {
                if height != 0 {
                    result.air_proof_inputs.push(air_proof_input);
                }
            }
            // range checker
            let chip = range_checker;
            let air = chip.air();
            let trace = chip.generate_trace();
            result
                .air_proof_inputs
                .push(AirProofInput::simple(air, trace, vec![]));
        }

        let trace = chip_set.connector_chip.generate_trace();
        result.air_proof_inputs.push(AirProofInput::simple_no_pis(
            Arc::new(chip_set.connector_chip.air),
            trace,
        ));

        result
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
        zip_eq(
            self.chip_set.memory_controller.borrow().air_names(),
            self.chip_set
                .memory_controller
                .borrow()
                .current_trace_cells(),
        )
        .chain(
            self.chip_set
                .chips
                .iter()
                .map(|chip| (chip.air_name(), chip.current_trace_cells())),
        )
        .collect()
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
