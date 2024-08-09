use std::{
    collections::{HashMap, VecDeque},
    mem::take,
    ops::Deref,
};

use afs_stark_backend::rap::AnyRap;
use afs_test_utils::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use cycle_tracker::CycleTracker;
use metrics::VmMetrics;
use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;
use p3_matrix::{dense::DenseMatrix, Matrix};
use p3_util::log2_strict_usize;

use crate::{
    cpu::{trace::ExecutionError, CpuOptions, ExecutionState},
    program::Program,
};

pub mod config;
pub mod cycle_tracker;
/// Instrumentation metrics for performance analysis and debugging
pub mod metrics;
mod segment;

pub use segment::{get_chips, ExecutionSegment};

use self::config::VmConfig;

/// Parent struct that holds all execution segments, program, config.
///
/// Key method is `vm.execute()` which consumes the VM and returns a `ExecutionResult` struct. Segment switching is handled by
/// `ExecutionSegment::should_segment()`, called every CPU clock cycle, which when `true`
///  triggers `VirtualMachine::next_segment()`.
///
/// Chips, traces, and public values should be retrieved by unpacking the `ExecutionResult` struct.
/// `VirtualMachine::get_chips()` can be used to convert the boxes of chips to concrete chips.
pub struct VirtualMachine<const WORD_SIZE: usize, F: PrimeField32> {
    pub config: VmConfig,
    pub program: Program<F>,
    pub segments: Vec<ExecutionSegment<WORD_SIZE, F>>,
    pub traces: Vec<DenseMatrix<F>>,
}

/// Enum representing the different types of chips used in the VM
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChipType {
    Cpu,
    Program,
    Memory,
    RangeChecker,
    FieldArithmetic,
    FieldExtension,
    Poseidon2,
}

/// Struct that holds the return state of the VM. StarkConfig is hardcoded to BabyBearPoseidon2Config.
pub struct ExecutionResult<const WORD_SIZE: usize> {
    /// Traces of the VM
    pub nonempty_traces: Vec<DenseMatrix<BabyBear>>,
    pub all_traces: Vec<DenseMatrix<BabyBear>>,
    /// Public inputs of the VM
    pub nonempty_pis: Vec<Vec<BabyBear>>,
    /// Boxed chips of the VM
    pub nonempty_chips: Vec<Box<dyn AnyRap<BabyBearPoseidon2Config>>>,
    pub unique_chips: Vec<Box<dyn AnyRap<BabyBearPoseidon2Config>>>,
    /// Types of the chips
    pub chip_types: Vec<ChipType>,
    /// Maximum log degree of the VM
    pub max_log_degree: usize,
    /// VM metrics per segment, only collected if enabled
    pub metrics: Vec<VmMetrics>,
    /// Cycle tracker for the entire execution
    pub cycle_tracker: CycleTracker,
}

/// Struct that holds the current state of the VM. For now, includes memory, input stream, and hint stream.
/// Hint stream cannot be added to during execution, but must be copied because it is popped from.
pub struct VirtualMachineState<F: PrimeField32> {
    /// Current state of the CPU
    state: ExecutionState,
    /// Current memory of the CPU
    memory: HashMap<(F, F), F>,
    /// Input stream of the CPU
    input_stream: VecDeque<Vec<F>>,
    /// Hint stream of the CPU
    hint_stream: VecDeque<F>,
}

impl<const WORD_SIZE: usize, F: PrimeField32> VirtualMachine<WORD_SIZE, F> {
    /// Create a new VM with a given config, program, and input stream.
    ///
    /// The VM will start with a single segment, which is created from the initial state of the CPU.
    pub fn new(config: VmConfig, program: Program<F>, input_stream: Vec<Vec<F>>) -> Self {
        let mut vm = Self {
            config,
            program,
            segments: vec![],
            traces: vec![],
        };
        vm.segment(
            VirtualMachineState {
                state: ExecutionState::default(),
                memory: HashMap::new(),
                input_stream: VecDeque::from(input_stream),
                hint_stream: VecDeque::new(),
            },
            CycleTracker::new(),
        );
        vm
    }

    /// Create a new segment with a given state.
    ///
    /// The segment will be created from the given state and the program.
    pub fn segment(&mut self, state: VirtualMachineState<F>, cycle_tracker: CycleTracker) {
        tracing::debug!(
            "Creating new continuation segment for {} total segments",
            self.segments.len() + 1
        );
        let program = self.program.clone();
        let mut segment = ExecutionSegment::new(self.config, program, state);
        segment.cycle_tracker = cycle_tracker;
        self.segments.push(segment);
    }

    /// Retrieves the current state of the VM by querying the last segment.
    pub fn current_state(&self) -> VirtualMachineState<F> {
        let last_seg = self.segments.last().unwrap();
        VirtualMachineState {
            state: last_seg.cpu_chip.state,
            memory: last_seg.memory_chip.memory_clone(),
            input_stream: last_seg.input_stream.clone(),
            hint_stream: last_seg.hint_stream.clone(),
        }
    }

    /// Retrieves the CPU options from the VM's config.
    pub fn options(&self) -> CpuOptions {
        self.config.cpu_options()
    }

    /// Enable metrics collection on all segments
    pub fn enable_metrics_collection(&mut self) {
        self.config.collect_metrics = true;
        for segment in &mut self.segments {
            segment.config.collect_metrics = true;
        }
    }

    /// Disable metrics collection on all segments
    pub fn disable_metrics_collection(&mut self) {
        self.config.collect_metrics = false;
        for segment in &mut self.segments {
            segment.config.collect_metrics = false;
        }
    }
}

/// Executes the VM by calling `ExecutionSegment::generate_traces()` until the CPU hits `TERMINATE`
/// and `cpu_chip.is_done`. Between every segment, the VM will call `generate_commitments()` and then
/// `next_segment()`.
impl<const WORD_SIZE: usize> VirtualMachine<WORD_SIZE, BabyBear> {
    pub fn execute(mut self) -> Result<ExecutionResult<WORD_SIZE>, ExecutionError> {
        let mut traces = vec![];
        let mut metrics = Vec::new();

        loop {
            let last_seg = self.segments.last_mut().unwrap();
            last_seg.cycle_tracker.print();
            last_seg.has_generation_happened = true;
            traces.extend(last_seg.generate_traces()?);
            traces.extend(last_seg.generate_commitments()?);
            if self.config.collect_metrics {
                metrics.push(last_seg.metrics.clone());
            }
            if last_seg.cpu_chip.state.is_done {
                break;
            }
            let cycle_tracker = take(&mut last_seg.cycle_tracker);
            self.segment(self.current_state(), cycle_tracker);
        }
        tracing::debug!("Number of continuation segments: {}", self.segments.len());
        let cycle_tracker = take(&mut self.segments.last_mut().unwrap().cycle_tracker);
        cycle_tracker.print();

        let mut pis = Vec::with_capacity(self.segments.len());
        let mut chips = Vec::with_capacity(self.segments.len());
        let mut types = Vec::with_capacity(self.segments.len());
        let num_chips = self.segments[0].get_num_chips();

        let empty_program = Program {
            instructions: vec![],
            debug_infos: vec![],
        };

        let unique_chips = get_chips::<WORD_SIZE, BabyBearPoseidon2Config>(
            ExecutionSegment::new(self.config, empty_program, self.current_state()),
            &vec![true; num_chips],
        );

        // Iterate over each segment and add its public inputs, types, and chips to the result,
        // skipping empty traces.
        for (i, segment) in self.segments.into_iter().enumerate() {
            let trace_slice = &traces[i * num_chips..(i + 1) * num_chips];
            let inclusion_mask = trace_slice
                .iter()
                .map(|trace| !trace.values.is_empty())
                .collect::<Vec<bool>>();

            let segment_pis = segment.get_pis();
            let segment_types = segment.get_types();
            chips.extend(get_chips::<WORD_SIZE, BabyBearPoseidon2Config>(
                segment,
                &inclusion_mask,
            ));
            for index in 0..inclusion_mask.len() {
                if inclusion_mask[index] {
                    pis.push(segment_pis[index].clone());
                    types.push(segment_types[index]);
                }
            }
        }

        let nonempty_traces = traces
            .iter()
            .filter(|trace| !trace.values.is_empty())
            .cloned()
            .collect::<Vec<DenseMatrix<BabyBear>>>();

        let max_log_degree =
            log2_strict_usize(traces.iter().map(|trace| trace.height()).max().unwrap());

        // Assert that trace heights are within the max_len, except for Program and RangeChecker
        // +31 is needed because Poseidon2Permute adds 32 rows to memory at once
        traces
            .iter()
            .zip(types.iter())
            .filter(|(_, &chip_type)| {
                chip_type != ChipType::Program && chip_type != ChipType::RangeChecker
            })
            .for_each(|(trace, chip_type)| {
                assert!(
                    trace.height() <= (self.config.max_segment_len + 31).next_power_of_two(),
                    "Trace height for {:?} exceeds max_len. Height: {}, Max: {}",
                    chip_type,
                    trace.height(),
                    self.config.max_segment_len
                );
            });

        let chip_data = ExecutionResult {
            nonempty_traces,
            all_traces: traces,
            nonempty_pis: pis,
            nonempty_chips: chips,
            unique_chips,
            chip_types: types,
            max_log_degree,
            metrics,
            cycle_tracker,
        };

        Ok(chip_data)
    }

    /// Convert the VM's chips from Boxes to concrete types.
    pub fn get_chips(
        chips: &[Box<dyn AnyRap<BabyBearPoseidon2Config>>],
    ) -> Vec<&dyn AnyRap<BabyBearPoseidon2Config>> {
        chips.iter().map(|x| x.deref()).collect()
    }
}

impl<const WORD_SIZE: usize> ExecutionResult<WORD_SIZE> {
    pub fn get_chips(&self) -> Vec<&dyn AnyRap<BabyBearPoseidon2Config>> {
        self.nonempty_chips.iter().map(|x| x.deref()).collect()
    }
}
