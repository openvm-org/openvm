use std::{collections::VecDeque, mem::take};

use cycle_tracker::CycleTracker;
use metrics::VmMetrics;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_uni_stark::{Domain, StarkGenericConfig};
pub use segment::ExecutionSegment;

use crate::{
    cpu::{trace::ExecutionError, CpuOptions, CpuState},
    program::Program,
    vm::{config::VmConfig, cycle_tracker::CanPrint, segment::SegmentResult},
};

pub mod config;
pub mod cycle_tracker;
/// Instrumentation metrics for performance analysis and debugging
pub mod metrics;
pub mod segment;

pub type VmCycleTracker = CycleTracker<VmMetrics>;

/// Parent struct that holds all execution segments, program, config.
pub struct VirtualMachine<F: PrimeField32> {
    pub config: VmConfig,
    pub program: Program<F>,
    pub segments: Vec<ExecutionSegment<F>>,
}

/// Struct that holds the current state of the VM. For now, includes memory, input stream, and hint stream.
/// Hint stream cannot be added to during execution, but must be copied because it is popped from.
#[derive(Clone, Debug)]
pub struct VirtualMachineState<F: PrimeField32> {
    /// Current state of the CPU
    pub state: CpuState,
    /// Input stream of the CPU
    pub input_stream: VecDeque<Vec<F>>,
    /// Hint stream of the CPU
    pub hint_stream: VecDeque<F>,
}

pub struct VirtualMachineResult<SC: StarkGenericConfig> {
    pub segment_results: Vec<SegmentResult<SC>>,
    pub cycle_tracker: VmCycleTracker,
}

impl<F: PrimeField32> VirtualMachine<F> {
    /// Create a new VM with a given config, program, and input stream.
    ///
    /// The VM will start with a single segment, which is created from the initial state of the CPU.
    pub fn new(config: VmConfig, program: Program<F>, input_stream: Vec<Vec<F>>) -> Self {
        let mut vm = Self {
            config,
            program,
            segments: vec![],
        };
        vm.segment(
            VirtualMachineState {
                state: CpuState::initial(),
                input_stream: VecDeque::from(input_stream),
                hint_stream: VecDeque::new(),
            },
            VmCycleTracker::new(),
        );
        vm
    }

    /// Create a new segment with a given state.
    ///
    /// The segment will be created from the given state and the program.
    fn segment(&mut self, state: VirtualMachineState<F>, cycle_tracker: VmCycleTracker) {
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
            state: last_seg.cpu_chip.borrow().state,
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

    /// Executes the VM by calling `ExecutionSegment::execute()` until the CPU hits `TERMINATE`
    /// and `cpu_chip.is_done`. Between every segment, the VM will call `next_segment()`.
    pub fn execute(mut self) -> Result<(), ExecutionError> {
        loop {
            let last_seg = self.segments.last_mut().unwrap();
            last_seg.cycle_tracker.print();
            last_seg.execute()?;
            if last_seg.cpu_chip.borrow().state.is_done {
                break;
            }
            let cycle_tracker = take(&mut last_seg.cycle_tracker);
            self.segment(self.current_state(), cycle_tracker);
        }
        tracing::debug!("Number of continuation segments: {}", self.segments.len());
        let cycle_tracker = take(&mut self.segments.last_mut().unwrap().cycle_tracker);
        cycle_tracker.print();

        Ok(())
    }

    pub fn execute_and_generate<SC: StarkGenericConfig>(
        mut self,
    ) -> Result<VirtualMachineResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        loop {
            let last_seg = self.segments.last_mut().unwrap();
            last_seg.cycle_tracker.print();
            last_seg.execute()?;
            if last_seg.cpu_chip.borrow().state.is_done {
                break;
            }
            let cycle_tracker = take(&mut last_seg.cycle_tracker);
            self.segment(self.current_state(), cycle_tracker);
        }
        tracing::debug!("Number of continuation segments: {}", self.segments.len());
        let cycle_tracker = take(&mut self.segments.last_mut().unwrap().cycle_tracker);
        cycle_tracker.print();

        Ok(VirtualMachineResult {
            segment_results: self
                .segments
                .into_iter()
                .map(ExecutionSegment::produce_result)
                .collect(),
            cycle_tracker,
        })
    }
}
