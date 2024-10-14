use std::{collections::VecDeque, mem::take};

use afs_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
};
use cycle_tracker::CycleTracker;
use metrics::VmMetrics;
use p3_field::PrimeField32;
pub use segment::ExecutionSegment;

use crate::{
    kernels::core::{CoreOptions, CoreState},
    system::{
        program::{ExecutionError, Program},
        vm::{config::VmConfig, segment::SegmentResult},
    },
};

pub mod config;
pub mod connector;
pub mod cycle_tracker;
/// Instrumentation metrics for performance analysis and debugging
pub mod metrics;
pub mod segment;

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
    /// Current state of the Core
    pub state: CoreState,
    /// Input stream of the Core
    pub input_stream: VecDeque<Vec<F>>,
    /// Hint stream of the Core
    pub hint_stream: VecDeque<F>,
}

pub struct VirtualMachineResult<SC: StarkGenericConfig> {
    pub segment_results: Vec<SegmentResult<SC>>,
}

impl<F: PrimeField32> VirtualMachine<F> {
    /// Create a new VM with a given config, program, and input stream.
    ///
    /// The VM will start with a single segment, which is created from the initial state of the Core.
    pub fn new(config: VmConfig, program: Program<F>, input_stream: Vec<Vec<F>>) -> Self {
        let mut vm = Self {
            config,
            program,
            segments: vec![],
        };
        vm.segment(
            VirtualMachineState {
                state: CoreState::initial(),
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
    fn segment(&mut self, state: VirtualMachineState<F>, cycle_tracker: CycleTracker) {
        tracing::debug!(
            "Creating new continuation segment for {} total segments",
            self.segments.len() + 1
        );
        let program = self.program.clone();
        let mut segment = ExecutionSegment::new(self.config.clone(), program, state);
        segment.cycle_tracker = cycle_tracker;
        self.segments.push(segment);
    }

    /// Retrieves the current state of the VM by querying the last segment.
    pub fn current_state(&self) -> VirtualMachineState<F> {
        let last_seg = self.segments.last().unwrap();
        VirtualMachineState {
            state: last_seg.core_chip.borrow().state,
            input_stream: last_seg.input_stream.clone(),
            hint_stream: last_seg.hint_stream.clone(),
        }
    }

    /// Retrieves the Core options from the VM's config.
    pub fn options(&self) -> CoreOptions {
        self.config.core_options()
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

    /// Executes the VM by calling `ExecutionSegment::execute()` until the Core hits `TERMINATE`
    /// and `core_chip.is_done`. Between every segment, the VM will call `next_segment()`.
    pub fn execute(mut self) -> Result<(), ExecutionError> {
        loop {
            let last_seg = self.segments.last_mut().unwrap();
            last_seg.execute()?;
            if last_seg.core_chip.borrow().state.is_done {
                break;
            }
            let cycle_tracker = take(&mut last_seg.cycle_tracker);
            self.segment(self.current_state(), cycle_tracker);
        }
        tracing::debug!("Number of continuation segments: {}", self.segments.len());

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
            last_seg.execute()?;
            if last_seg.core_chip.borrow().state.is_done {
                break;
            }
            let cycle_tracker = take(&mut last_seg.cycle_tracker);
            self.segment(self.current_state(), cycle_tracker);
        }
        tracing::debug!("Number of continuation segments: {}", self.segments.len());

        Ok(VirtualMachineResult {
            segment_results: self
                .segments
                .into_iter()
                .map(ExecutionSegment::produce_result)
                .collect(),
        })
    }
}
