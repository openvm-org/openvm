use std::{collections::VecDeque, mem};

use afs_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    prover::types::ProofInput,
};
use metrics::VmMetrics;
use p3_field::PrimeField32;
pub use segment::ExecutionSegment;

use crate::{
    intrinsics::hashes::poseidon2::CHUNK,
    kernels::core::CoreState,
    system::{
        memory::Equipartition,
        program::{ExecutionError, Program},
        vm::config::VmConfig,
    },
};

pub mod chip_set;
pub mod config;
pub mod connector;
pub mod cycle_tracker;
/// Instrumentation metrics for performance analysis and debugging
pub mod metrics;
pub mod segment;

/// Parent struct that holds all execution segments, program, config.
pub struct VirtualMachine<F: PrimeField32> {
    pub config: VmConfig,
    input_stream: VecDeque<Vec<F>>,
    initial_memory: Option<Equipartition<F, CHUNK>>,
    // TODO[zach]: Make better interface for user IOs
    program_inputs: Vec<(usize, F)>,
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
    pub per_segment: Vec<ProofInput<SC>>,
}

impl<F: PrimeField32> VirtualMachine<F> {
    /// Create a new VM with a given config, program, and input stream.
    ///
    /// The VM will start with a single segment, which is created from the initial state of the Core.
    pub fn new(config: VmConfig) -> Self {
        Self {
            config,
            input_stream: VecDeque::new(),
            initial_memory: None,
            program_inputs: vec![],
        }
    }

    pub fn with_input_stream(mut self, input_stream: Vec<Vec<F>>) -> Self {
        self.input_stream = VecDeque::from(input_stream);
        self
    }

    pub fn with_initial_memory(mut self, memory: Equipartition<F, CHUNK>) -> Self {
        self.initial_memory = Some(memory);
        self
    }

    pub fn with_program_inputs(mut self, program_inputs: Vec<(usize, F)>) -> Self {
        self.program_inputs = program_inputs;
        self
    }

    fn execute_segments(
        &mut self,
        program: Program<F>,
    ) -> Result<Vec<ExecutionSegment<F>>, ExecutionError> {
        let mut segments = vec![];
        let mut segment = ExecutionSegment::new(
            self.config.clone(),
            program.clone(),
            VirtualMachineState {
                state: CoreState::initial(program.pc_start),
                input_stream: mem::take(&mut self.input_stream),
                hint_stream: VecDeque::new(),
            },
            self.initial_memory.take(),
        );

        loop {
            // TODO[zach]: User public values currently set on all segments on the core chip.
            // This needs to change.
            {
                let mut core_chip = segment.core_chip.borrow_mut();
                for &(idx, public_value) in self.program_inputs.iter() {
                    core_chip.public_values[idx] = Some(public_value);
                }
            }
            segment.execute()?;
            if segment.did_terminate() {
                break;
            }

            let config = mem::take(&mut segment.config);
            let cycle_tracker = mem::take(&mut segment.cycle_tracker);
            let state = VirtualMachineState {
                state: CoreState {
                    pc: segment.chip_set.connector_chip.boundary_states[1]
                        .unwrap()
                        .pc,
                    is_done: false,
                },
                input_stream: mem::take(&mut segment.input_stream),
                hint_stream: mem::take(&mut segment.hint_stream),
            };
            let final_memory = mem::take(&mut segment.final_memory)
                .expect("final memory should be set in continuations segment");

            segments.push(segment);

            segment = ExecutionSegment::new(config, program.clone(), state, Some(final_memory));
            segment.cycle_tracker = cycle_tracker;
        }
        segments.push(segment);
        tracing::debug!("Number of continuation segments: {}", segments.len());

        Ok(segments)
    }

    pub fn execute(mut self, program: Program<F>) -> Result<(), ExecutionError> {
        self.execute_segments(program).map(|_| ())
    }

    pub fn execute_and_generate<SC: StarkGenericConfig>(
        mut self,
        program: Program<F>,
    ) -> Result<VirtualMachineResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let segments = self.execute_segments(program)?;

        Ok(VirtualMachineResult {
            per_segment: segments
                .into_iter()
                .map(ExecutionSegment::generate_proof_input)
                .collect(),
        })
    }
}
