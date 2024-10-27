use std::{collections::VecDeque, marker::PhantomData, mem, sync::Arc};

use ax_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    prover::types::ProofInput,
};
use axvm_instructions::program::Program;
use p3_field::PrimeField32;
use parking_lot::Mutex;

use crate::{
    arch::{ExecutionSegment, PersistenceType, VmConfig},
    intrinsics::hashes::poseidon2::CHUNK,
    system::{
        memory::Equipartition,
        program::{trace::CommittedProgram, ExecutionError},
    },
};

#[derive(Clone, Default, Debug)]
pub struct Streams<F> {
    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,
}

/// Parent struct that holds all execution segments, program, config.
pub struct VirtualMachine<F: PrimeField32> {
    pub config: VmConfig,
    /// Streams are shared between `ExecutionSegment`s and within each segment shared
    /// with any chip(s) that handle hint opcodes
    streams: Arc<Mutex<Streams<F>>>,
    initial_memory: Option<Equipartition<F, CHUNK>>,
}

#[repr(i32)]
pub enum ExitCode {
    Success = 0,
    Error = 1,
    Suspended = -1, // Continuations
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
            streams: Arc::new(Mutex::new(Streams::default())),
            initial_memory: None,
        }
    }

    pub fn with_input_stream(self, input_stream: Vec<Vec<F>>) -> Self {
        self.streams.lock().input_stream = VecDeque::from(input_stream);
        self
    }

    pub fn with_initial_memory(mut self, memory: Equipartition<F, CHUNK>) -> Self {
        self.initial_memory = Some(memory);
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
            self.streams.clone(),
            self.initial_memory.take(),
        );
        let mut pc = program.pc_start;

        loop {
            let state = segment.execute_from_pc(pc)?;
            pc = state.pc;

            if state.is_terminated {
                break;
            }

            assert_eq!(
                self.config.memory_config.persistence_type,
                PersistenceType::Persistent,
                "cannot segment in volatile memory mode"
            );

            assert_eq!(
                pc,
                segment.chip_set.connector_chip.boundary_states[1]
                    .unwrap()
                    .pc
            );

            let config = mem::take(&mut segment.config);
            let cycle_tracker = mem::take(&mut segment.cycle_tracker);
            let final_memory = mem::take(&mut segment.final_memory)
                .expect("final memory should be set in continuations segment");

            segments.push(segment);

            segment = ExecutionSegment::new(
                config,
                program.clone(),
                self.streams.clone(),
                Some(final_memory),
            );
            segment.cycle_tracker = cycle_tracker;
        }
        segments.push(segment);
        tracing::debug!("Number of continuation segments: {}", segments.len());

        Ok(segments)
    }

    pub fn execute(mut self, program: Program<F>) -> Result<(), ExecutionError> {
        #[cfg(test)]
        ax_sdk::config::setup_tracing_with_log_level(tracing::Level::WARN);
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
                .map(|seg| seg.generate_proof_input(None))
                .collect(),
        })
    }
    pub fn execute_and_generate_with_cached_program<SC: StarkGenericConfig>(
        mut self,
        committed_program: Arc<CommittedProgram<SC>>,
    ) -> Result<VirtualMachineResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let segments = self.execute_segments(committed_program.program.clone())?;

        Ok(VirtualMachineResult {
            per_segment: segments
                .into_iter()
                .map(|seg| {
                    seg.generate_proof_input(Some(committed_program.committed_trace_data.clone()))
                })
                .collect(),
        })
    }
}

/// A single segment VM.
pub struct SingleSegmentVM<F: PrimeField32> {
    pub config: VmConfig,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> SingleSegmentVM<F> {
    pub fn new(config: VmConfig) -> Self {
        assert_eq!(
            config.memory_config.persistence_type,
            PersistenceType::Volatile,
            "Single segment VM only supports volatile memory"
        );
        Self {
            config,
            _marker: Default::default(),
        }
    }

    /// Executes a program and returns the public values. None means the public value is not set.
    pub fn execute(
        &self,
        program: Program<F>,
        input: Vec<Vec<F>>,
    ) -> Result<Vec<Option<F>>, ExecutionError> {
        let segment = self.execute_impl(program, input.into())?;
        let pvs = if let Some(pv_chip) = segment.chip_set.public_values_chip {
            let borrowed_pv_chip = pv_chip.borrow();
            borrowed_pv_chip.core.get_custom_public_values()
        } else {
            vec![]
        };
        Ok(pvs)
    }

    /// Executes a program and returns its proof input.
    pub fn execute_and_generate<SC: StarkGenericConfig>(
        &self,
        commited_program: Arc<CommittedProgram<SC>>,
        input: Vec<Vec<F>>,
    ) -> Result<ProofInput<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let segment = self.execute_impl(commited_program.program.clone(), input.into())?;
        Ok(segment.generate_proof_input(Some(commited_program.committed_trace_data.clone())))
    }

    fn execute_impl(
        &self,
        program: Program<F>,
        input: VecDeque<Vec<F>>,
    ) -> Result<ExecutionSegment<F>, ExecutionError> {
        let pc_start = program.pc_start;
        let mut segment = ExecutionSegment::new(
            self.config.clone(),
            program,
            Arc::new(Mutex::new(Streams {
                input_stream: input,
                hint_stream: VecDeque::new(),
            })),
            None,
        );
        segment.execute_from_pc(pc_start)?;
        Ok(segment)
    }
}
