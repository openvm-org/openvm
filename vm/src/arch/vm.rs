use std::{
    borrow::Borrow, cell::RefCell, collections::VecDeque, marker::PhantomData, mem, sync::Arc,
};

use ax_stark_backend::{
    config::{Com, Domain, PcsProof, PcsProverData, StarkGenericConfig, Val},
    engine::StarkEngine,
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    p3_commit::PolynomialSpace,
    prover::types::{CommittedTraceData, Proof, ProofInput},
    verifier::VerificationError,
};
use axvm_instructions::exe::AxVmExe;
use p3_field::PrimeField32;
use parking_lot::Mutex;
use thiserror::Error;

use super::{CONNECTOR_AIR_ID, MERKLE_AIR_ID};
use crate::{
    arch::{ExecutionSegment, VmConfig},
    system::{
        connector::{VmConnectorPvs, DEFAULT_SUSPEND_EXIT_CODE},
        memory::{memory_image_to_equipartition, merkle::MemoryMerklePvs, Equipartition, CHUNK},
        program::{trace::AxVmCommittedExe, ExecutionError},
    },
};

/// VM memory state for continuations.
pub type VmMemoryState<F> = Equipartition<F, CHUNK>;

#[derive(Clone, Default, Debug)]
pub struct Streams<F> {
    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,
}

impl<F> Streams<F> {
    pub fn new(input_stream: impl Into<VecDeque<Vec<F>>>) -> Self {
        Self {
            input_stream: input_stream.into(),
            hint_stream: VecDeque::default(),
        }
    }
}

pub struct VmExecutor<F: PrimeField32> {
    pub config: VmConfig,
    _marker: PhantomData<F>,
}

#[repr(i32)]
pub enum ExitCode {
    Success = 0,
    Error = 1,
    Suspended = -1, // Continuations
}

pub struct VmExecutorResult<SC: StarkGenericConfig> {
    pub per_segment: Vec<ProofInput<SC>>,
    /// When VM is running on persistent mode, public values are stored in a special memory space.
    pub final_memory: Option<VmMemoryState<Val<SC>>>,
}

impl<F: PrimeField32> VmExecutor<F> {
    /// Create a new VM executor with a given config.
    ///
    /// The VM will start with a single segment, which is created from the initial state.
    pub fn new(config: VmConfig) -> Self {
        Self {
            config,
            _marker: Default::default(),
        }
    }

    pub fn continuation_enabled(&self) -> bool {
        self.config.continuation_enabled
    }

    pub fn execute_segments(
        &self,
        exe: impl Into<AxVmExe<F>>,
        input: impl Into<VecDeque<Vec<F>>>,
    ) -> Result<Vec<ExecutionSegment<F>>, ExecutionError> {
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();

        let exe = exe.into();
        let streams = Arc::new(Mutex::new(Streams::new(input)));
        let mut segments = vec![];
        let mut segment = ExecutionSegment::new(
            self.config.clone(),
            exe.program.clone(),
            streams.clone(),
            Some(memory_image_to_equipartition(exe.init_memory)),
            exe.fn_bounds.clone(),
        );
        let mut pc = exe.pc_start;

        loop {
            println!("Executing segment: {}", segments.len());
            let state = segment.execute_from_pc(pc)?;
            pc = state.pc;

            if state.is_terminated {
                break;
            }

            assert!(
                self.continuation_enabled(),
                "multiple segments require to enable continuations"
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
                exe.program.clone(),
                streams.clone(),
                Some(final_memory),
                exe.fn_bounds.clone(),
            );
            segment.cycle_tracker = cycle_tracker;
        }
        segments.push(segment);
        tracing::debug!("Number of continuation segments: {}", segments.len());
        #[cfg(feature = "bench-metrics")]
        metrics::gauge!("execute_time_ms").set(start.elapsed().as_millis() as f64);

        Ok(segments)
    }

    pub fn execute(
        &self,
        exe: impl Into<AxVmExe<F>>,
        input: impl Into<VecDeque<Vec<F>>>,
    ) -> Result<Option<VmMemoryState<F>>, ExecutionError> {
        let mut results = self.execute_segments(exe, input)?;
        let last = results.last_mut().unwrap();
        let final_memory = mem::take(&mut last.final_memory);
        let end_state =
            last.chip_set.connector_chip.boundary_states[1].expect("end state must be set");
        // TODO[jpw]: add these as execution errors
        assert_eq!(end_state.is_terminate, 1, "program must terminate");
        assert_eq!(
            end_state.exit_code,
            ExitCode::Success as u32,
            "program did not exit successfully"
        );
        Ok(final_memory)
    }

    pub fn execute_and_generate<SC: StarkGenericConfig>(
        &self,
        exe: impl Into<AxVmExe<F>>,
        input: impl Into<VecDeque<Vec<F>>>,
    ) -> Result<VmExecutorResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        self.execute_and_generate_impl(exe.into(), None, input.into())
    }
    pub fn execute_and_generate_with_cached_program<SC: StarkGenericConfig>(
        &self,
        commited_exe: Arc<AxVmCommittedExe<SC>>,
        input: impl Into<VecDeque<Vec<F>>>,
    ) -> Result<VmExecutorResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        self.execute_and_generate_impl(
            commited_exe.exe.clone(),
            Some(commited_exe.committed_program.clone()),
            input.into(),
        )
    }
    fn execute_and_generate_impl<SC: StarkGenericConfig>(
        &self,
        exe: AxVmExe<F>,
        committed_program: Option<CommittedTraceData<SC>>,
        input: VecDeque<Vec<F>>,
    ) -> Result<VmExecutorResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let mut segments = self.execute_segments(exe, input)?;
        let final_memory = mem::take(&mut segments.last_mut().unwrap().final_memory);

        #[allow(unused_variables)]
        Ok(VmExecutorResult {
            per_segment: segments
                .into_iter()
                .enumerate()
                .map(|(seg_idx, seg)| {
                    #[cfg(feature = "bench-metrics")]
                    let start = std::time::Instant::now();
                    let ret = seg.generate_proof_input(committed_program.clone());
                    #[cfg(feature = "bench-metrics")]
                    metrics::gauge!("execute_and_trace_gen_time_ms", "segment" => seg_idx.to_string())
                        .set(start.elapsed().as_millis() as f64);
                    ret
                })
                .collect(),
            final_memory,
        })
    }
}

/// A single segment VM.
pub struct SingleSegmentVmExecutor<F: PrimeField32> {
    pub config: VmConfig,
    _marker: PhantomData<F>,
}

/// Execution result of a single segment VM execution.
pub struct SingleSegmentVmExecutionResult<F: PrimeField32> {
    /// All user public values
    pub public_values: Vec<Option<F>>,
    /// Heights of each AIR
    pub heights: Vec<usize>,
}

impl<F: PrimeField32> SingleSegmentVmExecutor<F> {
    pub fn new(config: VmConfig) -> Self {
        assert!(
            !config.continuation_enabled,
            "Single segment VM doesn't support continuation mode"
        );
        Self {
            config,
            _marker: Default::default(),
        }
    }

    /// Executes a program and returns the public values. None means the public value is not set.
    pub fn execute(
        &self,
        exe: impl Into<AxVmExe<F>>,
        input: Vec<Vec<F>>,
    ) -> Result<SingleSegmentVmExecutionResult<F>, ExecutionError> {
        let segment = self.execute_impl(exe.into(), input.into())?;
        let heights = segment.chip_set.current_trace_heights();
        let public_values = if let Some(pv_chip) = segment.chip_set.public_values_chip {
            let borrowed_pv_chip = RefCell::borrow(&pv_chip);
            borrowed_pv_chip.core.get_custom_public_values()
        } else {
            vec![]
        };
        Ok(SingleSegmentVmExecutionResult {
            public_values,
            heights,
        })
    }

    /// Executes a program and returns its proof input.
    pub fn execute_and_generate<SC: StarkGenericConfig>(
        &self,
        commited_exe: Arc<AxVmCommittedExe<SC>>,
        input: Vec<Vec<F>>,
    ) -> Result<ProofInput<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let segment = self.execute_impl(commited_exe.exe.clone(), input.into())?;
        Ok(segment.generate_proof_input(Some(commited_exe.committed_program.clone())))
    }

    fn execute_impl(
        &self,
        exe: AxVmExe<F>,
        input: VecDeque<Vec<F>>,
    ) -> Result<ExecutionSegment<F>, ExecutionError> {
        let pc_start = exe.pc_start;
        let mut segment = ExecutionSegment::new(
            self.config.clone(),
            exe.program.clone(),
            Arc::new(Mutex::new(Streams {
                input_stream: input,
                hint_stream: VecDeque::new(),
            })),
            None,
            exe.fn_bounds,
        );
        segment.execute_from_pc(pc_start)?;
        Ok(segment)
    }
}

#[derive(Error, Debug)]
pub enum VmVerificationError {
    #[error("initial pc mismatch (initial: {initial}, prev_final: {prev_final})")]
    InitialPcMismatch { initial: u32, prev_final: u32 },

    #[error("initial memory root mismatch")]
    InitialMemoryRootMismatch,

    #[error("is terminate mismatch (expected: {expected}, actual: {actual})")]
    IsTerminateMismatch { expected: bool, actual: bool },

    #[error("exit code mismatch")]
    ExitCodeMismatch { expected: u32, actual: u32 },

    #[error("unexpected public values (expected: {expected}, actual: {actual})")]
    UnexpectedPvs { expected: usize, actual: usize },

    #[error("number of public values mismatch (expected: {expected}, actual: {actual})")]
    NumPublicValuesMismatch { expected: usize, actual: usize },

    #[error("stark verification error: {0}")]
    StarkError(#[from] VerificationError),
}

pub struct VirtualMachine<SC, E> {
    pub engine: E,
    pub config: VmConfig,
    _marker: PhantomData<SC>,
}

impl<F, SC, E> VirtualMachine<SC, E>
where
    F: PrimeField32,
    SC: StarkGenericConfig,
    E: StarkEngine<SC>,
    Domain<SC>: PolynomialSpace<Val = F>,
{
    pub fn new(engine: E, config: VmConfig) -> Self {
        Self {
            engine,
            config,
            _marker: PhantomData,
        }
    }

    pub fn keygen(&self) -> MultiStarkProvingKey<SC>
    where
        Val<SC>: PrimeField32,
    {
        self.config.generate_pk(self.engine.keygen_builder())
    }

    pub fn commit_exe(&self, exe: impl Into<AxVmExe<F>>) -> Arc<AxVmCommittedExe<SC>> {
        let exe = exe.into();
        Arc::new(AxVmCommittedExe::commit(exe, self.engine.config().pcs()))
    }

    pub fn execute(
        &self,
        exe: impl Into<AxVmExe<F>>,
        input: impl Into<VecDeque<Vec<F>>>,
    ) -> Result<Option<VmMemoryState<F>>, ExecutionError> {
        let executor = VmExecutor::new(self.config.clone());
        executor.execute(exe, input)
    }

    pub fn execute_and_generate(
        &self,
        exe: impl Into<AxVmExe<F>>,
        input: impl Into<VecDeque<Vec<F>>>,
    ) -> Result<VmExecutorResult<SC>, ExecutionError> {
        let executor = VmExecutor::new(self.config.clone());
        executor.execute_and_generate(exe, input)
    }

    pub fn execute_and_generate_with_cached_program(
        &self,
        committed_exe: Arc<AxVmCommittedExe<SC>>,
        input: impl Into<VecDeque<Vec<F>>>,
    ) -> Result<VmExecutorResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let executor = VmExecutor::new(self.config.clone());
        executor.execute_and_generate_with_cached_program(committed_exe, input)
    }

    pub fn prove_single(
        &self,
        pk: &MultiStarkProvingKey<SC>,
        proof_input: ProofInput<SC>,
    ) -> Proof<SC>
    where
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        tracing::info_span!("prove_segment", segment = 0)
            .in_scope(|| self.engine.prove(pk, proof_input))
    }

    pub fn prove(
        &self,
        pk: &MultiStarkProvingKey<SC>,
        results: VmExecutorResult<SC>,
    ) -> Vec<Proof<SC>>
    where
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        #[cfg(feature = "bench-metrics")]
        metrics::counter!("num_segments").absolute(results.per_segment.len() as u64);
        results
            .per_segment
            .into_iter()
            .enumerate()
            .map(|(seg_idx, proof_input)| {
                tracing::info_span!("prove_segment", segment = seg_idx)
                    .in_scope(|| self.engine.prove(pk, proof_input))
            })
            .collect()
    }

    pub fn verify_single(
        &self,
        vk: &MultiStarkVerifyingKey<SC>,
        proof: &Proof<SC>,
    ) -> Result<(), VerificationError> {
        self.engine.verify(vk, proof)
    }

    /// Verify segment proofs, checking continuation boundary conditions between segments if VM memory is persistent
    pub fn verify(
        &self,
        vk: &MultiStarkVerifyingKey<SC>,
        proofs: Vec<Proof<SC>>,
    ) -> Result<(), VmVerificationError>
    where
        Val<SC>: PrimeField32,
    {
        if self.config.continuation_enabled {
            self.verify_segments(vk, proofs)
        } else {
            assert_eq!(proofs.len(), 1);
            self.verify_single(vk, &proofs.into_iter().next().unwrap())
                .map_err(VmVerificationError::StarkError)
        }
    }

    /// Verify segment proofs with boundary condition checks for continuation between segments
    fn verify_segments(
        &self,
        vk: &MultiStarkVerifyingKey<SC>,
        proofs: Vec<Proof<SC>>,
    ) -> Result<(), VmVerificationError>
    where
        Val<SC>: PrimeField32,
    {
        let mut prev_final_memory_root = None;
        let mut prev_final_pc = None;

        for (i, proof) in proofs.iter().enumerate() {
            let res = self.engine.verify(vk, proof);
            match res {
                Ok(_) => (),
                Err(e) => return Err(VmVerificationError::StarkError(e)),
            };

            // Check public values.
            for air_proof_data in proof.per_air.iter() {
                let pvs = &air_proof_data.public_values;
                let air_vk = &vk.per_air[air_proof_data.air_id];

                if air_proof_data.air_id == CONNECTOR_AIR_ID {
                    let pvs: &VmConnectorPvs<_> = pvs.as_slice().borrow();

                    if i != 0 {
                        // Check initial pc matches the previous final pc.
                        if pvs.initial_pc != prev_final_pc.unwrap() {
                            return Err(VmVerificationError::InitialPcMismatch {
                                initial: pvs.initial_pc.as_canonical_u32(),
                                prev_final: prev_final_pc.unwrap().as_canonical_u32(),
                            });
                        }
                    } else {
                        // TODO: Fetch initial pc from program
                    }
                    prev_final_pc = Some(pvs.final_pc);

                    let expected_is_terminate = i == proofs.len() - 1;
                    if pvs.is_terminate != Val::<SC>::from_bool(expected_is_terminate) {
                        return Err(VmVerificationError::IsTerminateMismatch {
                            expected: expected_is_terminate,
                            actual: pvs.is_terminate.as_canonical_u32() != 0,
                        });
                    }

                    let expected_exit_code = if expected_is_terminate {
                        ExitCode::Success as u32
                    } else {
                        DEFAULT_SUSPEND_EXIT_CODE
                    };
                    if pvs.exit_code != Val::<SC>::from_canonical_u32(expected_exit_code) {
                        return Err(VmVerificationError::ExitCodeMismatch {
                            expected: expected_exit_code,
                            actual: pvs.exit_code.as_canonical_u32(),
                        });
                    }
                } else if air_proof_data.air_id == MERKLE_AIR_ID {
                    let pvs: &MemoryMerklePvs<_, CHUNK> = pvs.as_slice().borrow();

                    // Check that initial root matches the previous final root.
                    if i != 0 && pvs.initial_root != prev_final_memory_root.unwrap() {
                        return Err(VmVerificationError::InitialMemoryRootMismatch);
                    }
                    prev_final_memory_root = Some(pvs.final_root);
                } else {
                    if !pvs.is_empty() {
                        return Err(VmVerificationError::UnexpectedPvs {
                            expected: 0,
                            actual: pvs.len(),
                        });
                    }
                    if air_vk.params.num_public_values != 0 {
                        return Err(VmVerificationError::NumPublicValuesMismatch {
                            expected: 0,
                            actual: air_vk.params.num_public_values,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}
