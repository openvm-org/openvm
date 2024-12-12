use std::{borrow::Borrow, collections::VecDeque, marker::PhantomData, mem, sync::Arc};

use openvm_instructions::exe::VmExe;
use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig, Val},
    engine::StarkEngine,
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    p3_commit::PolynomialSpace,
    p3_field::PrimeField32,
    prover::types::{CommittedTraceData, Proof, ProofInput},
    verifier::VerificationError,
    Chip,
};
use thiserror::Error;

use super::{ExecutionError, VmComplexTraceHeights, VmConfig, CONNECTOR_AIR_ID, MERKLE_AIR_ID};
use crate::{
    arch::segment::ExecutionSegment,
    system::{
        connector::{VmConnectorPvs, DEFAULT_SUSPEND_EXIT_CODE},
        memory::{memory_image_to_equipartition, merkle::MemoryMerklePvs, Equipartition, CHUNK},
        program::trace::VmCommittedExe,
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

impl<F> From<VecDeque<Vec<F>>> for Streams<F> {
    fn from(value: VecDeque<Vec<F>>) -> Self {
        Streams::new(value)
    }
}

impl<F> From<Vec<Vec<F>>> for Streams<F> {
    fn from(value: Vec<Vec<F>>) -> Self {
        Streams::new(value)
    }
}

pub struct VmExecutor<F, VC> {
    pub config: VC,
    pub overridden_heights: Option<VmComplexTraceHeights>,
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

impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmConfig<F>,
{
    /// Create a new VM executor with a given config.
    ///
    /// The VM will start with a single segment, which is created from the initial state.
    pub fn new(config: VC) -> Self {
        Self::new_with_overridden_trace_heights(config, None)
    }

    pub fn set_override_trace_heights(&mut self, overridden_heights: VmComplexTraceHeights) {
        self.overridden_heights = Some(overridden_heights);
    }

    pub fn new_with_overridden_trace_heights(
        config: VC,
        overridden_heights: Option<VmComplexTraceHeights>,
    ) -> Self {
        Self {
            config,
            overridden_heights,
            _marker: Default::default(),
        }
    }

    pub fn continuation_enabled(&self) -> bool {
        self.config.system().continuation_enabled
    }

    pub fn execute_segments(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
    ) -> Result<Vec<ExecutionSegment<F, VC>>, ExecutionError> {
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();

        let exe = exe.into();
        let streams = input.into();
        let mut segments = vec![];
        let mut segment = ExecutionSegment::new(
            &self.config,
            exe.program.clone(),
            streams,
            Some(memory_image_to_equipartition(exe.init_memory)),
            exe.fn_bounds.clone(),
        );
        if let Some(overridden_heights) = self.overridden_heights.as_ref() {
            segment.set_override_trace_heights(overridden_heights.clone());
        }
        let mut pc = exe.pc_start;

        loop {
            let state = tracing::info_span!("execute_segment", segment = segments.len())
                .in_scope(|| segment.execute_from_pc(pc))?;
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
                segment.chip_complex.connector_chip().boundary_states[1]
                    .unwrap()
                    .pc
            );

            let cycle_tracker = mem::take(&mut segment.cycle_tracker);
            let final_memory = mem::take(&mut segment.final_memory)
                .expect("final memory should be set in continuations segment");
            let streams = segment.chip_complex.take_streams();

            segments.push(segment);

            segment = ExecutionSegment::new(
                &self.config,
                exe.program.clone(),
                streams,
                Some(final_memory),
                exe.fn_bounds.clone(),
            );
            if let Some(overridden_heights) = self.overridden_heights.as_ref() {
                segment.set_override_trace_heights(overridden_heights.clone());
            }
            segment.cycle_tracker = cycle_tracker;
        }
        segments.push(segment);
        tracing::debug!("Number of continuation segments: {}", segments.len());
        #[cfg(feature = "bench-metrics")]
        metrics::gauge!("execute_time_ms").set(start.elapsed().as_millis() as f64);
        #[cfg(feature = "bench-metrics")]
        tracing::info!("execute_time [all segments]: {:?}", start.elapsed());

        Ok(segments)
    }

    pub fn execute(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
    ) -> Result<Option<VmMemoryState<F>>, ExecutionError> {
        let mut results = self.execute_segments(exe, input)?;
        let last = results.last_mut().unwrap();
        let final_memory = mem::take(&mut last.final_memory);
        let end_state =
            last.chip_complex.connector_chip().boundary_states[1].expect("end state must be set");
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
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
    ) -> Result<VmExecutorResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        self.execute_and_generate_impl(exe.into(), None, input)
    }

    pub fn execute_and_generate_with_cached_program<SC: StarkGenericConfig>(
        &self,
        commited_exe: Arc<VmCommittedExe<SC>>,
        input: impl Into<Streams<F>>,
    ) -> Result<VmExecutorResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        self.execute_and_generate_impl(
            commited_exe.exe.clone(),
            Some(commited_exe.committed_program.clone()),
            input,
        )
    }
    fn execute_and_generate_impl<SC: StarkGenericConfig>(
        &self,
        exe: VmExe<F>,
        committed_program: Option<CommittedTraceData<SC>>,
        input: impl Into<Streams<F>>,
    ) -> Result<VmExecutorResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let mut segments = self.execute_segments(exe, input)?;
        let final_memory = mem::take(&mut segments.last_mut().unwrap().final_memory);

        #[allow(unused_variables)]
        Ok(VmExecutorResult {
            per_segment: segments
                .into_iter()
                .enumerate()
                .map(|(seg_idx, seg)| {
                    tracing::info_span!("trace_gen", segment = seg_idx)
                        .in_scope(|| seg.generate_proof_input(committed_program.clone()))
                })
                .collect(),
            final_memory,
        })
    }
}

/// A single segment VM.
pub struct SingleSegmentVmExecutor<F, VC> {
    pub config: VC,
    pub overridden_heights: Option<VmComplexTraceHeights>,
    _marker: PhantomData<F>,
}

/// Execution result of a single segment VM execution.
pub struct SingleSegmentVmExecutionResult<F> {
    /// All user public values
    pub public_values: Vec<Option<F>>,
    /// Heights of each AIR, ordered by AIR ID.
    pub air_heights: Vec<usize>,
    /// Heights of (SystemBase, Inventory), in an internal ordering.
    pub internal_heights: VmComplexTraceHeights,
}

impl<F, VC> SingleSegmentVmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmConfig<F>,
{
    pub fn new(config: VC) -> Self {
        Self::new_with_overridden_trace_heights(config, None)
    }

    pub fn new_with_overridden_trace_heights(
        config: VC,
        overridden_heights: Option<VmComplexTraceHeights>,
    ) -> Self {
        assert!(
            !config.system().continuation_enabled,
            "Single segment VM doesn't support continuation mode"
        );
        Self {
            config,
            overridden_heights,
            _marker: Default::default(),
        }
    }

    pub fn set_override_trace_heights(&mut self, overridden_heights: VmComplexTraceHeights) {
        self.overridden_heights = Some(overridden_heights);
    }

    /// Executes a program and returns the public values. None means the public value is not set.
    pub fn execute(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
    ) -> Result<SingleSegmentVmExecutionResult<F>, ExecutionError> {
        let segment = self.execute_impl(exe.into(), input)?;
        let air_heights = segment.chip_complex.current_trace_heights();
        let internal_heights = segment.chip_complex.get_internal_trace_heights();
        let public_values = if let Some(pv_chip) = segment.chip_complex.public_values_chip() {
            pv_chip.core.get_custom_public_values()
        } else {
            vec![]
        };
        Ok(SingleSegmentVmExecutionResult {
            public_values,
            air_heights,
            internal_heights,
        })
    }

    /// Executes a program and returns its proof input.
    pub fn execute_and_generate<SC: StarkGenericConfig>(
        &self,
        commited_exe: Arc<VmCommittedExe<SC>>,
        input: impl Into<Streams<F>>,
    ) -> Result<ProofInput<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let segment = self.execute_impl(commited_exe.exe.clone(), input)?;
        let proof_input = tracing::info_span!("trace_gen").in_scope(|| {
            segment.generate_proof_input(Some(commited_exe.committed_program.clone()))
        });
        Ok(proof_input)
    }

    fn execute_impl(
        &self,
        exe: VmExe<F>,
        input: impl Into<Streams<F>>,
    ) -> Result<ExecutionSegment<F, VC>, ExecutionError> {
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();
        let pc_start = exe.pc_start;
        let mut segment = ExecutionSegment::new(
            &self.config,
            exe.program.clone(),
            input.into(),
            None,
            exe.fn_bounds,
        );
        if let Some(overridden_heights) = self.overridden_heights.as_ref() {
            segment.set_override_trace_heights(overridden_heights.clone());
        }
        segment.execute_from_pc(pc_start)?;

        #[cfg(feature = "bench-metrics")]
        metrics::gauge!("execute_time_ms").set(start.elapsed().as_millis() as f64);
        #[cfg(feature = "bench-metrics")]
        tracing::info!("execute_time [single]: {:?}", start.elapsed());

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

pub struct VirtualMachine<SC: StarkGenericConfig, E, VC> {
    /// Proving engine
    pub engine: E,
    /// Runtime executor
    pub executor: VmExecutor<Val<SC>, VC>,
    _marker: PhantomData<SC>,
}

impl<F, SC, E, VC> VirtualMachine<SC, E, VC>
where
    F: PrimeField32,
    SC: StarkGenericConfig,
    E: StarkEngine<SC>,
    Domain<SC>: PolynomialSpace<Val = F>,
    VC: VmConfig<F>,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    pub fn new(engine: E, config: VC) -> Self {
        let executor = VmExecutor::new(config);
        Self {
            engine,
            executor,
            _marker: PhantomData,
        }
    }

    pub fn new_with_overridden_trace_heights(
        engine: E,
        config: VC,
        overridden_heights: Option<VmComplexTraceHeights>,
    ) -> Self {
        let executor = VmExecutor::new_with_overridden_trace_heights(config, overridden_heights);
        Self {
            engine,
            executor,
            _marker: PhantomData,
        }
    }

    pub fn config(&self) -> &VC {
        &self.executor.config
    }

    pub fn keygen(&self) -> MultiStarkProvingKey<SC> {
        let mut keygen_builder = self.engine.keygen_builder();
        let chip_complex = self.config().create_chip_complex().unwrap();
        for air in chip_complex.airs() {
            keygen_builder.add_air(air);
        }
        keygen_builder.generate_pk()
    }

    pub fn commit_exe(&self, exe: impl Into<VmExe<F>>) -> Arc<VmCommittedExe<SC>> {
        let exe = exe.into();
        Arc::new(VmCommittedExe::commit(exe, self.engine.config().pcs()))
    }

    pub fn execute(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
    ) -> Result<Option<VmMemoryState<F>>, ExecutionError> {
        self.executor.execute(exe, input)
    }

    pub fn execute_and_generate(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
    ) -> Result<VmExecutorResult<SC>, ExecutionError> {
        self.executor.execute_and_generate(exe, input)
    }

    pub fn execute_and_generate_with_cached_program(
        &self,
        committed_exe: Arc<VmCommittedExe<SC>>,
        input: impl Into<Streams<F>>,
    ) -> Result<VmExecutorResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        self.executor
            .execute_and_generate_with_cached_program(committed_exe, input)
    }

    pub fn prove_single(
        &self,
        pk: &MultiStarkProvingKey<SC>,
        proof_input: ProofInput<SC>,
    ) -> Proof<SC> {
        self.engine.prove(pk, proof_input)
    }

    pub fn prove(
        &self,
        pk: &MultiStarkProvingKey<SC>,
        results: VmExecutorResult<SC>,
    ) -> Vec<Proof<SC>> {
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
        if self.config().system().continuation_enabled {
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
