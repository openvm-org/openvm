use std::{
    borrow::Borrow,
    collections::{HashMap, VecDeque},
    marker::PhantomData,
    sync::Arc,
};

use openvm_circuit::system::program::trace::compute_exe_commit;
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    program::Program,
};
use openvm_stark_backend::{
    config::{Com, Domain, StarkGenericConfig, Val},
    engine::StarkEngine,
    keygen::types::{LinearConstraint, MultiStarkProvingKey, MultiStarkVerifyingKey},
    p3_commit::PolynomialSpace,
    p3_field::{FieldAlgebra, PrimeField32},
    p3_util::log2_strict_usize,
    proof::Proof,
    prover::types::ProofInput,
    verifier::VerificationError,
    Chip,
};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info_span;

use super::{
    execution_mode::e1::E1Ctx, ChipId, ExecutionError, InsExecutorE1, MemoryConfig, VmChipComplex,
    VmComplexTraceHeights, VmConfig, VmInventoryError, CONNECTOR_AIR_ID, MERKLE_AIR_ID,
    PROGRAM_AIR_ID, PROGRAM_CACHED_TRACE_INDEX, PUBLIC_VALUES_AIR_ID,
};
#[cfg(feature = "bench-metrics")]
use crate::metrics::VmMetrics;
use crate::{
    arch::{
        execution_mode::{
            e1::E1ExecutionControl,
            metered::{MeteredCtx, MeteredExecutionControl, Segment},
            tracegen::{TracegenCtx, TracegenExecutionControl},
        },
        hasher::poseidon2::vm_poseidon2_hasher,
        VmSegmentExecutor, VmSegmentState,
    },
    system::{
        connector::{VmConnectorPvs, DEFAULT_SUSPEND_EXIT_CODE},
        memory::{
            merkle::{
                public_values::{UserPublicValuesProof, UserPublicValuesProofError},
                MemoryMerklePvs,
            },
            online::GuestMemory,
            AddressMap, MemoryImage, CHUNK,
        },
        program::trace::VmCommittedExe,
    },
};

#[derive(Error, Debug)]
pub enum GenerationError {
    #[error("generated trace heights violate constraints")]
    TraceHeightsLimitExceeded,
    #[error(transparent)]
    Execution(#[from] ExecutionError),
}

/// A trait for key-value store for `Streams`.
pub trait KvStore: Send + Sync {
    fn get(&self, key: &[u8]) -> Option<&[u8]>;
}

impl KvStore for HashMap<Vec<u8>, Vec<u8>> {
    fn get(&self, key: &[u8]) -> Option<&[u8]> {
        self.get(key).map(|v| v.as_slice())
    }
}

#[derive(Clone)]
pub struct Streams<F> {
    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,
    pub hint_space: Vec<Vec<F>>,
    /// The key-value store for hints. Both key and value are byte arrays. Executors which
    /// read `kv_store` need to encode the key and decode the value.
    pub kv_store: Arc<dyn KvStore>,
}

impl<F> Streams<F> {
    pub fn new(input_stream: impl Into<VecDeque<Vec<F>>>) -> Self {
        Self {
            input_stream: input_stream.into(),
            hint_stream: VecDeque::default(),
            hint_space: Vec::default(),
            kv_store: Arc::new(HashMap::new()),
        }
    }
}

impl<F> Default for Streams<F> {
    fn default() -> Self {
        Self::new(VecDeque::default())
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
    pub trace_height_constraints: Vec<LinearConstraint>,
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
    pub final_memory: Option<MemoryImage>,
}

pub struct VmState<F>
where
    F: PrimeField32,
{
    pub instret: u64,
    pub pc: u32,
    pub memory: MemoryImage,
    pub input: Streams<F>,
    // TODO(ayush): make generic over SeedableRng
    pub rng: StdRng,
    #[cfg(feature = "bench-metrics")]
    pub metrics: VmMetrics,
}

impl<F: PrimeField32> VmState<F> {
    pub fn new(
        instret: u64,
        pc: u32,
        memory: MemoryImage,
        input: impl Into<Streams<F>>,
        seed: u64,
    ) -> Self {
        Self {
            instret,
            pc,
            memory,
            input: input.into(),
            rng: StdRng::seed_from_u64(seed),
            #[cfg(feature = "bench-metrics")]
            metrics: VmMetrics::default(),
        }
    }
}

pub struct VmExecutorOneSegmentResult<F, VC>
where
    F: PrimeField32,
    VC: VmConfig<F>,
{
    pub segment: VmSegmentExecutor<F, VC, TracegenExecutionControl>,
    pub next_state: Option<VmState<F>>,
}

impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    VC::Executor: InsExecutorE1<F>,
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
            trace_height_constraints: vec![],
            _marker: Default::default(),
        }
    }

    pub fn continuation_enabled(&self) -> bool {
        self.config.system().continuation_enabled
    }

    /// Base E1 execution function that operates from a given state
    pub fn execute_e1_from_state(
        &self,
        exe: VmExe<F>,
        state: VmState<F>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F>, ExecutionError> {
        let instret_end = num_insns.map(|n| state.instret + n);

        let chip_complex =
            create_and_initialize_chip_complex(&self.config, exe.program.clone(), None, None)
                .unwrap();
        let mut segment = VmSegmentExecutor::<F, VC, _>::new(
            chip_complex,
            self.trace_height_constraints.clone(),
            exe.fn_bounds.clone(),
            E1ExecutionControl,
        );
        #[cfg(feature = "bench-metrics")]
        {
            segment.metrics = state.metrics;
        }

        let ctx = E1Ctx::new(instret_end);
        let mut exec_state = VmSegmentState::new(
            state.instret,
            state.pc,
            Some(GuestMemory::new(state.memory)),
            state.input,
            state.rng,
            ctx,
        );
        segment.execute_spanned("execute_e1", &mut exec_state)?;

        if let Some(exit_code) = exec_state.exit_code {
            check_exit_code(exit_code)?;
        }
        if let Some(instret_end) = instret_end {
            assert_eq!(exec_state.instret, instret_end);
        }

        let state = VmState {
            instret: exec_state.instret,
            pc: exec_state.pc,
            memory: exec_state.memory.unwrap().memory,
            input: exec_state.streams,
            rng: exec_state.rng,
            #[cfg(feature = "bench-metrics")]
            metrics: segment.metrics.partial_take(),
        };

        Ok(state)
    }

    pub fn execute_e1(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F>, ExecutionError> {
        let exe = exe.into();
        let state = create_initial_state(&self.config.system().memory_config, &exe, input, 0);
        self.execute_e1_from_state(exe, state, num_insns)
    }

    /// Base metered execution function that operates from a given state
    pub fn execute_metered_from_state(
        &self,
        exe: VmExe<F>,
        state: VmState<F>,
        widths: &[usize],
        interactions: &[usize],
    ) -> Result<Vec<Segment>, ExecutionError> {
        let _span = info_span!("execute_metered").entered();

        let chip_complex =
            create_and_initialize_chip_complex(&self.config, exe.program.clone(), None, None)
                .unwrap();
        let air_names = chip_complex.air_names();
        let mut executor = VmSegmentExecutor::<F, VC, _>::new(
            chip_complex,
            self.trace_height_constraints.clone(),
            exe.fn_bounds.clone(),
            MeteredExecutionControl,
        );

        #[cfg(feature = "bench-metrics")]
        {
            executor.metrics = state.metrics;
        }

        let continuations_enabled = executor
            .chip_complex
            .memory_controller()
            .continuation_enabled();
        let as_alignment = executor
            .chip_complex
            .memory_controller()
            .memory
            .min_block_size
            .iter()
            .map(|&x| log2_strict_usize(x as usize) as u8)
            .collect();
        let constant_trace_heights = executor
            .chip_complex
            .constant_trace_heights()
            .collect::<Vec<_>>();

        let ctx = MeteredCtx::new(
            constant_trace_heights,
            continuations_enabled,
            as_alignment,
            executor
                .chip_complex
                .memory_controller()
                .mem_config()
                .memory_dimensions(),
            air_names,
            widths.to_vec(),
            interactions.to_vec(),
        )
        // TODO(ayush): get rid of segmentation_strategy altogether
        .with_max_trace_height(
            self.config
                .system()
                .segmentation_strategy
                .max_trace_height() as u32,
        )
        .with_max_cells(self.config.system().segmentation_strategy.max_cells());

        let mut exec_state = VmSegmentState::new(
            state.instret,
            state.pc,
            Some(GuestMemory::new(state.memory)),
            state.input,
            state.rng,
            ctx,
        );
        executor.execute_spanned("execute_metered", &mut exec_state)?;

        check_termination(exec_state.exit_code)?;

        Ok(exec_state.ctx.into_segments())
    }

    pub fn execute_metered(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
        widths: &[usize],
        interactions: &[usize],
    ) -> Result<Vec<Segment>, ExecutionError> {
        let exe = exe.into();
        let state = create_initial_state(&self.config.system().memory_config, &exe, input, 0);
        self.execute_metered_from_state(exe, state, widths, interactions)
    }

    /// Base execution function that operates from a given state
    /// After each segment is executed, call the provided closure on the execution result.
    /// Returns the results from each closure, one per segment.
    ///
    /// The closure takes `f(segment_idx, segment) -> R`.
    pub fn execute_and_then_from_state<R, E>(
        &self,
        exe: VmExe<F>,
        mut state: VmState<F>,
        segments: &[Segment],
        mut f: impl FnMut(usize, VmSegmentExecutor<F, VC, TracegenExecutionControl>) -> Result<R, E>,
        map_err: impl Fn(ExecutionError) -> E,
    ) -> Result<Vec<R>, E> {
        // assert that segments are valid
        assert_eq!(segments.first().unwrap().instret_start, state.instret);
        for (prev, current) in segments.iter().zip(segments.iter().skip(1)) {
            assert_eq!(current.instret_start, prev.instret_start + prev.num_insns);
        }

        let mut results = Vec::new();
        for (
            segment_idx,
            Segment {
                num_insns,
                trace_heights,
                ..
            },
        ) in segments.iter().enumerate()
        {
            let _span = info_span!("execute_segment", segment = segment_idx).entered();
            let chip_complex = create_and_initialize_chip_complex(
                &self.config,
                exe.program.clone(),
                Some(state.memory),
                Some(trace_heights),
            )
            .unwrap();

            let mut segment = VmSegmentExecutor::<_, VC, _>::new(
                chip_complex,
                self.trace_height_constraints.clone(),
                exe.fn_bounds.clone(),
                TracegenExecutionControl,
            );

            #[cfg(feature = "bench-metrics")]
            {
                segment.metrics = state.metrics;
            }

            let instret_end = state.instret + num_insns;
            let ctx = TracegenCtx::new(Some(instret_end));
            let mut exec_state =
                VmSegmentState::new(state.instret, state.pc, None, state.input, state.rng, ctx);
            segment
                .execute_spanned("execute_e3", &mut exec_state)
                .map_err(&map_err)?;

            assert_eq!(
                exec_state.pc,
                segment.chip_complex.connector_chip().boundary_states[1]
                    .unwrap()
                    .pc
            );

            state = VmState {
                instret: exec_state.instret,
                pc: exec_state.pc,
                memory: segment
                    .chip_complex
                    .base
                    .memory_controller
                    .memory_image()
                    .clone(),
                input: exec_state.streams,
                rng: exec_state.rng,
                #[cfg(feature = "bench-metrics")]
                metrics: segment.metrics.partial_take(),
            };

            results.push(f(segment_idx, segment)?);
        }
        tracing::debug!("Number of continuation segments: {}", results.len());
        #[cfg(feature = "bench-metrics")]
        metrics::counter!("num_segments").absolute(results.len() as u64);

        Ok(results)
    }

    pub fn execute_and_then<R, E>(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
        segments: &[Segment],
        f: impl FnMut(usize, VmSegmentExecutor<F, VC, TracegenExecutionControl>) -> Result<R, E>,
        map_err: impl Fn(ExecutionError) -> E,
    ) -> Result<Vec<R>, E> {
        let exe = exe.into();
        let state = create_initial_state(&self.config.system().memory_config, &exe, input, 0);
        self.execute_and_then_from_state(exe, state, segments, f, map_err)
    }

    pub fn execute_from_state(
        &self,
        exe: VmExe<F>,
        state: VmState<F>,
        segments: &[Segment],
    ) -> Result<Option<MemoryImage>, ExecutionError> {
        let executors =
            self.execute_and_then_from_state(exe, state, segments, |_, seg| Ok(seg), |err| err)?;
        let last = executors
            .last()
            .expect("at least one segment must be executed");
        let final_memory = Some(
            last.chip_complex
                .base
                .memory_controller
                .memory_image()
                .clone(),
        );
        let end_state =
            last.chip_complex.connector_chip().boundary_states[1].expect("end state must be set");
        if end_state.is_terminate != 1 {
            return Err(ExecutionError::DidNotTerminate);
        }
        check_exit_code(end_state.exit_code)?;
        Ok(final_memory)
    }

    pub fn execute(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
        segments: &[Segment],
    ) -> Result<Option<MemoryImage>, ExecutionError> {
        let exe = exe.into();
        let state = create_initial_state(&self.config.system().memory_config, &exe, input, 0);
        self.execute_from_state(exe, state, segments)
    }

    // TODO(ayush): this is required in dummy keygen because it expects heights
    //              in VmComplexTraceHeights format. should be removed later
    pub fn execute_segments(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
        segments: &[Segment],
    ) -> Result<Vec<VmSegmentExecutor<F, VC, TracegenExecutionControl>>, ExecutionError> {
        self.execute_and_then(exe, input, segments, |_, seg| Ok(seg), |err| err)
    }

    pub fn execute_from_state_and_generate<SC>(
        &self,
        exe: VmExe<F>,
        state: VmState<F>,
        segments: &[Segment],
    ) -> Result<VmExecutorResult<SC>, GenerationError>
    where
        SC: StarkGenericConfig,
        Domain<SC>: PolynomialSpace<Val = F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let mut final_memory = None;
        let per_segment = self.execute_and_then_from_state(
            exe,
            state,
            segments,
            |seg_idx, seg| {
                final_memory = Some(seg.chip_complex.memory_controller().memory_image().clone());
                tracing::info_span!("trace_gen", segment = seg_idx)
                    .in_scope(|| seg.generate_proof_input(None))
            },
            GenerationError::Execution,
        )?;

        Ok(VmExecutorResult {
            per_segment,
            final_memory,
        })
    }

    pub fn execute_and_generate<SC>(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
        segments: &[Segment],
    ) -> Result<VmExecutorResult<SC>, GenerationError>
    where
        SC: StarkGenericConfig,
        Domain<SC>: PolynomialSpace<Val = F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let exe = exe.into();
        let state = create_initial_state(&self.config.system().memory_config, &exe, input, 0);
        self.execute_from_state_and_generate(exe, state, segments)
    }

    pub fn execute_and_generate_with_cached_program<SC: StarkGenericConfig>(
        &self,
        committed_exe: Arc<VmCommittedExe<SC>>,
        input: impl Into<Streams<F>>,
        segments: &[Segment],
    ) -> Result<VmExecutorResult<SC>, GenerationError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
        VC::Executor: Chip<SC> + InsExecutorE1<F>,
        VC::Periphery: Chip<SC>,
    {
        let mut final_memory = None;
        let per_segment = self.execute_and_then(
            committed_exe.exe.clone(),
            input,
            segments,
            |seg_idx, seg| {
                final_memory = Some(seg.chip_complex.memory_controller().memory_image().clone());
                tracing::info_span!("trace_gen", segment = seg_idx).in_scope(|| {
                    seg.generate_proof_input(Some(committed_exe.committed_program.clone()))
                })
            },
            GenerationError::Execution,
        )?;

        Ok(VmExecutorResult {
            per_segment,
            final_memory,
        })
    }

    pub fn set_trace_height_constraints(&mut self, constraints: Vec<LinearConstraint>) {
        self.trace_height_constraints = constraints;
    }
}

/// A single segment VM.
pub struct SingleSegmentVmExecutor<F, VC> {
    pub config: VC,
    pub overridden_heights: Option<VmComplexTraceHeights>,
    pub trace_height_constraints: Vec<LinearConstraint>,
    _marker: PhantomData<F>,
}

/// Execution result of a single segment VM execution.
pub struct SingleSegmentVmExecutionResult<F> {
    /// All user public values
    pub public_values: Vec<Option<F>>,
    /// Heights of each AIR, ordered by AIR ID.
    pub air_heights: Vec<usize>,
    /// Heights of (SystemBase, Inventory), in an internal ordering.
    pub vm_heights: VmComplexTraceHeights,
}

impl<F, VC> SingleSegmentVmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    VC::Executor: InsExecutorE1<F>,
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
            trace_height_constraints: vec![],
            _marker: Default::default(),
        }
    }

    pub fn set_override_trace_heights(&mut self, overridden_heights: VmComplexTraceHeights) {
        self.overridden_heights = Some(overridden_heights);
    }

    pub fn set_trace_height_constraints(&mut self, constraints: Vec<LinearConstraint>) {
        self.trace_height_constraints = constraints;
    }

    pub fn execute_e1(
        &self,
        exe: VmExe<F>,
        input: impl Into<Streams<F>>,
    ) -> Result<(), ExecutionError> {
        let memory =
            create_memory_image(&self.config.system().memory_config, exe.init_memory.clone());
        let rng = StdRng::seed_from_u64(0);
        let chip_complex =
            create_and_initialize_chip_complex(&self.config, exe.program.clone(), None, None)
                .unwrap();
        let mut executor = VmSegmentExecutor::<F, VC, _>::new(
            chip_complex,
            self.trace_height_constraints.clone(),
            exe.fn_bounds.clone(),
            E1ExecutionControl,
        );

        let ctx = E1Ctx::default();

        let mut exec_state = VmSegmentState::new(
            0,
            exe.pc_start,
            Some(GuestMemory::new(memory)),
            input.into(),
            rng,
            ctx,
        );
        executor.execute_spanned("execute_e1", &mut exec_state)?;

        check_termination(exec_state.exit_code)?;

        Ok(())
    }

    pub fn execute_metered(
        &self,
        exe: VmExe<F>,
        input: impl Into<Streams<F>>,
        widths: &[usize],
        interactions: &[usize],
    ) -> Result<Vec<u32>, ExecutionError> {
        let memory =
            create_memory_image(&self.config.system().memory_config, exe.init_memory.clone());
        let rng = StdRng::seed_from_u64(0);
        let chip_complex =
            create_and_initialize_chip_complex(&self.config, exe.program.clone(), None, None)
                .unwrap();
        let air_names = chip_complex.air_names();
        let mut executor = VmSegmentExecutor::<F, VC, _>::new(
            chip_complex,
            self.trace_height_constraints.clone(),
            exe.fn_bounds.clone(),
            MeteredExecutionControl,
        );

        let continuations_enabled = executor
            .chip_complex
            .memory_controller()
            .continuation_enabled();
        let as_alignment = executor
            .chip_complex
            .memory_controller()
            .memory
            .min_block_size
            .iter()
            .map(|&x| log2_strict_usize(x as usize) as u8)
            .collect();
        let constant_trace_heights = executor
            .chip_complex
            .constant_trace_heights()
            .collect::<Vec<_>>();

        let ctx = MeteredCtx::new(
            constant_trace_heights,
            continuations_enabled,
            as_alignment,
            self.config.system().memory_config.memory_dimensions(),
            air_names,
            widths.to_vec(),
            interactions.to_vec(),
        )
        .with_max_trace_height(
            self.config
                .system()
                .segmentation_strategy
                .max_trace_height() as u32,
        )
        .with_max_cells(self.config.system().segmentation_strategy.max_cells());

        let mut exec_state = VmSegmentState::new(
            0,
            exe.pc_start,
            Some(GuestMemory::new(memory)),
            input.into(),
            rng,
            ctx,
        );
        executor.execute_spanned("execute_metered", &mut exec_state)?;

        check_termination(exec_state.exit_code)?;

        // Check segment count
        let segments = exec_state.ctx.into_segments();
        assert_eq!(
            segments.len(),
            1,
            "Expected exactly 1 segment, but got {}",
            segments.len()
        );
        let segment = segments.into_iter().next().unwrap();
        Ok(segment.trace_heights)
    }

    fn execute_impl(
        &self,
        exe: VmExe<F>,
        input: impl Into<Streams<F>>,
        trace_heights: Option<&[u32]>,
    ) -> Result<VmSegmentExecutor<F, VC, TracegenExecutionControl>, ExecutionError> {
        let rng = StdRng::seed_from_u64(0);
        let chip_complex = create_and_initialize_chip_complex(
            &self.config,
            exe.program.clone(),
            None,
            trace_heights,
        )
        .unwrap();

        let mut segment = VmSegmentExecutor::new(
            chip_complex,
            self.trace_height_constraints.clone(),
            exe.fn_bounds.clone(),
            TracegenExecutionControl,
        );

        if let Some(overridden_heights) = self.overridden_heights.as_ref() {
            segment.set_override_trace_heights(overridden_heights.clone());
        }

        let ctx = TracegenCtx::default();
        let mut exec_state = VmSegmentState::new(0, exe.pc_start, None, input.into(), rng, ctx);
        segment.execute_spanned("execute_e3", &mut exec_state)?;
        Ok(segment)
    }

    /// Executes a program and returns its proof input.
    pub fn execute_and_generate<SC: StarkGenericConfig>(
        &self,
        committed_exe: Arc<VmCommittedExe<SC>>,
        input: impl Into<Streams<F>>,
        max_trace_heights: &[u32],
    ) -> Result<ProofInput<SC>, GenerationError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let segment =
            self.execute_impl(committed_exe.exe.clone(), input, Some(max_trace_heights))?;
        let proof_input = tracing::info_span!("trace_gen").in_scope(|| {
            segment.generate_proof_input(Some(committed_exe.committed_program.clone()))
        })?;
        Ok(proof_input)
    }

    /// Executes a program, compute the trace heights, and returns the public values.
    pub fn execute_and_compute_heights(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
        max_trace_heights: &[u32],
    ) -> Result<SingleSegmentVmExecutionResult<F>, ExecutionError> {
        let executor = {
            let mut executor =
                self.execute_impl(exe.into(), input.into(), Some(max_trace_heights))?;
            executor.chip_complex.finalize_memory();
            executor
        };
        let air_heights = executor.chip_complex.current_trace_heights();
        let vm_heights = executor.chip_complex.get_internal_trace_heights();
        let public_values = if let Some(pv_chip) = executor.chip_complex.public_values_chip() {
            pv_chip.step.get_custom_public_values()
        } else {
            vec![]
        };
        Ok(SingleSegmentVmExecutionResult {
            public_values,
            air_heights,
            vm_heights,
        })
    }
}

#[derive(Error, Debug)]
pub enum VmVerificationError {
    #[error("no proof is provided")]
    ProofNotFound,

    #[error("program commit mismatch (index of mismatch proof: {index}")]
    ProgramCommitMismatch { index: usize },

    #[error("initial pc mismatch (initial: {initial}, prev_final: {prev_final})")]
    InitialPcMismatch { initial: u32, prev_final: u32 },

    #[error("initial memory root mismatch")]
    InitialMemoryRootMismatch,

    #[error("is terminate mismatch (expected: {expected}, actual: {actual})")]
    IsTerminateMismatch { expected: bool, actual: bool },

    #[error("exit code mismatch")]
    ExitCodeMismatch { expected: u32, actual: u32 },

    #[error("AIR has unexpected public values (expected: {expected}, actual: {actual})")]
    UnexpectedPvs { expected: usize, actual: usize },

    #[error("missing system AIR with ID {air_id}")]
    SystemAirMissing { air_id: usize },

    #[error("stark verification error: {0}")]
    StarkError(#[from] VerificationError),

    #[error("user public values proof error: {0}")]
    UserPublicValuesError(#[from] UserPublicValuesProofError),
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
    VC::Executor: Chip<SC> + InsExecutorE1<F>,
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

    pub fn set_trace_height_constraints(
        &mut self,
        trace_height_constraints: Vec<LinearConstraint>,
    ) {
        self.executor
            .set_trace_height_constraints(trace_height_constraints);
    }

    pub fn commit_exe(&self, exe: impl Into<VmExe<F>>) -> Arc<VmCommittedExe<SC>> {
        let exe = exe.into();
        Arc::new(VmCommittedExe::commit(exe, self.engine.config().pcs()))
    }

    pub fn execute_metered(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
        widths: &[usize],
        interactions: &[usize],
    ) -> Result<Vec<Segment>, ExecutionError> {
        self.executor
            .execute_metered(exe, input, widths, interactions)
    }

    pub fn execute(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
        segments: &[Segment],
    ) -> Result<Option<MemoryImage>, ExecutionError> {
        self.executor.execute(exe, input, segments)
    }

    pub fn execute_and_generate(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
        segments: &[Segment],
    ) -> Result<VmExecutorResult<SC>, GenerationError> {
        self.executor.execute_and_generate(exe, input, segments)
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

    /// Verify segment proofs, checking continuation boundary conditions between segments if VM
    /// memory is persistent The behavior of this function differs depending on whether
    /// continuations is enabled or not. We recommend to call the functions [`verify_segments`]
    /// or [`verify_single`] directly instead.
    pub fn verify(
        &self,
        vk: &MultiStarkVerifyingKey<SC>,
        proofs: Vec<Proof<SC>>,
    ) -> Result<(), VmVerificationError>
    where
        Val<SC>: PrimeField32,
        Com<SC>: AsRef<[Val<SC>; CHUNK]> + From<[Val<SC>; CHUNK]>,
    {
        if self.config().system().continuation_enabled {
            verify_segments(&self.engine, vk, &proofs).map(|_| ())
        } else {
            assert_eq!(proofs.len(), 1);
            verify_single(&self.engine, vk, &proofs.into_iter().next().unwrap())
                .map_err(VmVerificationError::StarkError)
        }
    }
}

/// Verifies a single proof. This should be used for proof of VM without continuations.
///
/// ## Note
/// This function does not check any public values or extract the starting pc or commitment
/// to the [VmCommittedExe].
pub fn verify_single<SC, E>(
    engine: &E,
    vk: &MultiStarkVerifyingKey<SC>,
    proof: &Proof<SC>,
) -> Result<(), VerificationError>
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC>,
{
    engine.verify(vk, proof)
}

/// The payload of a verified guest VM execution.
pub struct VerifiedExecutionPayload<F> {
    /// The Merklelized hash of:
    /// - Program code commitment (commitment of the cached trace)
    /// - Merkle root of the initial memory
    /// - Starting program counter (`pc_start`)
    ///
    /// The Merklelization uses Poseidon2 as a cryptographic hash function (for the leaves)
    /// and a cryptographic compression function (for internal nodes).
    pub exe_commit: [F; CHUNK],
    /// The Merkle root of the final memory state.
    pub final_memory_root: [F; CHUNK],
}

/// Verify segment proofs with boundary condition checks for continuation between segments.
///
/// Assumption:
/// - `vk` is a valid verifying key of a VM circuit.
///
/// Returns:
/// - The commitment to the [VmCommittedExe] extracted from `proofs`. It is the responsibility of
///   the caller to check that the returned commitment matches the VM executable that the VM was
///   supposed to execute.
/// - The Merkle root of the final memory state.
///
/// ## Note
/// This function does not extract or verify any user public values from the final memory state.
/// This verification requires an additional Merkle proof with respect to the Merkle root of
/// the final memory state.
// @dev: This function doesn't need to be generic in `VC`.
pub fn verify_segments<SC, E>(
    engine: &E,
    vk: &MultiStarkVerifyingKey<SC>,
    proofs: &[Proof<SC>],
) -> Result<VerifiedExecutionPayload<Val<SC>>, VmVerificationError>
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC>,
    Val<SC>: PrimeField32,
    Com<SC>: AsRef<[Val<SC>; CHUNK]>,
{
    if proofs.is_empty() {
        return Err(VmVerificationError::ProofNotFound);
    }
    let mut prev_final_memory_root = None;
    let mut prev_final_pc = None;
    let mut start_pc = None;
    let mut initial_memory_root = None;
    let mut program_commit = None;

    for (i, proof) in proofs.iter().enumerate() {
        let res = engine.verify(vk, proof);
        match res {
            Ok(_) => (),
            Err(e) => return Err(VmVerificationError::StarkError(e)),
        };

        let mut program_air_present = false;
        let mut connector_air_present = false;
        let mut merkle_air_present = false;

        // Check public values.
        for air_proof_data in proof.per_air.iter() {
            let pvs = &air_proof_data.public_values;
            let air_vk = &vk.inner.per_air[air_proof_data.air_id];
            if air_proof_data.air_id == PROGRAM_AIR_ID {
                program_air_present = true;
                if i == 0 {
                    program_commit =
                        Some(proof.commitments.main_trace[PROGRAM_CACHED_TRACE_INDEX].as_ref());
                } else if program_commit.unwrap()
                    != proof.commitments.main_trace[PROGRAM_CACHED_TRACE_INDEX].as_ref()
                {
                    return Err(VmVerificationError::ProgramCommitMismatch { index: i });
                }
            } else if air_proof_data.air_id == CONNECTOR_AIR_ID {
                connector_air_present = true;
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
                    start_pc = Some(pvs.initial_pc);
                }
                prev_final_pc = Some(pvs.final_pc);

                let expected_is_terminate = i == proofs.len() - 1;
                if pvs.is_terminate != FieldAlgebra::from_bool(expected_is_terminate) {
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
                if pvs.exit_code != FieldAlgebra::from_canonical_u32(expected_exit_code) {
                    return Err(VmVerificationError::ExitCodeMismatch {
                        expected: expected_exit_code,
                        actual: pvs.exit_code.as_canonical_u32(),
                    });
                }
            } else if air_proof_data.air_id == MERKLE_AIR_ID {
                merkle_air_present = true;
                let pvs: &MemoryMerklePvs<_, CHUNK> = pvs.as_slice().borrow();

                // Check that initial root matches the previous final root.
                if i != 0 {
                    if pvs.initial_root != prev_final_memory_root.unwrap() {
                        return Err(VmVerificationError::InitialMemoryRootMismatch);
                    }
                } else {
                    initial_memory_root = Some(pvs.initial_root);
                }
                prev_final_memory_root = Some(pvs.final_root);
            } else {
                if !pvs.is_empty() {
                    return Err(VmVerificationError::UnexpectedPvs {
                        expected: 0,
                        actual: pvs.len(),
                    });
                }
                // We assume the vk is valid, so this is only a debug assert.
                debug_assert_eq!(air_vk.params.num_public_values, 0);
            }
        }
        if !program_air_present {
            return Err(VmVerificationError::SystemAirMissing {
                air_id: PROGRAM_AIR_ID,
            });
        }
        if !connector_air_present {
            return Err(VmVerificationError::SystemAirMissing {
                air_id: CONNECTOR_AIR_ID,
            });
        }
        if !merkle_air_present {
            return Err(VmVerificationError::SystemAirMissing {
                air_id: MERKLE_AIR_ID,
            });
        }
    }
    let exe_commit = compute_exe_commit(
        &vm_poseidon2_hasher(),
        program_commit.unwrap(),
        initial_memory_root.as_ref().unwrap(),
        start_pc.unwrap(),
    );
    Ok(VerifiedExecutionPayload {
        exe_commit,
        final_memory_root: prev_final_memory_root.unwrap(),
    })
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Com<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>"
))]
pub struct ContinuationVmProof<SC: StarkGenericConfig> {
    pub per_segment: Vec<Proof<SC>>,
    pub user_public_values: UserPublicValuesProof<{ CHUNK }, Val<SC>>,
}

impl<SC: StarkGenericConfig> Clone for ContinuationVmProof<SC>
where
    Com<SC>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            per_segment: self.per_segment.clone(),
            user_public_values: self.user_public_values.clone(),
        }
    }
}

pub fn create_memory_image(
    memory_config: &MemoryConfig,
    init_memory: SparseMemoryImage,
) -> MemoryImage {
    AddressMap::from_sparse(memory_config.addr_space_sizes.clone(), init_memory)
}

pub fn create_initial_state<F>(
    memory_config: &MemoryConfig,
    exe: &VmExe<F>,
    input: impl Into<Streams<F>>,
    seed: u64,
) -> VmState<F>
where
    F: PrimeField32,
{
    let memory = create_memory_image(memory_config, exe.init_memory.clone());
    #[cfg(feature = "bench-metrics")]
    let mut state = VmState::new(0, exe.pc_start, memory, input, seed);
    #[cfg(not(feature = "bench-metrics"))]
    let state = VmState::new(0, exe.pc_start, memory, input, seed);
    #[cfg(feature = "bench-metrics")]
    {
        state.metrics.fn_bounds = exe.fn_bounds.clone();
    }
    state
}

/// Create and initialize a chip complex with program, streams, optional memory, and optional trace
/// heights
pub fn create_and_initialize_chip_complex<F, VC>(
    config: &VC,
    program: Program<F>,
    initial_memory: Option<MemoryImage>,
    max_trace_heights: Option<&[u32]>,
) -> Result<VmChipComplex<F, VC::Executor, VC::Periphery>, VmInventoryError>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    VC::Executor: InsExecutorE1<F>,
{
    let mut chip_complex = config.create_chip_complex()?;

    // Strip debug info if profiling is disabled
    let program = if !config.system().profiling {
        program.strip_debug_infos()
    } else {
        program
    };

    chip_complex.set_program(program);

    if let Some(initial_memory) = initial_memory {
        chip_complex.set_initial_memory(initial_memory);
    }

    if let Some(max_trace_heights) = max_trace_heights {
        let executor_chip_offset = if chip_complex.config().has_public_values_chip() {
            PUBLIC_VALUES_AIR_ID + 1 + chip_complex.memory_controller().num_airs()
        } else {
            PUBLIC_VALUES_AIR_ID + chip_complex.memory_controller().num_airs()
        };

        for (i, chip_id) in chip_complex
            .inventory
            .insertion_order
            .iter()
            .rev()
            .enumerate()
        {
            if let ChipId::Executor(exec_id) = chip_id {
                if let Some(height_index) = executor_chip_offset.checked_add(i) {
                    if let Some(&height) = max_trace_heights.get(height_index) {
                        if let Some(executor) = chip_complex.inventory.executors.get_mut(*exec_id) {
                            // TODO(ayush): remove conversion
                            executor.set_trace_height(height.next_power_of_two() as usize);
                        }
                    }
                }
            }
        }
    }

    Ok(chip_complex)
}

fn check_exit_code(exit_code: u32) -> Result<(), ExecutionError> {
    if exit_code != ExitCode::Success as u32 {
        return Err(ExecutionError::FailedWithExitCode(exit_code));
    }
    Ok(())
}

fn check_termination(exit_code: Option<u32>) -> Result<(), ExecutionError> {
    match exit_code {
        Some(code) => check_exit_code(code),
        None => Err(ExecutionError::DidNotTerminate),
    }
}
