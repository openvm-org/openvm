//! [VmExecutor] is the struct that can execute an _arbitrary_ program, provided in the form of a
//! [VmExe], for a fixed set of OpenVM instructions corresponding to a [VmExecutionConfig].
//! Internally once it is given a program, it will preprocess the program to rewrite it into a more
//! optimized format for runtime execution. This **instance** of the executor will be a separate
//! struct specialized to running a _fixed_ program on different program inputs.
//!
//! [VirtualMachine] will similarly be the struct that has done all the setup so it can
//! execute+prove an arbitrary program for a fixed config - it will internally still hold VmExecutor
use std::{
    borrow::Borrow,
    collections::{HashMap, VecDeque},
    marker::PhantomData,
    sync::Arc,
};

use getset::{Getters, Setters, WithSetters};
use itertools::{zip_eq, Itertools};
use openvm_circuit::system::program::trace::compute_exe_commit;
use openvm_instructions::exe::{SparseMemoryImage, VmExe};
use openvm_stark_backend::{
    config::{Com, StarkGenericConfig, Val},
    engine::StarkEngine,
    keygen::types::MultiStarkVerifyingKey,
    p3_field::{FieldAlgebra, FieldExtensionAlgebra, PrimeField32},
    p3_util::log2_strict_usize,
    proof::Proof,
    prover::{
        cpu::PcsData,
        hal::{DeviceDataTransporter, MatrixDimensions},
        types::{CommittedTraceData, DeviceMultiStarkProvingKey, ProvingContext},
    },
    verifier::VerificationError,
};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info_span, instrument};

use super::{
    execution_mode::e1::E1Ctx, ExecutionError, InsExecutorE1, MemoryConfig, VmChipComplex,
    CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID, PROGRAM_CACHED_TRACE_INDEX,
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
        AirInventoryError, AnyEnum, ChipInventoryError, ExecutionState, ExecutorInventory,
        ExecutorInventoryError, InstructionExecutor, SystemConfig, TraceFiller, VmExecutionConfig,
        VmProverConfig, VmSegmentExecutor, VmSegmentState, PUBLIC_VALUES_AIR_ID,
    },
    execute_spanned,
    system::{
        connector::{VmConnectorPvs, DEFAULT_SUSPEND_EXIT_CODE},
        memory::{
            adapter::records,
            merkle::{
                public_values::{UserPublicValuesProof, UserPublicValuesProofError},
                MemoryMerklePvs,
            },
            online::{GuestMemory, TracingMemory},
            AddressMap, CHUNK,
        },
        program::{trace::VmCommittedExe, ProgramHandler},
        public_values::PublicValuesStep,
        SystemChipComplex, SystemRecords, PV_EXECUTOR_IDX,
    },
};

#[derive(Error, Debug)]
pub enum GenerationError {
    #[error("unexpected number of arenas: {actual} (expected num_airs={expected})")]
    UnexpectedNumArenas { actual: usize, expected: usize },
    #[error("force_trace_heights len incorrect: {actual} (expected num_airs={expected})")]
    UnexpectedForceTraceHeightsLen { actual: usize, expected: usize },
    #[error("trace height of air {air_idx} has height {height} greater than maximum {max_height}")]
    TraceHeightsLimitExceeded {
        air_idx: usize,
        height: usize,
        max_height: usize,
    },
    #[error("trace heights violate linear constraint {constraint_idx} ({value} >= {threshold})")]
    LinearTraceHeightConstraintExceeded {
        constraint_idx: usize,
        value: u64,
        threshold: u32,
    },
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

/// [VmExecutor] is the struct that can execute an _arbitrary_ program, provided in the form of a
/// [VmExe], for a fixed set of OpenVM instructions corresponding to a [VmExecutionConfig].
/// Internally once it is given a program, it will preprocess the program to rewrite it into a more
/// optimized format for runtime execution. This **instance** of the executor will be a separate
/// struct specialized to running a _fixed_ program on different program inputs.
pub struct VmExecutor<F, VC>
where
    VC: VmExecutionConfig<F>,
{
    pub config: VC,
    /// If any executors are stateful (i.e., they mutate during execution), then the `inventory`
    /// must store the executors in their initialized state. Internally, the executors are cloned
    /// into a separate instance before running a program.
    inventory: ExecutorInventory<VC::Executor>,
    // pub overridden_heights: Option<VmComplexTraceHeights>,
    // pub trace_height_constraints: Vec<LinearConstraint>,
    // TEMPORARY: only needed for E3 arena allocation
    // pub main_widths: Vec<usize>,
    phantom: PhantomData<F>,
}

#[repr(i32)]
pub enum ExitCode {
    Success = 0,
    Error = 1,
    Suspended = -1, // Continuations
}

// TODO[jpw]: questionable struct
// pub struct VmExecutorResult<SC: StarkGenericConfig> {
//     pub per_segment: Vec<ProofInput<SC>>,
//     /// When VM is running on persistent mode, public values are stored in a special memory
// space.     pub final_memory: Option<MemoryImage>,
// }

pub struct VmState<F> {
    pub instret: u64,
    pub pc: u32,
    pub memory: GuestMemory,
    pub input: Streams<F>,
    // TODO(ayush): make generic over SeedableRng
    pub rng: StdRng,
    #[cfg(feature = "bench-metrics")]
    pub metrics: VmMetrics,
}

impl<F> VmState<F> {
    pub fn new(
        instret: u64,
        pc: u32,
        memory: GuestMemory,
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

pub struct PreflightExecutionOutput<F, RA> {
    pub system_records: SystemRecords<F>,
    pub record_arenas: Vec<RA>,
    pub to_state: VmState<F>,
}

impl<F, VC> VmExecutor<F, VC>
where
    VC: VmExecutionConfig<F>,
{
    /// Create a new VM executor with a given config.
    ///
    /// The VM will start with a single segment, which is created from the initial state.
    pub fn new(config: VC) -> Result<Self, ExecutorInventoryError> {
        let inventory = config.create_executors()?;
        Ok(Self {
            config,
            inventory,
            phantom: PhantomData,
        })
    }
}

impl<F, VC> VmExecutor<F, VC>
where
    VC: VmExecutionConfig<F> + AsRef<SystemConfig>,
{
    pub fn build_metered_ctx(
        &self,
        constant_trace_heights: &[Option<usize>],
        air_names: &[String],
        widths: &[usize],
        interactions: &[usize],
    ) -> MeteredCtx {
        let system_config = self.config.as_ref();
        let num_addr_sp = 1 + (1 << system_config.memory_config.addr_space_height);
        let mut min_block_size = vec![1; num_addr_sp];
        // TMP: hardcoding for now
        // TODO[jpw]: move to mem_config
        min_block_size[1] = 4;
        min_block_size[2] = 4;
        min_block_size[3] = 4;
        let as_byte_alignment_bits = min_block_size
            .iter()
            .map(|&x| log2_strict_usize(x as usize) as u8)
            .collect();

        let seg_strategy = &system_config.segmentation_strategy;
        let mut ctx = MeteredCtx::new(
            constant_trace_heights.to_vec(),
            system_config.has_public_values_chip(),
            system_config.continuation_enabled,
            as_byte_alignment_bits,
            system_config.memory_config.memory_dimensions(),
            air_names.to_vec(),
            widths.to_vec(),
            interactions.to_vec(),
        )
        .with_max_trace_height(seg_strategy.max_trace_height() as u32)
        .with_max_cells(seg_strategy.max_cells());
        if !system_config.continuation_enabled {
            // force single segment
            ctx.segmentation_ctx.set_segment_check_insns(u64::MAX);
        }
        ctx
    }

    pub fn create_initial_state(&self, exe: &VmExe<F>, input: impl Into<Streams<F>>) -> VmState<F> {
        let memory_config = &self.config.as_ref().memory_config;
        create_initial_state(memory_config, exe, input, 0)
    }
}

impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F> + AsRef<SystemConfig>,
    VC::Executor: Clone + InsExecutorE1<F>,
{
    /// Base E1 execution function that operates from a given state
    pub fn execute_e1_from_state(
        &self,
        exe: VmExe<F>,
        state: VmState<F>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F>, ExecutionError> {
        let instret_end = num_insns.map(|n| state.instret + n);

        let handler = ProgramHandler::new(exe.program, &self.inventory)?;
        let mut instance =
            VmSegmentExecutor::<F, VC::Executor, _>::new(handler, E1ExecutionControl);
        #[cfg(feature = "bench-metrics")]
        {
            instance.metrics = state.metrics;
            instance.set_fn_bounds(exe.fn_bounds.clone());
        }

        let ctx = E1Ctx::new(instret_end);
        let mut exec_state = VmSegmentState::new(
            state.instret,
            state.pc,
            state.memory,
            state.input,
            state.rng,
            ctx,
        );
        execute_spanned!("execute_e1", instance, &mut exec_state)?;

        if let Some(exit_code) = exec_state.exit_code {
            check_exit_code(exit_code)?;
        }
        if let Some(instret_end) = instret_end {
            assert_eq!(exec_state.instret, instret_end);
        }

        let state = VmState {
            instret: exec_state.instret,
            pc: exec_state.pc,
            memory: exec_state.memory,
            input: exec_state.streams,
            rng: exec_state.rng,
            #[cfg(feature = "bench-metrics")]
            metrics: instance.metrics.partial_take(),
        };

        Ok(state)
    }

    // TODO[jpw]: rename to just execute
    pub fn execute_e1(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F>, ExecutionError> {
        let exe = exe.into();
        let memory_config = &self.config.as_ref().memory_config;
        let state = create_initial_state(memory_config, &exe, input, 0);
        self.execute_e1_from_state(exe, state, num_insns)
    }

    /// Base metered execution function that operates from a given state
    pub fn execute_metered_from_state(
        &self,
        exe: VmExe<F>,
        state: VmState<F>,
        executor_idx_to_air_idx: &[usize],
        ctx: MeteredCtx,
    ) -> Result<Vec<Segment>, ExecutionError> {
        assert_eq!(
            executor_idx_to_air_idx.len(),
            self.inventory.executors().len()
        );
        let _span = info_span!("execute_metered").entered();

        let handler = ProgramHandler::new(exe.program, &self.inventory)?;
        let ctrl = MeteredExecutionControl::new(executor_idx_to_air_idx.to_vec());
        let mut instance = VmSegmentExecutor::<F, VC::Executor, _>::new(handler, ctrl);

        #[cfg(feature = "bench-metrics")]
        {
            instance.metrics = state.metrics;
            instance.set_fn_bounds(exe.fn_bounds.clone());
        }

        let mut exec_state = VmSegmentState::new(
            state.instret,
            state.pc,
            state.memory,
            state.input,
            state.rng,
            ctx,
        );
        execute_spanned!("execute_metered", instance, &mut exec_state)?;

        check_termination(exec_state.exit_code)?;
        let segments = exec_state.ctx.into_segments();

        tracing::debug!("Number of continuation segments: {}", segments.len());
        #[cfg(feature = "bench-metrics")]
        metrics::counter!("num_segments").absolute(segments.len() as u64);

        Ok(segments)
    }

    pub fn execute_metered(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
        executor_idx_to_air_idx: &[usize],
        ctx: MeteredCtx,
    ) -> Result<Vec<Segment>, ExecutionError> {
        let exe = exe.into();
        let state = self.create_initial_state(&exe, input);
        self.execute_metered_from_state(exe, state, executor_idx_to_air_idx, ctx)
    }
}

// /// A single segment VM.
// pub struct SingleSegmentVmExecutor<F, VC> {
//     pub config: VC,
//     pub overridden_heights: Option<VmComplexTraceHeights>,
//     pub trace_height_constraints: Vec<LinearConstraint>,
//     _marker: PhantomData<F>,
// }

// /// Execution result of a single segment VM execution.
// pub struct SingleSegmentVmExecutionResult<F> {
//     /// All user public values
//     pub public_values: Vec<Option<F>>,
//     /// Heights of each AIR, ordered by AIR ID.
//     pub air_heights: Vec<usize>,
//     /// Heights of (SystemBase, Inventory), in an internal ordering.
//     pub vm_heights: VmComplexTraceHeights,
// }

// impl<F, VC> SingleSegmentVmExecutor<F, VC>
// where
//     F: PrimeField32,
//     VC: VmExecutionConfig<F>,
//     VC::Executor: InsExecutorE1<F>,
// {
//     pub fn new(config: VC) -> Self {
//         Self::new_with_overridden_trace_heights(config, None)
//     }

//     pub fn new_with_overridden_trace_heights(
//         config: VC,
//         overridden_heights: Option<VmComplexTraceHeights>,
//     ) -> Self {
//         assert!(
//             !config.system().continuation_enabled,
//             "Single segment VM doesn't support continuation mode"
//         );
//         Self {
//             config,
//             overridden_heights,
//             trace_height_constraints: vec![],
//             _marker: Default::default(),
//         }
//     }

//     pub fn set_override_trace_heights(&mut self, overridden_heights: VmComplexTraceHeights) {
//         self.overridden_heights = Some(overridden_heights);
//     }

//     pub fn set_trace_height_constraints(&mut self, constraints: Vec<LinearConstraint>) {
//         self.trace_height_constraints = constraints;
//     }

//     // We will delete this anyways
//     // pub fn execute_metered(
//     //     &self,
//     //     exe: VmExe<F>,
//     //     input: impl Into<Streams<F>>,
//     //     widths: &[usize],
//     //     interactions: &[usize],
//     // ) -> Result<Vec<u32>, ExecutionError> {
//     //     let memory =
//     //         create_memory_image(&self.config.system().memory_config, exe.init_memory.clone());
//     //     let rng = StdRng::seed_from_u64(0);
//     //     let chip_complex =
//     //         create_and_initialize_chip_complex(&self.config, exe.program.clone(),
// None).unwrap();     //     let air_names = chip_complex.air_names();
//     //     let mut executor = VmSegmentExecutor::<F, VC, _>::new(
//     //         chip_complex,
//     //         self.trace_height_constraints.clone(),
//     //         exe.fn_bounds.clone(),
//     //         MeteredExecutionControl,
//     //     );

//     //     let has_public_values_chip = executor.chip_complex.config().has_public_values_chip();
//     //     let continuations_enabled = executor
//     //         .chip_complex
//     //         .memory_controller()
//     //         .continuation_enabled();
//     //     let as_alignment = executor
//     //         .chip_complex
//     //         .memory_controller()
//     //         .memory
//     //         .min_block_size
//     //         .iter()
//     //         .map(|&x| log2_strict_usize(x as usize) as u8)
//     //         .collect();
//     //     let constant_trace_heights = executor
//     //         .chip_complex
//     //         .constant_trace_heights()
//     //         .collect::<Vec<_>>();

//     //     let ctx = MeteredCtx::new(
//     //         constant_trace_heights,
//     //         has_public_values_chip,
//     //         continuations_enabled,
//     //         as_alignment,
//     //         self.config.system().memory_config.memory_dimensions(),
//     //         air_names,
//     //         widths.to_vec(),
//     //         interactions.to_vec(),
//     //     )
//     //     .with_segment_check_insns(u64::MAX);

//     //     let mut exec_state = VmSegmentState::new(
//     //         0,
//     //         exe.pc_start,
//     //         Some(GuestMemory::new(memory)),
//     //         input.into(),
//     //         rng,
//     //         ctx,
//     //     );
//     //     execute_spanned!("execute_metered", executor, &mut exec_state)?;

//     //     check_termination(exec_state.exit_code)?;

//     //     // Check segment count
//     //     let segments = exec_state.ctx.into_segments();
//     //     assert_eq!(
//     //         segments.len(),
//     //         1,
//     //         "Expected exactly 1 segment, but got {}",
//     //         segments.len()
//     //     );
//     //     let segment = segments.into_iter().next().unwrap();
//     //     Ok(segment.trace_heights)
//     // }

//     fn execute_impl(
//         &self,
//         exe: VmExe<F>,
//         input: impl Into<Streams<F>>,
//         trace_heights: Option<&[u32]>,
//     ) -> Result<VmSegmentExecutor<F, VC, TracegenExecutionControl>, ExecutionError> {
//         let rng = StdRng::seed_from_u64(0);
//         let chip_complex =
//             create_and_initialize_chip_complex(&self.config, exe.program.clone(), None).unwrap();

//         let mut segment = VmSegmentExecutor::new(
//             chip_complex,
//             self.trace_height_constraints.clone(),
//             exe.fn_bounds.clone(),
//             TracegenExecutionControl,
//         );

//         if let Some(overridden_heights) = self.overridden_heights.as_ref() {
//             segment.set_override_trace_heights(overridden_heights.clone());
//         }

//         let ctx = TracegenCtx::default();
//         let mut exec_state = VmSegmentState::new(0, exe.pc_start, None, input.into(), rng, ctx);
//         execute_spanned!("execute_e3", segment, &mut exec_state)?;
//         Ok(segment)
//     }

//     /// Executes a program and returns its proof input.
//     pub fn execute_and_generate<SC: StarkGenericConfig>(
//         &self,
//         committed_exe: Arc<VmCommittedExe<SC>>,
//         input: impl Into<Streams<F>>,
//         max_trace_heights: &[u32],
//     ) -> Result<ProofInput<SC>, GenerationError>
//     where
//         Domain<SC>: PolynomialSpace<Val = F>,
//         VC::Executor: Chip<SC>,
//         VC::Periphery: Chip<SC>,
//     {
//         let segment =
//             self.execute_impl(committed_exe.exe.clone(), input, Some(max_trace_heights))?;
//         let proof_input = tracing::info_span!("trace_gen").in_scope(|| {
//             segment.generate_proof_input(Some(committed_exe.committed_program.clone()))
//         })?;
//         Ok(proof_input)
//     }

//     /// Executes a program, compute the trace heights, and returns the public values.
//     pub fn execute_and_compute_heights(
//         &self,
//         exe: impl Into<VmExe<F>>,
//         input: impl Into<Streams<F>>,
//         max_trace_heights: &[u32],
//     ) -> Result<SingleSegmentVmExecutionResult<F>, ExecutionError> {
//         let executor = {
//             let mut executor =
//                 self.execute_impl(exe.into(), input.into(), Some(max_trace_heights))?;
//             executor.chip_complex.finalize_memory();
//             executor
//         };
//         let air_heights = executor.chip_complex.current_trace_heights();
//         let vm_heights = executor.chip_complex.get_internal_trace_heights();
//         let public_values = if let Some(pv_chip) = executor.chip_complex.public_values_chip() {
//             pv_chip.step.get_custom_public_values()
//         } else {
//             vec![]
//         };
//         Ok(SingleSegmentVmExecutionResult {
//             public_values,
//             air_heights,
//             vm_heights,
//         })
//     }
// }

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

#[derive(Error, Debug)]
pub enum VirtualMachineError {
    #[error("executor inventory error: {0}")]
    ExecutorInventory(#[from] ExecutorInventoryError),
    #[error("air inventory error: {0}")]
    AirInventory(#[from] AirInventoryError),
    #[error("chip inventory error: {0}")]
    ChipInventory(#[from] ChipInventoryError),
    #[error("execution error: {0}")]
    Execution(#[from] ExecutionError),
    #[error("trace generation error: {0}")]
    Generation(#[from] GenerationError),
    #[error("program committed trade data not loaded")]
    ProgramIsNotCommitted,
}

/// The [VirtualMachine] struct contains the API to generate proofs for _arbitrary_ programs for a
/// fixed set of OpenVM instructions and a fixed VM circuit corresponding to those instructions. The
/// API is specific to a particular [StarkEngine], which specifies a fixed [StarkGenericConfig] and
/// [ProverBackend] via associated types. The [VmProverConfig] also fixes the choice of
/// `RecordArena` associated to the prover backend via an associated type.
///
/// In other words, this struct _is_ the zkVM.
#[derive(Getters, Setters, WithSetters)]
pub struct VirtualMachine<E, VC>
where
    E: StarkEngine,
    VC: VmProverConfig<E::SC, E::PB>,
{
    /// Proving engine
    pub engine: E,
    /// Runtime executor
    #[getset(get = "pub")]
    executor: VmExecutor<Val<E::SC>, VC>,
    #[getset(get = "pub")]
    pk: DeviceMultiStarkProvingKey<E::PB>,
    chip_complex: VmChipComplex<E::SC, VC::RecordArena, E::PB, VC::SystemChipInventory>,
}

impl<E, VC> VirtualMachine<E, VC>
where
    E: StarkEngine,
    VC: VmProverConfig<E::SC, E::PB>,
{
    pub fn new(
        engine: E,
        config: VC,
        d_pk: DeviceMultiStarkProvingKey<E::PB>,
    ) -> Result<Self, VirtualMachineError> {
        let circuit = config.create_airs()?;
        let chip_complex = config.create_chip_complex(circuit)?;
        let executor = VmExecutor::<Val<E::SC>, VC>::new(config)?;
        Ok(Self {
            engine,
            executor,
            pk: d_pk,
            chip_complex,
        })
    }

    pub fn config(&self) -> &VC {
        &self.executor.config
    }

    // TODO[jpw]
    // pub fn new_with_overridden_trace_heights(
    //     engine: E,
    //     config: VC,
    //     overridden_heights: Option<VmComplexTraceHeights>,
    // ) -> Self {
    //     let executor = VmExecutor::new_with_overridden_trace_heights(config, overridden_heights);
    //     Self {
    //         engine,
    //         executor,
    //         _marker: PhantomData,
    //     }
    // }

    // TODO[jpw]: I'd like to make a VmInstance struct that has a loaded program
    //
    /// Preflight execution for a single segment. Executes for exactly `num_insns` instructions
    /// using an interpreter. Preflight execution must be provided with `trace_heights`
    /// instrumentation data that was collected from a previous run of metered execution so that the
    /// preflight execution knows how much memory to allocate for record arenas.
    ///
    /// This function should rarely be called on its own. Users are advised to call
    /// [`prove`](Self::prove) directly.
    pub fn execute_preflight(
        &self,
        exe: VmExe<Val<E::SC>>,
        state: VmState<Val<E::SC>>,
        num_insns: Option<u64>,
        trace_heights: &[u32],
    ) -> Result<PreflightExecutionOutput<Val<E::SC>, VC::RecordArena>, ExecutionError>
    where
        Val<E::SC>: PrimeField32,
        VC::Executor: InstructionExecutor<Val<E::SC>, VC::RecordArena>,
    {
        let handler = ProgramHandler::new(exe.program, &self.executor.inventory)?;
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        debug_assert!(executor_idx_to_air_idx
            .iter()
            .all(|&air_idx| air_idx < trace_heights.len()));
        let ctrl = TracegenExecutionControl::new(executor_idx_to_air_idx);
        let mut instance = VmSegmentExecutor::<Val<E::SC>, VC::Executor, _>::new(handler, ctrl);

        #[cfg(feature = "bench-metrics")]
        {
            instance.metrics = state.metrics;
            instance.set_fn_bounds(exe.fn_bounds.clone());
        }

        let instret_end = num_insns.map(|ni| state.instret.saturating_add(ni));
        // TODO[jpw]: figure out how to compute RA specific main_widths
        let main_widths = self
            .pk
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.main_width())
            .collect_vec();
        let capacities = zip_eq(trace_heights, main_widths)
            .map(|(&h, w)| (h as usize, w))
            .collect::<Vec<_>>();
        let ctx = TracegenCtx::new_with_capacity(&capacities, instret_end);

        let system_config: &SystemConfig = self.config().as_ref();
        let adapter_offset = system_config.access_adapter_air_id_offset();
        // ATTENTION: this must agree with `num_memory_airs`
        let num_adapters = log2_strict_usize(system_config.memory_config.max_access_adapter_n);
        assert_eq!(adapter_offset + num_adapters, system_config.num_airs());
        let access_adapter_arena_size_bound = records::arena_size_bound(
            &trace_heights[adapter_offset..adapter_offset + num_adapters],
        );
        let memory = TracingMemory::from_image(
            state.memory,
            &system_config.memory_config,
            system_config.initial_block_size(),
            access_adapter_arena_size_bound,
        );
        let from_state = ExecutionState::new(state.pc, memory.timestamp());
        let mut exec_state =
            VmSegmentState::new(state.instret, state.pc, memory, state.input, state.rng, ctx);
        execute_spanned!("execute_e3", instance, &mut exec_state)?;
        let filtered_exec_frequencies = instance.handler.filtered_execution_frequencies();
        let mut memory = exec_state.memory;
        let touched_memory = memory.finalize::<Val<E::SC>>(system_config.continuation_enabled);

        let to_state = ExecutionState::new(exec_state.pc, memory.timestamp());
        let public_values = system_config
            .has_public_values_chip()
            .then(|| {
                instance.handler.executors[PV_EXECUTOR_IDX]
                    .as_any_kind()
                    .downcast_ref::<PublicValuesStep<Val<E::SC>>>()
                    .unwrap()
                    .generate_public_values()
            })
            .unwrap_or_default();
        let system_records = SystemRecords {
            from_state,
            to_state,
            exit_code: exec_state.exit_code,
            filtered_exec_frequencies,
            access_adapter_records: memory.access_adapter_records,
            touched_memory,
            public_values,
        };
        let record_arenas = exec_state.ctx.arenas;
        let to_state = VmState {
            instret: exec_state.instret,
            pc: exec_state.pc,
            memory: memory.data,
            input: exec_state.streams,
            rng: exec_state.rng,
            #[cfg(feature = "bench-metrics")]
            metrics: Default::default(),
        };
        Ok(PreflightExecutionOutput {
            system_records,
            record_arenas,
            to_state,
        })
    }

    // @dev: This function mutates `self` but should only depend on internal state in the sense
    // that:
    // - program must already be loaded as cached trace
    // - initial memory image was already sent to device
    // - all other state should be given by `system_records` and `record_arenas`
    #[instrument(name = "tracegen", skip_all)]
    pub fn generate_proving_ctx(
        &mut self,
        system_records: SystemRecords<Val<E::SC>>,
        record_arenas: Vec<VC::RecordArena>,
    ) -> Result<ProvingContext<E::PB>, GenerationError> {
        let ctx = self
            .chip_complex
            .generate_proving_ctx(system_records, record_arenas)?;

        // Defensive checks that the trace heights satisfy the linear constraints:
        let idx_trace_heights = ctx
            .per_air
            .iter()
            .map(|(air_idx, ctx)| (*air_idx, ctx.main_trace_height()))
            .collect_vec();
        // TODO[jpw]: put back self.max_trace_height
        // if let Some(&(air_idx, height)) = idx_trace_heights
        //     .iter()
        //     .find(|(_, height)| *height > self.max_trace_height)
        // {
        //     return Err(GenerationError::TraceHeightsLimitExceeded {
        //         air_idx,
        //         height,
        //         max_height: self.max_trace_height,
        //     });
        // }
        let trace_height_constraints = &self.pk.trace_height_constraints;
        if trace_height_constraints.is_empty() {
            tracing::warn!("generating proving context without trace height constraints");
        }
        for (i, constraint) in trace_height_constraints.iter().enumerate() {
            let value = idx_trace_heights
                .iter()
                .map(|&(air_idx, h)| constraint.coefficients[air_idx] as u64 * h as u64)
                .sum::<u64>();

            if value >= constraint.threshold as u64 {
                tracing::info!(
                    "trace heights {:?} violate linear constraint {} ({} >= {})",
                    idx_trace_heights,
                    i,
                    value,
                    constraint.threshold
                );
                return Err(GenerationError::LinearTraceHeightConstraintExceeded {
                    constraint_idx: i,
                    value,
                    threshold: constraint.threshold,
                });
            }
        }

        Ok(ctx)
    }

    /// Generates proof for zkVM execution for exactly `num_insns` instructions for a given program
    /// and a given starting state.
    ///
    /// **Note**: The cached program trace must be loaded via [`load_program`](Self::load_program)
    /// before calling this function.
    ///
    /// Returns:
    /// - proof for the execution segment
    /// - final memory state only if execution ends in successful termination (exit code 0). This
    ///   final memory state may be used to extract user public values afterwards.
    pub fn prove(
        &mut self,
        exe: VmExe<Val<E::SC>>,
        state: VmState<Val<E::SC>>,
        num_insns: Option<u64>,
        trace_heights: &[u32],
    ) -> Result<(Proof<E::SC>, Option<GuestMemory>), VirtualMachineError>
    where
        Val<E::SC>: PrimeField32,
        VC::Executor: InstructionExecutor<Val<E::SC>, VC::RecordArena>,
    {
        self.transport_init_memory_to_device(&state.memory);

        let PreflightExecutionOutput {
            system_records,
            record_arenas,
            to_state,
        } = self.execute_preflight(exe, state, num_insns, trace_heights)?;
        // drop final memory unless this is a terminal segment and the exit code is success
        let final_memory =
            (system_records.exit_code == Some(ExitCode::Success as u32)).then_some(to_state.memory);
        let ctx = self.generate_proving_ctx(system_records, record_arenas)?;

        let proof = self.engine.prove(&self.pk, ctx);

        Ok((proof, final_memory))
    }

    /// Verify segment proofs, checking continuation boundary conditions between segments if VM
    /// memory is persistent The behavior of this function differs depending on whether
    /// continuations is enabled or not. We recommend to call the functions [`verify_segments`]
    /// or [`verify_single`] directly instead.
    pub fn verify(
        &self,
        vk: &MultiStarkVerifyingKey<E::SC>,
        proofs: &[Proof<E::SC>],
    ) -> Result<(), VmVerificationError>
    where
        Com<E::SC>: AsRef<[Val<E::SC>; CHUNK]> + From<[Val<E::SC>; CHUNK]>,
        Val<E::SC>: PrimeField32,
    {
        if self.config().as_ref().continuation_enabled {
            verify_segments(&self.engine, vk, proofs).map(|_| ())
        } else {
            assert_eq!(proofs.len(), 1);
            verify_single(&self.engine, vk, &proofs[0]).map_err(VmVerificationError::StarkError)
        }
    }

    pub fn commit_exe(&self, exe: impl Into<VmExe<Val<E::SC>>>) -> VmCommittedExe<E::SC> {
        let exe = exe.into();
        VmCommittedExe::commit(exe, self.engine.config().pcs())
    }

    pub fn transport_committed_exe_to_device(
        &self,
        committed_exe: &VmCommittedExe<E::SC>,
    ) -> CommittedTraceData<E::PB> {
        let commitment = committed_exe.commitment.clone();
        let trace = self
            .engine
            .device()
            .transport_matrix_to_device(&committed_exe.trace);
        let data = PcsData::new(
            committed_exe.prover_data.clone(),
            vec![log2_strict_usize(trace.height()).try_into().unwrap()],
        );
        let data = self.engine.device().transport_pcs_data_to_device(&data);
        CommittedTraceData {
            commitment,
            trace,
            data,
        }
    }

    pub fn load_program(&mut self, cached_program_trace: CommittedTraceData<E::PB>) {
        self.chip_complex.system.load_program(cached_program_trace);
    }

    pub fn transport_init_memory_to_device(&mut self, memory: &GuestMemory) {
        self.chip_complex
            .system
            .transport_init_memory_to_device(memory);
    }

    pub fn executor_idx_to_air_idx(&self) -> Vec<usize> {
        let ret = self.chip_complex.inventory.executor_idx_to_air_idx();
        tracing::debug!("executor_idx_to_air_idx: {:?}", ret);
        assert_eq!(self.executor().inventory.executors().len(), ret.len());
        ret
    }

    /// Convenience method to construct a [MeteredCtx] using data from the stored proving key.
    pub fn build_metered_ctx(&self) -> MeteredCtx {
        let (constant_trace_heights, air_names, widths, interactions): (
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
        ) = self
            .pk
            .per_air
            .iter()
            .map(|pk| {
                let constant_trace_height =
                    pk.preprocessed_data.as_ref().map(|pd| pd.trace.height());
                let air_names = pk.air_name.clone();
                let width = pk
                    .vk
                    .params
                    .width
                    .total_width(<<E::SC as StarkGenericConfig>::Challenge>::D);
                let num_interactions = pk.vk.symbolic_constraints.interactions.len();
                (constant_trace_height, air_names, width, num_interactions)
            })
            .multiunzip();

        self.executor().build_metered_ctx(
            &constant_trace_heights,
            &air_names,
            &widths,
            &interactions,
        )
    }
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

/// Prover for a specific exe in a specific continuation VM using a specific Stark config.
pub trait ContinuationVmProver<SC: StarkGenericConfig> {
    fn prove(
        &mut self,
        input: impl Into<Streams<Val<SC>>>,
    ) -> Result<ContinuationVmProof<SC>, VirtualMachineError>;
}

/// Prover for a specific exe in a specific single-segment VM using a specific Stark config.
///
/// Does not run metered execution and directly runs preflight execution. The `prove` function must
/// be provided with the expected maximum `trace_heights` to use to allocate record arena
/// capacities.
pub trait SingleSegmentVmProver<SC: StarkGenericConfig> {
    fn prove(
        &mut self,
        input: impl Into<Streams<Val<SC>>>,
        trace_heights: &[u32],
    ) -> Result<Proof<SC>, VirtualMachineError>;
}

/// Virtual machine prover instance for a fixed VM config and a fixed program. For use in proving a
/// program directly on bare metal.
pub struct VmLocalProver<E, VC>
where
    E: StarkEngine,
    VC: VmProverConfig<E::SC, E::PB>,
{
    pub vm: VirtualMachine<E, VC>,
    // TODO: store immutable parts of program handler here
    exe: VmExe<Val<E::SC>>,
}

impl<E, VC> VmLocalProver<E, VC>
where
    E: StarkEngine,
    VC: VmProverConfig<E::SC, E::PB>,
{
    pub fn new(
        mut vm: VirtualMachine<E, VC>,
        exe: VmExe<Val<E::SC>>,
        cached_program_trace: CommittedTraceData<E::PB>,
    ) -> Self {
        vm.load_program(cached_program_trace);
        Self { vm, exe }
    }
}

impl<E, VC> ContinuationVmProver<E::SC> for VmLocalProver<E, VC>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    VC: VmProverConfig<E::SC, E::PB>,
    VC::Executor: InsExecutorE1<Val<E::SC>> + InstructionExecutor<Val<E::SC>, VC::RecordArena>,
{
    /// First performs metered execution (E2) to determine segments. Then sequentially proves each
    /// segment. The proof for each segment uses the specified [ProverBackend], but the proof for
    /// the next segment does not start before the current proof finishes.
    fn prove(
        &mut self,
        input: impl Into<Streams<Val<E::SC>>>,
    ) -> Result<ContinuationVmProof<E::SC>, VirtualMachineError> {
        let input = input.into();
        let vm = &mut self.vm;
        let exe = &self.exe;
        let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
        let e2_ctx = vm.build_metered_ctx();
        let segments = vm.executor().execute_metered(
            self.exe.clone(),
            input.clone(),
            &executor_idx_to_air_idx,
            e2_ctx,
        )?;
        let mut proofs = Vec::with_capacity(segments.len());
        let mut state = Some(vm.executor().create_initial_state(exe, input));
        for (seg_idx, segment) in segments.into_iter().enumerate() {
            let _span = info_span!("prove_segment", segment = seg_idx).entered();
            let Segment {
                instret_start,
                num_insns,
                trace_heights,
            } = segment;
            assert_eq!(state.as_ref().unwrap().instret, instret_start);
            let from_state = Option::take(&mut state).unwrap();
            vm.transport_init_memory_to_device(&from_state.memory);
            let PreflightExecutionOutput {
                system_records,
                record_arenas,
                to_state,
            } = vm.execute_preflight(exe.clone(), from_state, Some(num_insns), &trace_heights)?;
            state = Some(to_state);

            let ctx = vm.generate_proving_ctx(system_records, record_arenas)?;
            let proof = vm.engine.prove(vm.pk(), ctx);
            proofs.push(proof);
        }
        let to_state = state.unwrap();
        let final_memory = to_state.memory.memory;
        let user_public_values = UserPublicValuesProof::compute(
            vm.config().as_ref().memory_config.memory_dimensions(),
            vm.config().as_ref().num_public_values,
            &vm_poseidon2_hasher(),
            &final_memory,
        );
        Ok(ContinuationVmProof {
            per_segment: proofs,
            user_public_values,
        })
    }
}

impl<E, VC> SingleSegmentVmProver<E::SC> for VmLocalProver<E, VC>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    VC: VmProverConfig<E::SC, E::PB>,
    VC::Executor: InstructionExecutor<Val<E::SC>, VC::RecordArena>,
{
    fn prove(
        &mut self,
        input: impl Into<Streams<Val<E::SC>>>,
        trace_heights: &[u32],
    ) -> Result<Proof<E::SC>, VirtualMachineError> {
        let input = input.into();
        let vm = &mut self.vm;
        let exe = &self.exe;
        assert!(!vm.config().as_ref().continuation_enabled);
        let mut trace_heights = trace_heights.to_vec();
        trace_heights[PUBLIC_VALUES_AIR_ID] = vm.config().as_ref().num_public_values as u32;
        let state = vm.executor().create_initial_state(exe, input);
        let (proof, _) = vm.prove(exe.clone(), state, None, &trace_heights)?;
        Ok(proof)
    }
}

/// Verifies a single proof. This should be used for proof of VM without continuations.
///
/// ## Note
/// This function does not check any public values or extract the starting pc or commitment
/// to the [VmCommittedExe].
pub fn verify_single<E>(
    engine: &E,
    vk: &MultiStarkVerifyingKey<E::SC>,
    proof: &Proof<E::SC>,
) -> Result<(), VerificationError>
where
    E: StarkEngine,
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
pub fn verify_segments<E>(
    engine: &E,
    vk: &MultiStarkVerifyingKey<E::SC>,
    proofs: &[Proof<E::SC>],
) -> Result<VerifiedExecutionPayload<Val<E::SC>>, VmVerificationError>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    Com<E::SC>: AsRef<[Val<E::SC>; CHUNK]>,
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

fn create_memory_image(
    memory_config: &MemoryConfig,
    init_memory: SparseMemoryImage,
) -> GuestMemory {
    GuestMemory::new(AddressMap::from_sparse(
        memory_config.addr_space_sizes.clone(),
        init_memory,
    ))
}

fn create_initial_state<F>(
    memory_config: &MemoryConfig,
    exe: &VmExe<F>,
    input: impl Into<Streams<F>>,
    seed: u64,
) -> VmState<F> {
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
