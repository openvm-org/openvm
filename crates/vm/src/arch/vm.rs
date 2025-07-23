//! [VmExecutor] is the struct that can execute an _arbitrary_ program, provided in the form of a
//! [VmExe], for a fixed set of OpenVM instructions corresponding to a [VmExecutionConfig].
//! Internally once it is given a program, it will preprocess the program to rewrite it into a more
//! optimized format for runtime execution. This **instance** of the executor will be a separate
//! struct specialized to running a _fixed_ program on different program inputs.
//!
//! [VirtualMachine] will similarly be the struct that has done all the setup so it can
//! execute+prove an arbitrary program for a fixed config - it will internally still hold VmExecutor
use std::{
    any::TypeId,
    borrow::Borrow,
    collections::{HashMap, VecDeque},
    marker::PhantomData,
    sync::Arc,
};

use getset::{Getters, MutGetters, Setters, WithSetters};
use itertools::{zip_eq, Itertools};
use openvm_circuit::system::program::trace::compute_exe_commit;
use openvm_instructions::exe::{SparseMemoryImage, VmExe};
use openvm_stark_backend::{
    config::{Com, StarkGenericConfig, Val},
    engine::StarkEngine,
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    p3_field::{FieldAlgebra, FieldExtensionAlgebra, PrimeField32, TwoAdicField},
    p3_util::{log2_ceil_usize, log2_strict_usize},
    proof::Proof,
    prover::{
        hal::{DeviceDataTransporter, MatrixDimensions},
        types::{CommittedTraceData, DeviceMultiStarkProvingKey, ProvingContext},
    },
    verifier::VerificationError,
};
use p3_baby_bear::BabyBear;
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
            metered::{MeteredCtx, Segment},
            tracegen::{TracegenCtx, TracegenExecutionControl},
        },
        hasher::poseidon2::vm_poseidon2_hasher,
        interpreter::InterpretedInstance,
        AirInventoryError, AnyEnum, ChipInventoryError, ExecutionState, ExecutorInventory,
        ExecutorInventoryError, InsExecutorE2, InstructionExecutor, SystemConfig, TraceFiller,
        VmBuilder, VmCircuitConfig, VmExecutionConfig, VmSegmentExecutor, VmSegmentState,
        PUBLIC_VALUES_AIR_ID,
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
        SystemChipComplex, SystemRecords, SystemWithFixedTraceHeights, PV_EXECUTOR_IDX,
    },
};

#[derive(Error, Debug)]
pub enum GenerationError {
    #[error("unexpected number of arenas: {actual} (expected num_airs={expected})")]
    UnexpectedNumArenas { actual: usize, expected: usize },
    #[error("trace height for air_idx={air_idx} must be fixed to {expected}, actual={actual}")]
    ForceTraceHeightIncorrect {
        air_idx: usize,
        actual: usize,
        expected: usize,
    },
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
    phantom: PhantomData<F>,
}

#[repr(i32)]
pub enum ExitCode {
    Success = 0,
    Error = 1,
    Suspended = -1, // Continuations
}

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
        MeteredCtx::new(
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
        .with_max_cells(seg_strategy.max_cells())
    }

    pub fn create_initial_state(&self, exe: &VmExe<F>, input: impl Into<Streams<F>>) -> VmState<F> {
        let memory_config = &self.config.as_ref().memory_config;
        create_initial_state(memory_config, exe, input, 0)
    }
}

impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F> + AsRef<SystemConfig> + Clone,
    VC::Executor: Clone + InsExecutorE1<F> + InsExecutorE2<F>,
{
    // /// Base E1 execution function that operates from a given state
    // pub fn execute_e1_from_state(
    //     &self,
    //     exe: VmExe<F>,
    //     state: VmState<F>,
    //     num_insns: Option<u64>,
    // ) -> Result<VmState<F>, ExecutionError> {
    //     let instret_end = num_insns.map(|n| state.instret + n);

    //     let handler = ProgramHandler::new(exe.program, &self.inventory)?;
    //     let mut instance =
    //         VmSegmentExecutor::<F, VC::Executor, _>::new(handler, E1ExecutionControl);
    //     #[cfg(feature = "bench-metrics")]
    //     {
    //         instance.metrics = state.metrics;
    //         instance.set_fn_bounds(exe.fn_bounds.clone());
    //     }

    //     let ctx = E1Ctx::new(instret_end);
    //     let mut exec_state = VmSegmentState::new(
    //         state.instret,
    //         state.pc,
    //         state.memory,
    //         state.input,
    //         state.rng,
    //         ctx,
    //     );
    //     execute_spanned!("execute_e1", instance, &mut exec_state)?;

    //     if let Some(exit_code) = exec_state.exit_code {
    //         check_exit_code(exit_code)?;
    //     }
    //     if let Some(instret_end) = instret_end {
    //         assert_eq!(exec_state.instret, instret_end);
    //     }

    //     let state = VmState {
    //         instret: exec_state.instret,
    //         pc: exec_state.pc,
    //         memory: exec_state.memory,
    //         input: exec_state.streams,
    //         rng: exec_state.rng,
    //         #[cfg(feature = "bench-metrics")]
    //         metrics: instance.metrics.partial_take(),
    //     };

    //     Ok(state)
    // }

    // TODO[jpw]: rename to just execute
    pub fn execute_e1(
        &self,
        exe: impl Into<VmExe<F>>,
        inputs: impl Into<Streams<F>>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F>, ExecutionError> {
        let interpreter = InterpretedInstance::new(self.config.clone(), exe)?;

        let ctx = E1Ctx::new(num_insns);
        let state = interpreter.execute(ctx, inputs)?;

        Ok(VmState {
            instret: state.instret,
            pc: state.pc,
            memory: state.memory,
            input: state.streams,
            rng: state.rng,
            #[cfg(feature = "bench-metrics")]
            metrics: VmMetrics::default(),
        })
    }

    pub fn execute_metered(
        &self,
        exe: impl Into<VmExe<F>>,
        input: impl Into<Streams<F>>,
        executor_idx_to_air_idx: &[usize],
        ctx: MeteredCtx,
    ) -> Result<Vec<Segment>, ExecutionError> {
        let interpreter = InterpretedInstance::new(self.config.clone(), exe)?;

        let state = interpreter.execute_e2(ctx, input, executor_idx_to_air_idx)?;
        check_termination(state.exit_code)?;

        Ok(state.ctx.into_segments())
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
    #[error("verification error: {0}")]
    Verification(#[from] VmVerificationError),
}

/// The [VirtualMachine] struct contains the API to generate proofs for _arbitrary_ programs for a
/// fixed set of OpenVM instructions and a fixed VM circuit corresponding to those instructions. The
/// API is specific to a particular [StarkEngine], which specifies a fixed [StarkGenericConfig] and
/// [ProverBackend] via associated types. The [VmProverBuilder] also fixes the choice of
/// `RecordArena` associated to the prover backend via an associated type.
///
/// In other words, this struct _is_ the zkVM.
#[derive(Getters, MutGetters, Setters, WithSetters)]
pub struct VirtualMachine<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    /// Proving engine
    pub engine: E,
    /// Runtime executor
    #[getset(get = "pub")]
    executor: VmExecutor<Val<E::SC>, VB::VmConfig>,
    #[getset(get = "pub", get_mut = "pub")]
    pk: DeviceMultiStarkProvingKey<E::PB>,
    chip_complex: VmChipComplex<E::SC, VB::RecordArena, E::PB, VB::SystemChipInventory>,
}

impl<E, VB> VirtualMachine<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub fn new(
        engine: E,
        builder: VB,
        config: VB::VmConfig,
        d_pk: DeviceMultiStarkProvingKey<E::PB>,
    ) -> Result<Self, VirtualMachineError> {
        let circuit = config.create_airs()?;
        let chip_complex = builder.create_chip_complex(&config, circuit)?;
        let executor = VmExecutor::<Val<E::SC>, _>::new(config)?;
        Ok(Self {
            engine,
            executor,
            pk: d_pk,
            chip_complex,
        })
    }

    pub fn new_with_keygen(
        engine: E,
        builder: VB,
        config: VB::VmConfig,
    ) -> Result<(Self, MultiStarkProvingKey<E::SC>), VirtualMachineError> {
        let circuit = config.create_airs()?;
        let pk = circuit.keygen(&engine);
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let vm = Self::new(engine, builder, config, d_pk)?;
        Ok((vm, pk))
    }

    pub fn config(&self) -> &VB::VmConfig {
        &self.executor.config
    }

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
    ) -> Result<PreflightExecutionOutput<Val<E::SC>, VB::RecordArena>, ExecutionError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
            InstructionExecutor<Val<E::SC>, VB::RecordArena>,
    {
        let handler = ProgramHandler::new(exe.program, &self.executor.inventory)?;
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        debug_assert!(executor_idx_to_air_idx
            .iter()
            .all(|&air_idx| air_idx < trace_heights.len()));
        let ctrl = TracegenExecutionControl::new(executor_idx_to_air_idx);
        let mut instance = VmSegmentExecutor::new(handler, ctrl);

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
        let exit_code = exec_state.exit_code?;
        let system_records = SystemRecords {
            from_state,
            to_state,
            exit_code,
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

    /// This function mutates `self` but should only depend on internal state in the sense that:
    /// - program must already be loaded as cached trace via [`load_program`](Self::load_program).
    /// - initial memory image was already sent to device via
    ///   [`transport_init_memory_to_device`](Self::transport_init_memory_to_device).
    /// - all other state should be given by `system_records` and `record_arenas`
    #[instrument(name = "tracegen", skip_all)]
    pub fn generate_proving_ctx(
        &mut self,
        system_records: SystemRecords<Val<E::SC>>,
        record_arenas: Vec<VB::RecordArena>,
    ) -> Result<ProvingContext<E::PB>, GenerationError> {
        #[cfg(feature = "bench-metrics")]
        let mut current_trace_heights =
            self.get_trace_heights_from_arenas(&system_records, &record_arenas);
        // main tracegen call:
        let ctx = self
            .chip_complex
            .generate_proving_ctx(system_records, record_arenas)?;

        // ==== Defensive checks that the trace heights satisfy the linear constraints: ====
        let idx_trace_heights = ctx
            .per_air
            .iter()
            .map(|(air_idx, ctx)| (*air_idx, ctx.main_trace_height()))
            .collect_vec();
        // 1. check max trace height isn't exceeded
        let max_trace_height = if TypeId::of::<Val<E::SC>>() == TypeId::of::<BabyBear>() {
            let min_log_blowup = log2_ceil_usize(self.config().as_ref().max_constraint_degree - 1);
            1 << (BabyBear::TWO_ADICITY - min_log_blowup)
        } else {
            tracing::warn!(
                "constructing VirtualMachine for unrecognized field; using max_trace_height=2^30"
            );
            1 << 30
        };
        if let Some(&(air_idx, height)) = idx_trace_heights
            .iter()
            .find(|(_, height)| *height > max_trace_height)
        {
            return Err(GenerationError::TraceHeightsLimitExceeded {
                air_idx,
                height,
                max_height: max_trace_height,
            });
        }
        // 2. check linear constraints on trace heights are satisfied
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
        #[cfg(feature = "bench-metrics")]
        self.finalize_metrics(&mut current_trace_heights);

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
        exe: impl Into<VmExe<Val<E::SC>>>,
        state: VmState<Val<E::SC>>,
        num_insns: Option<u64>,
        trace_heights: &[u32],
    ) -> Result<(Proof<E::SC>, Option<GuestMemory>), VirtualMachineError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
            InstructionExecutor<Val<E::SC>, VB::RecordArena>,
    {
        self.transport_init_memory_to_device(&state.memory);

        let PreflightExecutionOutput {
            system_records,
            record_arenas,
            to_state,
        } = self.execute_preflight(exe.into(), state, num_insns, trace_heights)?;
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

    /// Generates and then commits to program trace entirely on host.
    pub fn commit_exe(&self, exe: impl Into<VmExe<Val<E::SC>>>) -> VmCommittedExe<E::SC> {
        let exe = exe.into();
        VmCommittedExe::commit(exe, self.engine.config().pcs())
    }

    /// Convenience method to transport a host committed Exe to device. If the Exe has already been
    /// committed directly on device (either via a different caching mechanism or directly using
    /// device committer), then you can directly call [`load_program`](Self::load_program) and skip
    /// this function.
    pub fn transport_committed_exe_to_device(
        &self,
        committed_exe: &VmCommittedExe<E::SC>,
    ) -> CommittedTraceData<E::PB> {
        let commitment = committed_exe.commitment.clone();
        let trace = &committed_exe.trace;
        let prover_data = &committed_exe.prover_data;
        self.engine
            .device()
            .transport_committed_trace_to_device(commitment, trace, prover_data)
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

    pub fn num_airs(&self) -> usize {
        let num_airs = self.pk.per_air.len();
        debug_assert_eq!(num_airs, self.chip_complex.inventory.airs().num_airs());
        num_airs
    }

    pub fn air_names(&self) -> impl Iterator<Item = &'_ str> {
        self.pk.per_air.iter().map(|pk| pk.air_name.as_str())
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
#[derive(Getters)]
pub struct VmLocalProver<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub vm: VirtualMachine<E, VB>,
    #[getset(get = "pub")]
    exe_commitment: Com<E::SC>,
    // TODO: store immutable parts of program handler here
    #[getset(get = "pub")]
    exe: VmExe<Val<E::SC>>,
}

impl<E, VB> VmLocalProver<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub fn new(
        mut vm: VirtualMachine<E, VB>,
        exe: VmExe<Val<E::SC>>,
        cached_program_trace: CommittedTraceData<E::PB>,
    ) -> Self {
        let exe_commitment = cached_program_trace.commitment.clone();
        vm.load_program(cached_program_trace);
        Self {
            vm,
            exe,
            exe_commitment,
        }
    }
}

impl<E, VB> ContinuationVmProver<E::SC> for VmLocalProver<E, VB>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    VB: VmBuilder<E>,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: InsExecutorE1<Val<E::SC>>
        + InsExecutorE2<Val<E::SC>>
        + InstructionExecutor<Val<E::SC>, VB::RecordArena>,
{
    /// First performs metered execution (E2) to determine segments. Then sequentially proves each
    /// segment. The proof for each segment uses the specified [ProverBackend], but the proof for
    /// the next segment does not start before the current proof finishes.
    fn prove(
        &mut self,
        input: impl Into<Streams<Val<E::SC>>>,
    ) -> Result<ContinuationVmProof<E::SC>, VirtualMachineError> {
        self.prove_continuations(input, |_, _| {})
    }
}

impl<E, VB> VmLocalProver<E, VB>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    VB: VmBuilder<E>,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: InsExecutorE1<Val<E::SC>>
        + InsExecutorE2<Val<E::SC>>
        + InstructionExecutor<Val<E::SC>, VB::RecordArena>,
{
    /// For internal use to resize trace matrices before proving.
    ///
    /// The closure `modify_ctx(seg_idx, &mut ctx)` is called sequentially for each segment.
    pub fn prove_continuations(
        &mut self,
        input: impl Into<Streams<Val<E::SC>>>,
        mut modify_ctx: impl FnMut(usize, &mut ProvingContext<E::PB>),
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

            let mut ctx = vm.generate_proving_ctx(system_records, record_arenas)?;
            modify_ctx(seg_idx, &mut ctx);
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

impl<E, VB> SingleSegmentVmProver<E::SC> for VmLocalProver<E, VB>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    VB: VmBuilder<E>,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
        InstructionExecutor<Val<E::SC>, VB::RecordArena>,
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

pub(super) fn create_memory_image(
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

fn check_termination(exit_code: Result<Option<u32>, ExecutionError>) -> Result<(), ExecutionError> {
    let exit_code = exit_code?;
    match exit_code {
        Some(code) => check_exit_code(code),
        None => Err(ExecutionError::DidNotTerminate),
    }
}

impl<E, VC> VirtualMachine<E, VC>
where
    E: StarkEngine,
    VC: VmBuilder<E>,
    VC::SystemChipInventory: SystemWithFixedTraceHeights,
{
    /// Sets fixed trace heights for the system AIRs' trace matrices.
    pub fn override_system_trace_heights(&mut self, heights: &[u32]) {
        let num_sys_airs = self.config().as_ref().num_airs();
        assert!(heights.len() >= num_sys_airs);
        self.chip_complex
            .system
            .override_trace_heights(&heights[..num_sys_airs]);
    }
}

#[cfg(feature = "bench-metrics")]
mod vm_metrics {
    use std::iter::zip;

    use metrics::counter;

    use super::*;
    use crate::{arch::Arena, system::memory::adapter::AccessAdapterInventory};

    impl<E, VB> VirtualMachine<E, VB>
    where
        E: StarkEngine,
        VB: VmBuilder<E>,
    {
        /// Best effort calculation of the used trace heights per chip without padding to powers of
        /// two. This is best effort because some periphery chips may not have record arenas
        /// to instrument.
        pub(crate) fn get_trace_heights_from_arenas(
            &self,
            system_records: &SystemRecords<Val<E::SC>>,
            record_arenas: &[VB::RecordArena],
        ) -> Vec<usize> {
            let num_airs = self.num_airs();
            assert_eq!(num_airs, record_arenas.len());
            // First, get used heights from record arenas
            let mut heights: Vec<usize> = record_arenas
                .iter()
                .map(|arena| arena.current_trace_height())
                .collect();
            // Memory is special case, so extract the memory AIR's trace heights from the special
            // arena
            let sys_config = self.config().as_ref();
            let num_sys_airs = sys_config.num_airs();
            let access_adapter_offset = sys_config.access_adapter_air_id_offset();
            AccessAdapterInventory::<Val<E::SC>>::compute_heights_from_arena(
                &system_records.access_adapter_records,
                &mut heights[access_adapter_offset..num_sys_airs],
            );
            // If there are any constant trace heights, set them
            for (pk, height) in zip(&self.pk.per_air, &mut heights) {
                if let Some(constant_height) =
                    pk.preprocessed_data.as_ref().map(|pd| pd.trace.height())
                {
                    *height = constant_height;
                }
            }
            // Program chip used height
            heights[PROGRAM_AIR_ID] = system_records.filtered_exec_frequencies.len();

            heights
        }

        /// Update used trace heights after tracegen is done (primarily updating memory-related
        /// metrics) and then emit the final metrics.
        pub(crate) fn finalize_metrics(&self, heights: &mut [usize]) {
            self.chip_complex.system.update_trace_heights(heights);
            let mut main_cells_used = 0usize;
            let mut total_cells_used = 0usize;
            for (pk, height) in zip(&self.pk.per_air, heights.iter()) {
                let width = &pk.vk.params.width;
                main_cells_used += width.main_width() * *height;
                total_cells_used +=
                    width.total_width(<E::SC as StarkGenericConfig>::Challenge::D) * *height;
            }
            counter!("main_cells_used").absolute(main_cells_used as u64);
            counter!("total_cells_used").absolute(total_cells_used as u64);

            if self.config().as_ref().profiling {
                for (name, value) in zip(self.air_names(), heights) {
                    let labels = [("air_name", name.to_string())];
                    counter!("rows_used", &labels).absolute(*value as u64);
                }
            }
        }
    }
}
