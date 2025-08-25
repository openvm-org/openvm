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
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    program::Program,
};
use openvm_stark_backend::{
    config::{Com, StarkGenericConfig, Val},
    engine::StarkEngine,
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    p3_field::{FieldAlgebra, FieldExtensionAlgebra, PrimeField32, TwoAdicField},
    p3_util::{log2_ceil_usize, log2_strict_usize},
    proof::Proof,
    prover::{
        hal::{DeviceDataTransporter, MatrixDimensions, TraceCommitter},
        types::{CommittedTraceData, DeviceMultiStarkProvingKey, ProvingContext},
    },
    verifier::VerificationError,
};
use p3_baby_bear::BabyBear;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info_span, instrument};

use super::{
    execution_mode::{ExecutionCtx, MeteredCostCtx, MeteredCtx, PreflightCtx, Segment},
    hasher::poseidon2::vm_poseidon2_hasher,
    interpreter::InterpretedInstance,
    interpreter_preflight::PreflightInterpretedInstance,
    AirInventoryError, ChipInventoryError, ExecutionError, ExecutionState, Executor,
    ExecutorInventory, ExecutorInventoryError, MemoryConfig, MeteredExecutor, PreflightExecutor,
    StaticProgramError, SystemConfig, VmBuilder, VmChipComplex, VmCircuitConfig, VmExecState,
    VmExecutionConfig, VmState, CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID,
    PROGRAM_CACHED_TRACE_INDEX, PUBLIC_VALUES_AIR_ID,
};
use crate::{
    arch::DEFAULT_RNG_SEED,
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
        program::trace::{generate_cached_trace, VmCommittedExe},
        SystemChipComplex, SystemRecords, SystemWithFixedTraceHeights,
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

/// Typedef for [PreflightInterpretedInstance] that is generic in `VC: VmExecutionConfig<F>`
type PreflightInterpretedInstance2<F, VC> =
    PreflightInterpretedInstance<F, <VC as VmExecutionConfig<F>>::Executor>;

/// [VmExecutor] is the struct that can execute an _arbitrary_ program, provided in the form of a
/// [VmExe], for a fixed set of OpenVM instructions corresponding to a [VmExecutionConfig].
/// Internally once it is given a program, it will preprocess the program to rewrite it into a more
/// optimized format for runtime execution. This **instance** of the executor will be a separate
/// struct specialized to running a _fixed_ program on different program inputs.
#[derive(Clone)]
pub struct VmExecutor<F, VC>
where
    VC: VmExecutionConfig<F>,
{
    pub config: VC,
    inventory: Arc<ExecutorInventory<VC::Executor>>,
    phantom: PhantomData<F>,
}

#[repr(i32)]
pub enum ExitCode {
    Success = 0,
    Error = 1,
    Suspended = -1, // Continuations
}

pub struct PreflightExecutionOutput<F, RA> {
    pub system_records: SystemRecords<F>,
    pub record_arenas: Vec<RA>,
    pub to_state: VmState<F, GuestMemory>,
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
            inventory: Arc::new(inventory),
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
        MeteredCtx::new(
            constant_trace_heights.to_vec(),
            air_names.to_vec(),
            widths.to_vec(),
            interactions.to_vec(),
            self.config.as_ref(),
        )
    }

    pub fn build_metered_cost_ctx(&self, widths: &[usize]) -> MeteredCostCtx {
        MeteredCostCtx::new(widths.to_vec(), self.config.as_ref())
    }
}

impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F>,
    VC::Executor: Executor<F>,
{
    /// Creates an instance of the interpreter specialized for pure execution, without metering, of
    /// the given `exe`.
    ///
    /// For metered execution, use the [`metered_instance`](Self::metered_instance) constructor.
    pub fn instance(
        &self,
        exe: &VmExe<F>,
    ) -> Result<InterpretedInstance<F, ExecutionCtx>, StaticProgramError> {
        InterpretedInstance::new(&self.inventory, exe)
    }
}

impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F>,
    VC::Executor: MeteredExecutor<F>,
{
    /// Creates an instance of the interpreter specialized for metered execution of the given `exe`.
    pub fn metered_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<InterpretedInstance<F, MeteredCtx>, StaticProgramError> {
        InterpretedInstance::new_metered(&self.inventory, exe, executor_idx_to_air_idx)
    }

    /// Creates an instance of the interpreter specialized for cost metering execution of the given
    /// `exe`.
    pub fn metered_cost_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<InterpretedInstance<F, MeteredCostCtx>, StaticProgramError> {
        InterpretedInstance::new_metered(&self.inventory, exe, executor_idx_to_air_idx)
    }
}

#[derive(Error, Debug)]
pub enum VmVerificationError {
    #[error("no proof is provided")]
    ProofNotFound,

    #[error("program commit mismatch (index of mismatch proof: {index}")]
    ProgramCommitMismatch { index: usize },

    #[error("exe commit mismatch (expected: {expected:?}, actual: {actual:?})")]
    ExeCommitMismatch {
        expected: [u32; CHUNK],
        actual: [u32; CHUNK],
    },

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

    #[error("Invalid number of AIRs: expected at least 3, got {0}")]
    NotEnoughAirs(usize),

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
    #[error("static program error: {0}")]
    StaticProgram(#[from] StaticProgramError),
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
    #[cfg(feature = "stark-debug")]
    pub h_pk: Option<MultiStarkProvingKey<E::SC>>,
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
            #[cfg(feature = "stark-debug")]
            h_pk: None,
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

    /// Pure interpreter.
    pub fn interpreter(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<InterpretedInstance<Val<E::SC>, ExecutionCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        self.executor().instance(exe)
    }

    pub fn metered_interpreter(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<InterpretedInstance<Val<E::SC>, MeteredCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_instance(exe, &executor_idx_to_air_idx)
    }

    pub fn metered_cost_interpreter(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<InterpretedInstance<Val<E::SC>, MeteredCostCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_cost_instance(exe, &executor_idx_to_air_idx)
    }

    pub fn preflight_interpreter(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<PreflightInterpretedInstance2<Val<E::SC>, VB::VmConfig>, StaticProgramError> {
        PreflightInterpretedInstance::new(
            &exe.program,
            self.executor.inventory.clone(),
            self.executor_idx_to_air_idx(),
        )
    }

    /// Preflight execution for a single segment. Executes for exactly `num_insns` instructions
    /// using an interpreter. Preflight execution must be provided with `trace_heights`
    /// instrumentation data that was collected from a previous run of metered execution so that the
    /// preflight execution knows how much memory to allocate for record arenas.
    ///
    /// This function should rarely be called on its own. Users are advised to call
    /// [`prove`](Self::prove) directly.
    #[instrument(name = "execute_preflight", skip_all)]
    pub fn execute_preflight(
        &self,
        interpreter: &mut PreflightInterpretedInstance2<Val<E::SC>, VB::VmConfig>,
        state: VmState<Val<E::SC>, GuestMemory>,
        num_insns: Option<u64>,
        trace_heights: &[u32],
    ) -> Result<PreflightExecutionOutput<Val<E::SC>, VB::RecordArena>, ExecutionError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
            PreflightExecutor<Val<E::SC>, VB::RecordArena>,
    {
        debug_assert!(interpreter
            .executor_idx_to_air_idx
            .iter()
            .all(|&air_idx| air_idx < trace_heights.len()));

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
        let ctx = PreflightCtx::new_with_capacity(&capacities, instret_end);

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
            system_config.initial_block_size(),
            access_adapter_arena_size_bound,
        );
        let from_state = ExecutionState::new(state.pc, memory.timestamp());
        let vm_state = VmState {
            instret: state.instret,
            pc: state.pc,
            memory,
            streams: state.streams,
            rng: state.rng,
            custom_pvs: state.custom_pvs,
            #[cfg(feature = "metrics")]
            metrics: state.metrics,
        };
        let mut exec_state = VmExecState::new(vm_state, ctx);
        interpreter.reset_execution_frequencies();
        execute_spanned!("execute_preflight", interpreter, &mut exec_state)?;
        let filtered_exec_frequencies = interpreter.filtered_execution_frequencies();
        let touched_memory = exec_state
            .vm_state
            .memory
            .finalize::<Val<E::SC>>(system_config.continuation_enabled);
        #[cfg(feature = "perf-metrics")]
        crate::metrics::end_segment_metrics(&mut exec_state);

        let memory = exec_state.vm_state.memory;
        let to_state = ExecutionState::new(exec_state.vm_state.pc, memory.timestamp());
        let public_values = exec_state
            .vm_state
            .custom_pvs
            .iter()
            .map(|&x| x.unwrap_or(Val::<E::SC>::ZERO))
            .collect();
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
            instret: exec_state.vm_state.instret,
            pc: exec_state.vm_state.pc,
            memory: memory.data,
            streams: exec_state.vm_state.streams,
            rng: exec_state.vm_state.rng,
            custom_pvs: exec_state.vm_state.custom_pvs,
            #[cfg(feature = "metrics")]
            metrics: exec_state.vm_state.metrics,
        };
        Ok(PreflightExecutionOutput {
            system_records,
            record_arenas,
            to_state,
        })
    }

    /// Calls [`VmState::initial`] but sets more information for
    /// performance metrics when feature "perf-metrics" is enabled.
    #[instrument(name = "vm.create_initial_state", level = "debug", skip_all)]
    pub fn create_initial_state(
        &self,
        exe: &VmExe<Val<E::SC>>,
        inputs: impl Into<Streams<Val<E::SC>>>,
    ) -> VmState<Val<E::SC>, GuestMemory> {
        #[allow(unused_mut)]
        let mut state = VmState::initial(
            self.config().as_ref(),
            &exe.init_memory,
            exe.pc_start,
            inputs,
        );
        // Add backtrace information for either:
        // - debugging
        // - performance metrics
        #[cfg(all(feature = "metrics", any(feature = "perf-metrics", debug_assertions)))]
        {
            state.metrics.fn_bounds = exe.fn_bounds.clone();
            state.metrics.debug_infos = exe.program.debug_infos();
        }
        #[cfg(feature = "perf-metrics")]
        {
            state.metrics.set_pk_info(&self.pk);
            state.metrics.num_sys_airs = self.config().as_ref().num_airs();
            state.metrics.access_adapter_offset =
                self.config().as_ref().access_adapter_air_id_offset();
        }
        state
    }

    /// This function mutates `self` but should only depend on internal state in the sense that:
    /// - program must already be loaded as cached trace via [`load_program`](Self::load_program).
    /// - initial memory image was already sent to device via
    ///   [`transport_init_memory_to_device`](Self::transport_init_memory_to_device).
    /// - all other state should be given by `system_records` and `record_arenas`
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &mut self,
        system_records: SystemRecords<Val<E::SC>>,
        record_arenas: Vec<VB::RecordArena>,
    ) -> Result<ProvingContext<E::PB>, GenerationError> {
        #[cfg(feature = "metrics")]
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
        #[cfg(feature = "metrics")]
        self.finalize_metrics(&mut current_trace_heights);
        #[cfg(feature = "stark-debug")]
        self.debug_proving_ctx(&ctx);

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
        interpreter: &mut PreflightInterpretedInstance2<Val<E::SC>, VB::VmConfig>,
        state: VmState<Val<E::SC>, GuestMemory>,
        num_insns: Option<u64>,
        trace_heights: &[u32],
    ) -> Result<(Proof<E::SC>, Option<GuestMemory>), VirtualMachineError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
            PreflightExecutor<Val<E::SC>, VB::RecordArena>,
    {
        self.transport_init_memory_to_device(&state.memory);

        let PreflightExecutionOutput {
            system_records,
            record_arenas,
            to_state,
        } = self.execute_preflight(interpreter, state, num_insns, trace_heights)?;
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

    /// Transforms the program into a cached trace and commits it _on device_ using the proof system
    /// polynomial commitment scheme.
    ///
    /// Returns the cached program trace.
    /// Note that [`load_program`](Self::load_program) must be called separately to load the cached
    /// program trace into the VM itself.
    pub fn commit_program_on_device(
        &self,
        program: &Program<Val<E::SC>>,
    ) -> CommittedTraceData<E::PB> {
        let trace = generate_cached_trace(program);
        let d_trace = self
            .engine
            .device()
            .transport_matrix_to_device(&Arc::new(trace));
        let (commitment, data) = self.engine.device().commit(std::slice::from_ref(&d_trace));
        CommittedTraceData {
            commitment,
            trace: d_trace,
            data,
        }
    }

    /// Convenience method to transport a host committed Exe to device. This can be used if you have
    /// a pre-committed program and want to transport to device instead of re-committing. One should
    /// benchmark the latency of this function versus
    /// [`commit_program_on_device`](Self::commit_program_on_device), which directly re-commits on
    /// device, to determine which method is more suitable.
    pub fn transport_committed_exe_to_device(
        &self,
        committed_exe: &VmCommittedExe<E::SC>,
    ) -> CommittedTraceData<E::PB> {
        let commitment = committed_exe.get_program_commit();
        let trace = &committed_exe.trace;
        let prover_data = &committed_exe.prover_data;
        self.engine
            .device()
            .transport_committed_trace_to_device(commitment, trace, prover_data)
    }

    /// Loads cached program trace into the VM.
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

    /// Convenience method to construct a [MeteredCostCtx] using data from the stored proving key.
    pub fn build_metered_cost_ctx(&self) -> MeteredCostCtx {
        let widths: Vec<_> = self
            .pk
            .per_air
            .iter()
            .map(|pk| {
                pk.vk
                    .params
                    .width
                    .total_width(<<E::SC as StarkGenericConfig>::Challenge>::D)
            })
            .collect();

        self.executor().build_metered_cost_ctx(&widths)
    }

    pub fn num_airs(&self) -> usize {
        let num_airs = self.pk.per_air.len();
        debug_assert_eq!(num_airs, self.chip_complex.inventory.airs().num_airs());
        num_airs
    }

    pub fn air_names(&self) -> impl Iterator<Item = &'_ str> {
        self.pk.per_air.iter().map(|pk| pk.air_name.as_str())
    }

    /// See [`debug_proving_ctx`].
    #[cfg(feature = "stark-debug")]
    pub fn debug_proving_ctx(&mut self, ctx: &ProvingContext<E::PB>) {
        if self.h_pk.is_none() {
            let air_inv = self.config().create_airs().unwrap();
            self.h_pk = Some(air_inv.keygen(&self.engine));
        }
        let pk = self.h_pk.as_ref().unwrap();
        debug_proving_ctx(self, pk, ctx);
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
///
/// This struct contains the [VmState] itself to avoid re-allocating guest memory. The memory is
/// reset with zeros before execution.
#[derive(Getters, MutGetters)]
pub struct VmInstance<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub vm: VirtualMachine<E, VB>,
    pub interpreter: PreflightInterpretedInstance2<Val<E::SC>, VB::VmConfig>,
    #[getset(get = "pub")]
    program_commitment: Com<E::SC>,
    #[getset(get = "pub")]
    exe: Arc<VmExe<Val<E::SC>>>,
    #[getset(get = "pub", get_mut = "pub")]
    state: Option<VmState<Val<E::SC>, GuestMemory>>,
}

impl<E, VB> VmInstance<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub fn new(
        mut vm: VirtualMachine<E, VB>,
        exe: Arc<VmExe<Val<E::SC>>>,
        cached_program_trace: CommittedTraceData<E::PB>,
    ) -> Result<Self, StaticProgramError> {
        let program_commitment = cached_program_trace.commitment.clone();
        vm.load_program(cached_program_trace);
        let interpreter = vm.preflight_interpreter(&exe)?;
        let state = vm.create_initial_state(&exe, vec![]);
        Ok(Self {
            vm,
            interpreter,
            program_commitment,
            exe,
            state: Some(state),
        })
    }

    #[instrument(name = "vm.reset_state", level = "debug", skip_all)]
    pub fn reset_state(&mut self, inputs: impl Into<Streams<Val<E::SC>>>) {
        self.state
            .as_mut()
            .unwrap()
            .reset(&self.exe.init_memory, self.exe.pc_start, inputs);
    }
}

impl<E, VB> ContinuationVmProver<E::SC> for VmInstance<E, VB>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    VB: VmBuilder<E>,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
        + MeteredExecutor<Val<E::SC>>
        + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
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

impl<E, VB> VmInstance<E, VB>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    VB: VmBuilder<E>,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
        + MeteredExecutor<Val<E::SC>>
        + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
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
        self.reset_state(input.clone());
        let vm = &mut self.vm;
        let metered_ctx = vm.build_metered_ctx();
        let metered_interpreter = vm.metered_interpreter(&self.exe)?;
        let (segments, _) = metered_interpreter.execute_metered(input, metered_ctx)?;
        let mut proofs = Vec::with_capacity(segments.len());
        let mut state = self.state.take();
        for (seg_idx, segment) in segments.into_iter().enumerate() {
            let _segment_span = info_span!("prove_segment", segment = seg_idx).entered();
            // We need a separate span so the metric label includes "segment" from _segment_span
            let _prove_span = info_span!("total_proof").entered();
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
            } = vm.execute_preflight(
                &mut self.interpreter,
                from_state,
                Some(num_insns),
                &trace_heights,
            )?;
            state = Some(to_state);

            let mut ctx = vm.generate_proving_ctx(system_records, record_arenas)?;
            modify_ctx(seg_idx, &mut ctx);
            let proof = vm.engine.prove(vm.pk(), ctx);
            proofs.push(proof);
        }
        let to_state = state.unwrap();
        let final_memory = &to_state.memory.memory;
        let user_public_values = UserPublicValuesProof::compute(
            vm.config().as_ref().memory_config.memory_dimensions(),
            vm.config().as_ref().num_public_values,
            &vm_poseidon2_hasher(),
            final_memory,
        );
        self.state = Some(to_state);
        Ok(ContinuationVmProof {
            per_segment: proofs,
            user_public_values,
        })
    }
}

impl<E, VB> SingleSegmentVmProver<E::SC> for VmInstance<E, VB>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    VB: VmBuilder<E>,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
        PreflightExecutor<Val<E::SC>, VB::RecordArena>,
{
    #[instrument(name = "total_proof", skip_all)]
    fn prove(
        &mut self,
        input: impl Into<Streams<Val<E::SC>>>,
        trace_heights: &[u32],
    ) -> Result<Proof<E::SC>, VirtualMachineError> {
        self.reset_state(input);
        let vm = &mut self.vm;
        let exe = &self.exe;
        assert!(!vm.config().as_ref().continuation_enabled);
        let mut trace_heights = trace_heights.to_vec();
        trace_heights[PUBLIC_VALUES_AIR_ID] = vm.config().as_ref().num_public_values as u32;
        let state = self.state.take().expect("State should always be present");
        let num_custom_pvs = state.custom_pvs.len();
        let (proof, final_memory) = vm.prove(&mut self.interpreter, state, None, &trace_heights)?;
        let final_memory = final_memory.ok_or(ExecutionError::DidNotTerminate)?;
        // Put back state to avoid re-allocation
        self.state = Some(VmState::new(
            0,
            exe.pc_start,
            final_memory,
            vec![],
            DEFAULT_RNG_SEED,
            num_custom_pvs,
        ));
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
    init_memory: &SparseMemoryImage,
) -> GuestMemory {
    let mut inner = AddressMap::new(memory_config.addr_spaces.clone());
    inner.set_from_sparse(init_memory);
    GuestMemory::new(inner)
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

/// Runs the STARK backend debugger to check the constraints against the trace matrices
/// logically, instead of cryptographically. This will panic if any constraint is violated, and
/// using `RUST_BACKTRACE=1` can be used to read the stack backtrace of where the constraint
/// failed in the code (this requires the code to be compiled with debug=true). Using lower
/// optimization levels like -O0 will prevent the compiler from inlining and give better
/// debugging information.
// @dev The debugger needs the host proving key.
//      This function is used both by VirtualMachine::debug_proving_ctx and by
// stark_utils::air_test_impl
#[cfg(any(debug_assertions, feature = "test-utils", feature = "stark-debug"))]
#[tracing::instrument(level = "debug", skip_all)]
pub fn debug_proving_ctx<E, VB>(
    vm: &VirtualMachine<E, VB>,
    pk: &MultiStarkProvingKey<E::SC>,
    ctx: &ProvingContext<E::PB>,
) where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    use itertools::multiunzip;
    use openvm_stark_backend::prover::types::AirProofRawInput;

    let device = vm.engine.device();
    let air_inv = vm.config().create_airs().unwrap();
    let global_airs = air_inv.into_airs().collect_vec();
    let (airs, pks, proof_inputs): (Vec<_>, Vec<_>, Vec<_>) =
        multiunzip(ctx.per_air.iter().map(|(air_id, air_ctx)| {
            // Transfer from device **back** to host so the debugger can read the data.
            let cached_mains = air_ctx
                .cached_mains
                .iter()
                .map(|pre| device.transport_matrix_from_device_to_host(&pre.trace))
                .collect_vec();
            let common_main = air_ctx
                .common_main
                .as_ref()
                .map(|m| device.transport_matrix_from_device_to_host(m));
            let public_values = air_ctx.public_values.clone();
            let raw = AirProofRawInput {
                cached_mains,
                common_main,
                public_values,
            };
            (
                global_airs[*air_id].clone(),
                pk.per_air[*air_id].clone(),
                raw,
            )
        }));
    vm.engine.debug(&airs, &pks, &proof_inputs);
}

#[cfg(feature = "metrics")]
mod vm_metrics {
    use std::iter::zip;

    use metrics::counter;

    use super::*;
    use crate::arch::Arena;

    impl<E, VB> VirtualMachine<E, VB>
    where
        E: StarkEngine,
        VB: VmBuilder<E>,
    {
        /// Assumed that `record_arenas` has length equal to number of AIRs.
        ///
        /// Best effort calculation of the used trace heights per chip without padding to powers of
        /// two. This is best effort because some periphery chips may not have record arenas to
        /// instrument. This function includes the constant trace heights, and the used height of
        /// the program trace. It does not include the memory access adapter trace heights,
        /// which is included in `SystemChipComplex::finalize_trace_heights`.
        pub(crate) fn get_trace_heights_from_arenas(
            &self,
            system_records: &SystemRecords<Val<E::SC>>,
            record_arenas: &[VB::RecordArena],
        ) -> Vec<usize> {
            let num_airs = self.num_airs();
            assert_eq!(num_airs, record_arenas.len());
            let mut heights: Vec<usize> = record_arenas
                .iter()
                .map(|arena| arena.current_trace_height())
                .collect();
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
            self.chip_complex.system.finalize_trace_heights(heights);
            let mut main_cells_used = 0usize;
            let mut total_cells_used = 0usize;
            for (pk, height) in zip(&self.pk.per_air, heights.iter()) {
                let width = &pk.vk.params.width;
                main_cells_used += width.main_width() * *height;
                total_cells_used +=
                    width.total_width(<E::SC as StarkGenericConfig>::Challenge::D) * *height;
            }
            tracing::debug!(?heights);
            tracing::info!(main_cells_used, total_cells_used);
            counter!("main_cells_used").absolute(main_cells_used as u64);
            counter!("total_cells_used").absolute(total_cells_used as u64);

            #[cfg(feature = "perf-metrics")]
            {
                for (name, value) in zip(self.air_names(), heights) {
                    let labels = [("air_name", name.to_string())];
                    counter!("rows_used", &labels).absolute(*value as u64);
                }
            }
        }
    }
}
