//! [VmExecutor] is the struct that can execute an _arbitrary_ program, provided in the form of a
//! [VmExe](openvm_instructions::exe::VmExe), for a fixed set of OpenVM instructions
//! corresponding to a [VmExecutionConfig].
//! Internally once it is given a program, it will preprocess the program to rewrite it into a more
//! optimized format for runtime execution. This **instance** of the executor will be a separate
//! struct specialized to running a _fixed_ program on different program inputs.
//!
//! [VirtualMachine] will similarly be the struct that has done all the setup so it can
//! execute+prove an arbitrary program for a fixed config - it will internally still hold VmExecutor
use std::{any::TypeId, borrow::Borrow, collections::VecDeque, sync::Arc};

use getset::{Getters, MutGetters, Setters, WithSetters};
use itertools::{zip_eq, Itertools};
use openvm_circuit::system::program::trace::compute_exe_commit;
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    program::Program,
    VM_DIGEST_WIDTH,
};
#[cfg(feature = "rvr")]
use openvm_instructions::{LocalOpcode, SystemOpcode};
#[cfg(any(debug_assertions, feature = "test-utils", feature = "stark-debug"))]
use openvm_stark_backend::AirRef;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    memory_metering::ProvingMemoryConfig,
    p3_field::{InjectiveMonomial, PrimeCharacteristicRing, PrimeField32, TwoAdicField},
    p3_util::log2_ceil_usize,
    proof::Proof,
    prover::{
        ColMajorMatrix, CommittedTraceData, DeviceDataTransporter, DeviceMultiStarkProvingKey,
        MatrixDimensions, ProverBackend, ProverDevice, ProvingContext, TraceCommitter,
    },
    verifier::VerifierError,
    Com, StarkEngine, StarkProtocolConfig, Val,
};
use p3_baby_bear::BabyBear;
#[cfg(feature = "rvr")]
use rvr_openvm_lift::{RvrExtensions, RvrRuntimeExtension};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info_span, instrument};

#[cfg(feature = "rvr")]
use super::rvr::{
    bridge::map_rvr_compile_error,
    build_pc_to_chip, classify_preflight_opcodes_with_extensions, compile, compile_metered,
    compile_metered_cost, compile_metered_segment_boundary, compile_with_instret_tracking,
    load_compiled_from_path,
    preflight::execute_rvr_preflight,
    preflight::{ChipRecordBuf, RvrArenaNativeTarget},
    rvr_preflight_engine_env_override, ArenaNativeGeometry, ChipMapping, GuestDebugMap,
    LogNativeOpcodeAdmitter, PreflightMemoryAccessAux, PreflightRawLogs, RvrCompiled,
    RvrDeltaRecords, RvrExecutionKind, RvrInitialImage, RvrInlineChipRecords,
    RvrMeteredCostInstance, RvrMeteredInstance, RvrMeteredSegmentInstance, RvrPreflightBufferPool,
    RvrPreflightEngine, RvrPreflightInstance, RvrPreflightOpcodeClass, RvrPreflightOutput,
    RvrPreflightRoute, RvrPureInstance, RvrPureWithInstretTrackingInstance,
};
use super::{
    execution_mode::{
        ExecutionCtx, MeteredCostCtx, MeteredCtx, MeteredCtxInputs, PreflightCtx, Segment,
        SegmentationLimits,
    },
    hasher::poseidon2::vm_poseidon2_hasher,
    hint_stream::HintStream,
    interpreter::InterpretedInstance,
    interpreter_preflight::PreflightInterpretedInstance,
    AirInventoryError, ChipInventoryError, ExecutionError, ExecutionState, Executor,
    ExecutorInventory, ExecutorInventoryError, MemoryConfig, MeteredExecutor, PreflightExecutor,
    StaticProgramError, SystemConfig, VmBuilder, VmChipComplex, VmCircuitConfig, VmExecState,
    VmExecutionConfig, VmState, BOUNDARY_AIR_ID, CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID,
    PROGRAM_CACHED_TRACE_INDEX,
};
#[cfg(feature = "metrics")]
use crate::metrics::emit_opcode_counts;
#[cfg(feature = "perf-metrics")]
use crate::metrics::end_segment_metrics;
use crate::{
    arch::deferral::DeferralState,
    execute_spanned,
    system::{
        connector::{VmConnectorPvs, DEFAULT_SUSPEND_EXIT_CODE},
        memory::{
            merkle::{
                public_values::{UserPublicValuesProof, UserPublicValuesProofError},
                MemoryMerklePvs,
            },
            online::{GuestMemory, TracingMemory},
            AddressMap,
        },
        program::trace::generate_cached_trace,
        SystemChipComplex, SystemRecords, SystemWithFixedTraceHeights,
    },
};

/// Canonical field bound for VM execution/circuit code.
pub const BABYBEAR_S_BOX_DEGREE: u64 = 7;

pub trait VmField: PrimeField32 + InjectiveMonomial<BABYBEAR_S_BOX_DEGREE> {}
impl<T> VmField for T where T: PrimeField32 + InjectiveMonomial<BABYBEAR_S_BOX_DEGREE> {}

#[cfg(feature = "rvr")]
type VmRvrPreflightRoute<'a, F, VC> =
    RvrPreflightRoute<'a, F, <VC as VmExecutionConfig<F>>::Executor>;

#[cfg(feature = "rvr")]
trait CachedRvrMeteredExecutor: Send + Sync {
    fn execute_metered(
        &self,
        inputs: Streams,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<GuestMemory>), ExecutionError>;
}

#[cfg(feature = "rvr")]
impl CachedRvrMeteredExecutor for RvrMeteredInstance<'static> {
    fn execute_metered(
        &self,
        inputs: Streams,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<GuestMemory>), ExecutionError> {
        RvrMeteredInstance::execute_metered(self, inputs, ctx)
    }
}

#[cfg(feature = "rvr")]
trait CachedRvrPreflightExecutor<F>: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        exe: &VmExe<F>,
        state: VmState<GuestMemory>,
        num_insns: Option<u64>,
        record_capacity_rows: Option<&[u32]>,
        arena_targets: Option<&std::collections::BTreeMap<usize, ChipRecordBuf>>,
    ) -> Result<RvrPreflightOutput<F>, ExecutionError>;

    /// Return a consumed segment output's large buffers to the executor's
    /// pool (the record arenas hold expanded copies by this point) so the
    /// next segment's `execute` reuses them.
    fn recycle_segment_buffers(
        &self,
        raw_logs: PreflightRawLogs,
        inline_records: Vec<RvrInlineChipRecords>,
        delta_records: Option<RvrDeltaRecords>,
    );
    /// Return the final-layout access replay after log-native assembly has consumed it.
    fn recycle_access_aux(&self, access_aux: Vec<PreflightMemoryAccessAux<F>>);

    /// R4: airs whose records the compiled library writes arena-native.
    fn arena_native_airs(&self) -> &[(usize, ArenaNativeGeometry)];

    /// R3/G2: `(air, packed wire record size)` for every air the compiled
    /// library emits inline compact records for.
    fn inline_wire_airs(&self) -> &[(usize, usize)];

    /// VmExe-keyed static PC-to-AIR route, built once with the compiled
    /// preflight executor rather than once per continuation segment.
    fn pc_to_air_idx(&self) -> &[Option<usize>];

    fn wire_airs(&self) -> &[usize];

    fn buffer_pool(&self) -> &RvrPreflightBufferPool;

    /// Grow and fault in all direct-final compact backings before the timed
    /// per-segment preflight call.
    fn prepare_wire_backings(&self, trace_heights: &[u32]);

    /// CUDA: reserve resident maximum-shape arena-native backings before the segment loop.
    #[cfg(feature = "cuda")]
    fn prepare_arena_native_backings(
        &self,
        trace_heights: &[u32],
        g2_capacity_bytes: Option<usize>,
    );

    /// CUDA/G2: return the backing capacity for one actual metered segment.
    /// Taking the maximum of these scalar joint shapes avoids constructing an
    /// impossible shape from independent per-AIR maxima.
    #[cfg(feature = "cuda")]
    fn g2_capacity_bytes(&self, trace_heights: &[u32], num_insns: u64) -> Option<usize>;

    #[cfg(feature = "cuda")]
    fn g2_air_indices(&self) -> Vec<usize>;
}

/// The program-dependent, owned pieces of an rvr preflight instance.
#[cfg(feature = "rvr")]
struct CachedRvrCompiledPreflight {
    compiled: RvrCompiled,
    runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
    chip_counts_len: usize,
    pool: RvrPreflightBufferPool,
    pc_to_air_idx: Vec<Option<usize>>,
    wire_airs: Vec<usize>,
    /// False only when compiler metadata plus the builder's whole-AIR device
    /// coverage prove that no program slot can reach the host log assembler.
    build_access_aux: bool,
    /// True only for a fully-direct delta route backed by the builder's
    /// device decoder. CPU and all-custom arena-only routes keep the full
    /// host memory schema even when they can safely omit access aux.
    compact_delta_memory: bool,
    /// True only when the backend has bound the CUDA replay consumer that can
    /// replace host touched-memory finalization.
    device_touched_memory: bool,
}

#[cfg(feature = "rvr")]
impl<F: PrimeField32> CachedRvrPreflightExecutor<F> for CachedRvrCompiledPreflight {
    fn execute(
        &self,
        exe: &VmExe<F>,
        state: VmState<GuestMemory>,
        num_insns: Option<u64>,
        record_capacity_rows: Option<&[u32]>,
        arena_targets: Option<&std::collections::BTreeMap<usize, ChipRecordBuf>>,
    ) -> Result<RvrPreflightOutput<F>, ExecutionError> {
        execute_rvr_preflight(
            exe,
            &self.runtime_hooks,
            &self.compiled,
            self.chip_counts_len,
            &self.pool,
            state,
            num_insns,
            record_capacity_rows,
            arena_targets,
            self.build_access_aux,
            self.compact_delta_memory,
            self.device_touched_memory,
        )
    }

    fn recycle_segment_buffers(
        &self,
        raw_logs: PreflightRawLogs,
        inline_records: Vec<RvrInlineChipRecords>,
        delta_records: Option<RvrDeltaRecords>,
    ) {
        self.pool
            .recycle_segment_buffers(raw_logs, inline_records, delta_records);
    }
    fn recycle_access_aux(&self, access_aux: Vec<PreflightMemoryAccessAux<F>>) {
        self.pool.recycle_access_aux(access_aux);
    }

    fn arena_native_airs(&self) -> &[(usize, ArenaNativeGeometry)] {
        &self.compiled.inline_records().arena_native_airs
    }

    fn inline_wire_airs(&self) -> &[(usize, usize)] {
        &self.compiled.inline_records().airs
    }

    fn pc_to_air_idx(&self) -> &[Option<usize>] {
        &self.pc_to_air_idx
    }

    fn wire_airs(&self) -> &[usize] {
        &self.wire_airs
    }

    fn buffer_pool(&self) -> &RvrPreflightBufferPool {
        &self.pool
    }

    fn prepare_wire_backings(&self, trace_heights: &[u32]) {
        for &air in &self.wire_airs {
            let wire_size = self
                .compiled
                .inline_records()
                .airs
                .iter()
                .find(|&&(wire_air, _)| wire_air == air)
                .map(|&(_, size)| size)
                .expect("cached wire air missing compiled record size");
            self.pool
                .prepare_wire_backing(air, trace_heights[air] as usize * wire_size);
        }
    }

    #[cfg(feature = "cuda")]
    fn prepare_arena_native_backings(
        &self,
        trace_heights: &[u32],
        g2_capacity_bytes: Option<usize>,
    ) {
        let stats_enabled = crate::arch::cuda::pinned::stats_enabled();
        let before = stats_enabled.then(crate::arch::cuda::pinned::PoolStatsSnapshot::capture);
        let prewarm_started = stats_enabled.then(std::time::Instant::now);
        for &(air, ref geometry) in &self.compiled.inline_records().arena_native_airs {
            let capacity_bytes = trace_heights[air] as usize * geometry.stride_dense();
            self.pool.prepare_arena_native_dense_backings(
                crate::arch::rvr::preflight_pool::ArenaNativeBackingKey::new(
                    air,
                    geometry.stride_dense(),
                    capacity_bytes,
                ),
            );
        }
        if let Some(capacity_bytes) = g2_capacity_bytes {
            if std::env::var("OPENVM_RVR_G2_GPU_PROFILE").as_deref() == Ok("1") {
                eprintln!("OPENVM_RVR_G2_JOINT_PREWARM capacity_bytes={capacity_bytes}");
            }
            let g2 = self
                .compiled
                .inline_records()
                .g2
                .as_deref()
                .expect("G2 capacity requires compiled G2 metadata");
            let route = if g2.checked_emission() {
                super::rvr::preflight_pool::G2BackingRoute::Checked
            } else {
                super::rvr::preflight_pool::G2BackingRoute::Production
            };
            self.pool.prepare_g2_backings(capacity_bytes, route);
        }
        if let Some(before) = before {
            let after = crate::arch::cuda::pinned::PoolStatsSnapshot::capture();
            eprintln!(
                "OPENVM_RVR_CUDA_POOL_PREWARM elapsed_ms={} hits={} misses={} populate_calls={} \
                 populate_bytes={} ready_buffers={} ready_bytes={} pending={} \
                 quarantined_total={} sync_failures_total={}",
                prewarm_started
                    .expect("pool prewarm timer missing with stats enabled")
                    .elapsed()
                    .as_millis(),
                after.hits.saturating_sub(before.hits),
                after.misses.saturating_sub(before.misses),
                after.populate_calls.saturating_sub(before.populate_calls),
                after.populate_bytes.saturating_sub(before.populate_bytes),
                after.ready_buffers,
                after.ready_bytes,
                after.pending,
                after.quarantined,
                after.sync_failures,
            );
        }
    }

    #[cfg(feature = "cuda")]
    fn g2_capacity_bytes(&self, trace_heights: &[u32], num_insns: u64) -> Option<usize> {
        let g2 = self.compiled.inline_records().g2.as_deref()?;
        let capacities =
            super::rvr::g2::RvrG2CapacitiesV1::for_metered_segment(g2, trace_heights, num_insns)
                .expect("G2 metered capacity model overflow");
        Some(
            super::rvr::g2::RvrG2PreparedV1::capacity_bytes(&capacities)
                .expect("G2 segment capacity overflow"),
        )
    }

    #[cfg(feature = "cuda")]
    fn g2_air_indices(&self) -> Vec<usize> {
        self.compiled
            .inline_records()
            .g2
            .as_deref()
            .map(|g2| {
                g2.air_bindings
                    .iter()
                    .map(|binding| binding.air_idx)
                    .chain(g2.opaque_bindings.iter().map(|binding| binding.air_idx))
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Cached program-derived preflight route for a fixed [`VmInstance`].
#[cfg(feature = "rvr")]
enum CachedRvrPreflight<F> {
    Rvr(Box<dyn CachedRvrPreflightExecutor<F>>),
    Interpreter,
}

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

#[derive(Clone, Default)]
pub struct Streams {
    pub input_stream: VecDeque<Vec<u8>>,
    pub hint_stream: HintStream,
    /// Cached deferred operation inputs and outputs. Each idx corresponds to a
    /// unique function that is constrained outside the VM in its own deferral circuit.
    pub deferrals: Vec<DeferralState>,
}

impl Streams {
    pub fn new(input_stream: impl Into<VecDeque<Vec<u8>>>) -> Self {
        Self {
            input_stream: input_stream.into(),
            hint_stream: HintStream::default(),
            deferrals: Vec::default(),
        }
    }
}

impl From<VecDeque<Vec<u8>>> for Streams {
    fn from(value: VecDeque<Vec<u8>>) -> Self {
        Streams::new(value)
    }
}

impl From<Vec<Vec<u8>>> for Streams {
    fn from(value: Vec<Vec<u8>>) -> Self {
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
    pub to_state: VmState<GuestMemory>,
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
        })
    }
}

impl<F, VC> VmExecutor<F, VC>
where
    VC: VmExecutionConfig<F> + AsRef<SystemConfig>,
{
    pub fn build_metered_ctx(
        &self,
        inputs: MeteredCtxInputs<'_>,
        memory_config: ProvingMemoryConfig,
    ) -> MeteredCtx {
        MeteredCtx::new(inputs, self.config.as_ref(), memory_config)
    }

    pub fn build_metered_cost_ctx(&self, widths: &[usize]) -> MeteredCostCtx {
        MeteredCostCtx::new(widths.to_vec())
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
    #[cfg(not(feature = "rvr"))]
    pub fn instance(
        &self,
        exe: &VmExe<F>,
    ) -> Result<InterpretedInstance<'_, ExecutionCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_pure", backend = "interpreter").entered();
        InterpretedInstance::new(&self.inventory, exe)
    }

    #[cfg(feature = "rvr")]
    pub fn interpreter_instance(
        &self,
        exe: &VmExe<F>,
    ) -> Result<InterpretedInstance<'_, ExecutionCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_pure", backend = "interpreter").entered();
        InterpretedInstance::new(&self.inventory, exe)
    }

    #[cfg(feature = "rvr")]
    pub fn instance(&self, exe: &VmExe<F>) -> Result<RvrPureInstance<'_>, StaticProgramError> {
        Self::rvr_instance(self, exe, None)
    }
}

#[cfg(feature = "rvr")]
impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F>,
{
    fn build_rvr_extensions(&self, executor_idx_to_air_idx: Option<&[usize]>) -> RvrExtensions {
        self.config.create_rvr_extensions(executor_idx_to_air_idx)
    }
}

#[cfg(feature = "rvr")]
impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F>,
    VC::Executor: Executor<F>,
{
    pub fn rvr_instance(
        &self,
        exe: &VmExe<F>,
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<RvrPureInstance<'_>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span = tracing::info_span!("compile_pure", backend = "rvr").entered();
        let extensions = self.build_rvr_extensions(None);
        let compiled =
            compile(exe, extensions.lifters(), guest_debug_map).map_err(map_rvr_compile_error)?;
        Ok(RvrPureInstance::new(
            self.inventory.config(),
            RvrInitialImage::from(exe),
            compiled,
            extensions.into_runtime_hooks(),
        ))
    }

    /// Compile a pure RVR instance with instret tracking and block-boundary suspension.
    pub fn rvr_instret_tracking_instance(
        &self,
        exe: &VmExe<F>,
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<RvrPureWithInstretTrackingInstance<'_>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span = tracing::info_span!("compile_pure", backend = "rvr").entered();
        let extensions = self.build_rvr_extensions(None);
        let compiled = compile_with_instret_tracking(exe, extensions.lifters(), guest_debug_map)
            .map_err(map_rvr_compile_error)?;
        Ok(RvrPureWithInstretTrackingInstance::new(
            self.inventory.config(),
            RvrInitialImage::from(exe),
            compiled,
            extensions.into_runtime_hooks(),
        ))
    }

    /// Load a previously saved unlimited-pure artifact.
    pub fn load_instance(
        &self,
        lib_path: &std::path::Path,
        exe: &VmExe<F>,
    ) -> Result<RvrPureInstance<'_>, StaticProgramError> {
        let extensions = self.build_rvr_extensions(None);
        let compiled = load_compiled_from_path(lib_path).map_err(map_rvr_compile_error)?;
        compiled
            .require_execution_kind(&[RvrExecutionKind::Pure])
            .map_err(map_rvr_compile_error)?;
        Ok(RvrPureInstance::new(
            self.inventory.config(),
            RvrInitialImage::from(exe),
            compiled,
            extensions.into_runtime_hooks(),
        ))
    }

    /// Load a previously saved pure artifact with instret tracking.
    pub fn load_instret_tracking_instance(
        &self,
        lib_path: &std::path::Path,
        exe: &VmExe<F>,
    ) -> Result<RvrPureWithInstretTrackingInstance<'_>, StaticProgramError> {
        let extensions = self.build_rvr_extensions(None);
        let compiled = load_compiled_from_path(lib_path).map_err(map_rvr_compile_error)?;
        compiled
            .require_execution_kind(&[RvrExecutionKind::PureWithInstretTracking])
            .map_err(map_rvr_compile_error)?;
        Ok(RvrPureWithInstretTrackingInstance::new(
            self.inventory.config(),
            RvrInitialImage::from(exe),
            compiled,
            extensions.into_runtime_hooks(),
        ))
    }
}

impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F>,
    VC::Executor: MeteredExecutor<F>,
{
    /// Creates an instance of the interpreter specialized for metered execution of the given `exe`.
    #[cfg(not(feature = "rvr"))]
    pub fn metered_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<InterpretedInstance<'_, MeteredCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered", backend = "interpreter").entered();
        InterpretedInstance::new_metered(&self.inventory, exe, executor_idx_to_air_idx)
    }

    #[cfg(feature = "rvr")]
    pub fn metered_interpreter_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<InterpretedInstance<'_, MeteredCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered", backend = "interpreter").entered();
        InterpretedInstance::new_metered(&self.inventory, exe, executor_idx_to_air_idx)
    }

    /// Creates an instance of the interpreter specialized for cost metering execution of the given
    /// `exe`.
    #[cfg(not(feature = "rvr"))]
    pub fn metered_cost_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<InterpretedInstance<'_, MeteredCostCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered_cost", backend = "interpreter").entered();
        InterpretedInstance::new_metered(&self.inventory, exe, executor_idx_to_air_idx)
    }

    #[cfg(feature = "rvr")]
    pub fn metered_cost_interpreter_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<InterpretedInstance<'_, MeteredCostCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered_cost", backend = "interpreter").entered();
        InterpretedInstance::new_metered(&self.inventory, exe, executor_idx_to_air_idx)
    }
}

#[cfg(feature = "rvr")]
impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F>,
    VC::Executor: MeteredExecutor<F>,
{
    pub fn metered_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
        num_airs: usize,
    ) -> Result<RvrMeteredInstance<'_>, StaticProgramError> {
        self.metered_rvr_instance(exe, executor_idx_to_air_idx, num_airs, None)
    }

    pub fn metered_rvr_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
        num_airs: usize,
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<RvrMeteredInstance<'_>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span = tracing::info_span!("compile_metered", backend = "rvr").entered();
        let extensions = self.build_rvr_extensions(Some(executor_idx_to_air_idx));
        let chips = ChipMapping {
            num_airs,
            pc_to_chip: build_pc_to_chip(exe, &self.inventory, executor_idx_to_air_idx)
                .map_err(map_rvr_compile_error)?,
            chip_widths: None,
        };
        let compiled = compile_metered(exe, extensions.lifters(), &chips, guest_debug_map)
            .map_err(map_rvr_compile_error)?;
        let runtime_hooks = extensions.into_runtime_hooks();

        Ok(RvrMeteredInstance::new(
            self.inventory.config(),
            RvrInitialImage::from(exe),
            runtime_hooks,
            compiled,
        ))
    }

    pub fn metered_segment_rvr_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
        num_airs: usize,
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<RvrMeteredSegmentInstance<'_>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered_segment", backend = "rvr").entered();
        let extensions = self.build_rvr_extensions(Some(executor_idx_to_air_idx));
        let chips = ChipMapping {
            num_airs,
            pc_to_chip: build_pc_to_chip(exe, &self.inventory, executor_idx_to_air_idx)
                .map_err(map_rvr_compile_error)?,
            chip_widths: None,
        };
        let compiled =
            compile_metered_segment_boundary(exe, extensions.lifters(), &chips, guest_debug_map)
                .map_err(map_rvr_compile_error)?;
        let runtime_hooks = extensions.into_runtime_hooks();

        Ok(RvrMeteredSegmentInstance::new(
            self.inventory.config(),
            RvrInitialImage::from(exe),
            runtime_hooks,
            compiled,
        ))
    }

    pub fn preflight_routed_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
        num_airs: usize,
        guest_debug_map: Option<&GuestDebugMap>,
        assembler_admitter: &dyn LogNativeOpcodeAdmitter<F>,
        gpu_records_default: Option<&str>,
    ) -> Result<RvrPreflightRoute<'_, F, VC::Executor>, StaticProgramError> {
        let extensions = self.build_rvr_extensions(Some(executor_idx_to_air_idx));
        // Pure capability routing: Rvr iff every opcode is on the rvr
        // preflight surface. The backend-keyed engine DEFAULT (CPU →
        // interpreter, GPU → rvr) is applied by the proving path in
        // `VmInstance::prove_continuations`, not here, so this stays usable
        // as a capability query and an explicit rvr constructor.
        match classify_preflight_opcodes_with_extensions(
            exe,
            extensions.lifters(),
            assembler_admitter,
        ) {
            RvrPreflightOpcodeClass::Supported => {
                #[cfg(feature = "metrics")]
                let _compilation_span =
                    tracing::info_span!("compile_preflight", backend = "rvr").entered();
                let chips = ChipMapping {
                    num_airs,
                    pc_to_chip: build_pc_to_chip(exe, &self.inventory, executor_idx_to_air_idx)
                        .map_err(map_rvr_compile_error)?,
                    chip_widths: None,
                };
                let compiled = super::rvr::compile::compile_preflight_with_extensions_and_default(
                    exe,
                    extensions.lifters(),
                    assembler_admitter,
                    &chips,
                    guest_debug_map,
                    gpu_records_default,
                )
                .map_err(map_rvr_compile_error)?;
                let runtime_hooks = extensions.into_runtime_hooks();
                Ok(RvrPreflightRoute::Rvr(RvrPreflightInstance::new(
                    self.inventory.config(),
                    Arc::new(exe.clone()),
                    runtime_hooks,
                    compiled,
                    &chips,
                )))
            }
            _ => {
                #[cfg(feature = "metrics")]
                let _compilation_span =
                    tracing::info_span!("compile_preflight", backend = "interpreter").entered();
                Ok(RvrPreflightRoute::Interpreter(
                    PreflightInterpretedInstance::new(
                        &exe.program,
                        self.inventory.clone(),
                        executor_idx_to_air_idx.to_vec(),
                    )?,
                ))
            }
        }
    }

    pub fn metered_cost_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
        widths: &[usize],
    ) -> Result<RvrMeteredCostInstance<'_>, StaticProgramError> {
        self.metered_cost_rvr_instance(exe, executor_idx_to_air_idx, widths, None)
    }

    /// Load a previously saved metered-mode artifact. Its generated execution
    /// kind is validated; the caller supplies matching `exe` and
    /// `executor_idx_to_air_idx`.
    pub fn load_metered_instance(
        &self,
        lib_path: &std::path::Path,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<RvrMeteredInstance<'_>, StaticProgramError> {
        let runtime_hooks = self
            .build_rvr_extensions(Some(executor_idx_to_air_idx))
            .into_runtime_hooks();
        let compiled = load_compiled_from_path(lib_path).map_err(map_rvr_compile_error)?;
        compiled
            .require_execution_kind(&[RvrExecutionKind::Metered])
            .map_err(map_rvr_compile_error)?;

        Ok(RvrMeteredInstance::new(
            self.inventory.config(),
            RvrInitialImage::from(exe),
            runtime_hooks,
            compiled,
        ))
    }

    /// Load a previously saved segment-boundary metered artifact. Its generated
    /// execution kind is validated; the caller supplies matching `exe` and
    /// `executor_idx_to_air_idx`.
    pub fn load_metered_segment_instance(
        &self,
        lib_path: &std::path::Path,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<RvrMeteredSegmentInstance<'_>, StaticProgramError> {
        let runtime_hooks = self
            .build_rvr_extensions(Some(executor_idx_to_air_idx))
            .into_runtime_hooks();
        let compiled = load_compiled_from_path(lib_path).map_err(map_rvr_compile_error)?;
        compiled
            .require_execution_kind(&[RvrExecutionKind::MeteredSegment])
            .map_err(map_rvr_compile_error)?;

        Ok(RvrMeteredSegmentInstance::new(
            self.inventory.config(),
            RvrInitialImage::from(exe),
            runtime_hooks,
            compiled,
        ))
    }

    /// Load a saved metered-cost artifact and check its execution kind and chip
    /// widths. The caller must provide matching `exe`,
    /// `executor_idx_to_air_idx`, and `widths`.
    pub fn load_metered_cost_instance(
        &self,
        lib_path: &std::path::Path,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
        widths: &[usize],
    ) -> Result<RvrMeteredCostInstance<'_>, StaticProgramError> {
        let compiled = load_compiled_from_path(lib_path).map_err(map_rvr_compile_error)?;
        compiled
            .require_chip_widths(widths)
            .map_err(map_rvr_compile_error)?;
        let runtime_hooks = self
            .build_rvr_extensions(Some(executor_idx_to_air_idx))
            .into_runtime_hooks();

        Ok(RvrMeteredCostInstance {
            system_config: self.inventory.config(),
            initial_image: RvrInitialImage::from(exe),
            runtime_hooks,
            compiled,
        })
    }

    pub fn metered_cost_rvr_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
        widths: &[usize],
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<RvrMeteredCostInstance<'_>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered_cost", backend = "rvr").entered();
        let extensions = self.build_rvr_extensions(Some(executor_idx_to_air_idx));
        let emitted_widths: Vec<u64> = widths.iter().map(|&width| width as u64).collect();
        let chips = ChipMapping {
            num_airs: emitted_widths.len(),
            pc_to_chip: build_pc_to_chip(exe, &self.inventory, executor_idx_to_air_idx)
                .map_err(map_rvr_compile_error)?,
            chip_widths: Some(emitted_widths),
        };
        let compiled = compile_metered_cost(exe, extensions.lifters(), &chips, guest_debug_map)
            .map_err(map_rvr_compile_error)?;
        let runtime_hooks = extensions.into_runtime_hooks();

        Ok(RvrMeteredCostInstance {
            system_config: self.inventory.config(),
            initial_image: RvrInitialImage::from(exe),
            runtime_hooks,
            compiled,
        })
    }
}

#[derive(Error, Debug)]
pub enum VmVerificationError<SC: StarkProtocolConfig> {
    #[error("no proof is provided")]
    ProofNotFound,

    #[error("program commit mismatch (index of mismatch proof: {index}")]
    ProgramCommitMismatch { index: usize },

    #[error("exe commit mismatch (expected: {expected:?}, actual: {actual:?})")]
    ExeCommitMismatch {
        expected: [u32; VM_DIGEST_WIDTH],
        actual: [u32; VM_DIGEST_WIDTH],
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
    StarkError(#[from] VerifierError<SC::EF>),

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
}

/// The [VirtualMachine] struct contains the API to generate proofs for _arbitrary_ programs for a
/// fixed set of OpenVM instructions and a fixed VM circuit corresponding to those instructions. The
/// API is specific to a particular [StarkEngine], which specifies a fixed [StarkProtocolConfig] and
/// [ProverBackend] via associated types. The [VmBuilder] also fixes the choice of
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
    #[cfg(feature = "rvr")]
    builder: VB,
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
        let chip_complex =
            builder.create_chip_complex(&config, circuit, engine.device().device_ctx())?;
        let executor = VmExecutor::<Val<E::SC>, _>::new(config)?;
        Ok(Self {
            engine,
            executor,
            pk: d_pk,
            chip_complex,
            #[cfg(feature = "rvr")]
            builder,
        })
    }

    pub fn new_with_keygen(
        engine: E,
        builder: VB,
        config: VB::VmConfig,
    ) -> Result<(Self, MultiStarkProvingKey<E::SC>), VirtualMachineError> {
        let circuit = config.create_airs()?;
        let pk = circuit.keygen(engine.config());
        let _vk = pk.get_vk();
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let vm = Self::new(engine, builder, config, d_pk)?;
        Ok((vm, pk))
    }

    pub fn config(&self) -> &VB::VmConfig {
        &self.executor.config
    }

    /// Pure interpreter.
    #[cfg(not(feature = "rvr"))]
    pub fn interpreter(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<InterpretedInstance<'_, ExecutionCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        self.executor().instance(exe)
    }

    #[cfg(feature = "rvr")]
    pub fn interpreter(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrPureInstance<'_>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        Self::get_rvr_instance(self, exe)
    }

    #[cfg(feature = "rvr")]
    pub fn get_rvr_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrPureInstance<'_>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        self.executor().rvr_instance(exe, None)
    }

    // Pure RVR execution with interpreter access for equivalence checking.
    #[cfg(feature = "rvr")]
    pub fn naive_interpreter(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<InterpretedInstance<'_, ExecutionCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        self.executor().interpreter_instance(exe)
    }

    #[cfg(not(feature = "rvr"))]
    pub fn metered_interpreter(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<InterpretedInstance<'_, MeteredCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_instance(exe, &executor_idx_to_air_idx)
    }

    #[cfg(feature = "rvr")]
    pub fn metered_interpreter(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredInstance<'_>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_instance(exe, &executor_idx_to_air_idx, self.num_airs())
    }

    #[cfg(feature = "rvr")]
    pub fn get_metered_rvr_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredInstance<'_>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_rvr_instance(exe, &executor_idx_to_air_idx, self.num_airs(), None)
    }

    #[cfg(feature = "rvr")]
    pub fn get_metered_segment_rvr_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredSegmentInstance<'_>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor().metered_segment_rvr_instance(
            exe,
            &executor_idx_to_air_idx,
            self.num_airs(),
            None,
        )
    }

    #[cfg(feature = "rvr")]
    pub fn load_metered_interpreter(
        &self,
        lib_path: &std::path::Path,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredInstance<'_>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .load_metered_instance(lib_path, exe, &executor_idx_to_air_idx)
    }

    #[cfg(feature = "rvr")]
    pub fn load_metered_segment_instance(
        &self,
        lib_path: &std::path::Path,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredSegmentInstance<'_>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .load_metered_segment_instance(lib_path, exe, &executor_idx_to_air_idx)
    }

    #[cfg(feature = "rvr")]
    pub fn naive_metered_interpreter(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<InterpretedInstance<'_, MeteredCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_interpreter_instance(exe, &executor_idx_to_air_idx)
    }

    #[cfg(not(feature = "rvr"))]
    pub fn metered_cost_interpreter(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<InterpretedInstance<'_, MeteredCostCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_cost_instance(exe, &executor_idx_to_air_idx)
    }

    #[cfg(feature = "rvr")]
    pub fn metered_cost_interpreter(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredCostInstance<'_>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        Self::get_metered_cost_rvr_instance(self, exe)
    }

    #[cfg(feature = "rvr")]
    pub fn get_metered_cost_rvr_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredCostInstance<'_>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        let widths: Vec<usize> = self
            .pk
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.total_width())
            .collect();
        self.executor()
            .metered_cost_rvr_instance(exe, &executor_idx_to_air_idx, &widths, None)
    }

    #[cfg(feature = "rvr")]
    pub fn load_metered_cost_interpreter(
        &self,
        lib_path: &std::path::Path,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredCostInstance<'_>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        let widths: Vec<usize> = self
            .pk
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.total_width())
            .collect();
        self.executor()
            .load_metered_cost_instance(lib_path, exe, &executor_idx_to_air_idx, &widths)
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

    #[cfg(feature = "rvr")]
    pub fn preflight_routed_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<VmRvrPreflightRoute<'_, Val<E::SC>, VB::VmConfig>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        self.preflight_routed_instance_with_gpu_records_default(exe, None)
    }

    #[cfg(feature = "rvr")]
    fn preflight_routed_instance_with_gpu_records_default(
        &self,
        exe: &VmExe<Val<E::SC>>,
        gpu_records_default: Option<&str>,
    ) -> Result<VmRvrPreflightRoute<'_, Val<E::SC>, VB::VmConfig>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        let assembler_registry = self
            .builder
            .create_rvr_log_native_assembler_registry(self.config());
        self.executor().preflight_routed_instance(
            exe,
            &executor_idx_to_air_idx,
            self.num_airs(),
            None,
            &assembler_registry,
            gpu_records_default,
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
        state: VmState<GuestMemory>,
        trace_heights: &[u32],
    ) -> Result<PreflightExecutionOutput<Val<E::SC>, VB::RecordArena>, ExecutionError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
            PreflightExecutor<Val<E::SC>, VB::RecordArena>,
    {
        self.execute_preflight_inner(interpreter, state, None, trace_heights)
    }

    /// Preflight execution for at most `num_insns` instructions from the given state.
    /// `system_records.exit_code` is `Some` when the program terminated and `None` when execution
    /// stopped at the instruction limit.
    #[instrument(name = "execute_preflight", skip_all)]
    pub fn execute_preflight_for(
        &self,
        interpreter: &mut PreflightInterpretedInstance2<Val<E::SC>, VB::VmConfig>,
        state: VmState<GuestMemory>,
        num_insns: u64,
        trace_heights: &[u32],
    ) -> Result<PreflightExecutionOutput<Val<E::SC>, VB::RecordArena>, ExecutionError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
            PreflightExecutor<Val<E::SC>, VB::RecordArena>,
    {
        self.execute_preflight_inner(interpreter, state, Some(num_insns), trace_heights)
    }

    fn execute_preflight_inner(
        &self,
        interpreter: &mut PreflightInterpretedInstance2<Val<E::SC>, VB::VmConfig>,
        state: VmState<GuestMemory>,
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

        let capacities = self.preflight_capacities(trace_heights);
        let ctx = PreflightCtx::new_with_capacity(&capacities, num_insns);

        let pc = state.pc();
        let memory = TracingMemory::from_image(state.memory);
        let from_state = ExecutionState::new(pc, memory.timestamp());
        let vm_state = VmState::new(
            pc,
            memory,
            state.streams,
            state.rng,
            #[cfg(feature = "metrics")]
            state.metrics,
        );
        let mut exec_state = VmExecState::new(vm_state, ctx);
        interpreter.reset_execution_frequencies();
        execute_spanned!("execute_preflight", interpreter, &mut exec_state)?;
        let filtered_exec_frequencies = interpreter.filtered_execution_frequencies();
        #[cfg(feature = "metrics")]
        emit_opcode_counts(
            &exec_state.vm_state.metrics,
            interpreter.opcode_counts_by_air::<VB::RecordArena>(),
        );
        let touched_memory = exec_state.vm_state.memory.finalize::<Val<E::SC>>();
        // Keep the sparse GPU initial-memory image in sync with the state carried to the next
        // continuation segment. Interpreter preflight writes through TracingMemory, whose
        // finalized touched blocks are the authoritative record of pages that may now be
        // non-zero.
        exec_state
            .vm_state
            .memory
            .data
            .memory
            .extend_touched_pages_from_touched(&touched_memory);
        #[cfg(feature = "perf-metrics")]
        end_segment_metrics(&mut exec_state);

        let pc = exec_state.vm_state.pc();
        let memory = exec_state.vm_state.memory;
        let to_state = ExecutionState::new(pc, memory.timestamp());
        let exit_code = exec_state.exit_code?;
        let system_records = SystemRecords {
            from_state,
            to_state,
            exit_code,
            filtered_exec_frequencies,
            program_frequencies_on_device: false,
            #[cfg(feature = "rvr")]
            rvr_exec_frequencies_touched: Vec::new(),
            #[cfg(feature = "rvr")]
            rvr_exec_frequencies_pool: None,
            touched_memory,
            touched_memory_on_device: false,
            device_replay_oracle: false,
        };
        let record_arenas = exec_state.ctx.arenas;
        let to_state = VmState::new(
            pc,
            memory.data,
            exec_state.vm_state.streams,
            exec_state.vm_state.rng,
            #[cfg(feature = "metrics")]
            exec_state.vm_state.metrics,
        );
        Ok(PreflightExecutionOutput {
            system_records,
            record_arenas,
            to_state,
        })
    }

    #[cfg(feature = "rvr")]
    fn execute_rvr_preflight_for_proving(
        &self,
        rvr_preflight: &CachedRvrPreflight<Val<E::SC>>,
        interpreter: &mut PreflightInterpretedInstance2<Val<E::SC>, VB::VmConfig>,
        exe: &VmExe<Val<E::SC>>,
        state: VmState<GuestMemory>,
        num_insns: Option<u64>,
        trace_heights: &[u32],
    ) -> Result<PreflightExecutionOutput<Val<E::SC>, VB::RecordArena>, ExecutionError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
            MeteredExecutor<Val<E::SC>> + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
    {
        match rvr_preflight {
            CachedRvrPreflight::Rvr(rvr_preflight) => {
                let _preflight_span = tracing::info_span!("execute_preflight").entered();
                // `state` moves in whole: `execute_rvr_preflight` clones it
                // internally for its capacity-retry loop, and a guest-state
                // clone is the dominant per-segment fixed cost (~hundreds of
                // ms), so it must not be paid twice.
                let detail_wrapper_started = std::time::Instant::now();
                #[cfg(any(feature = "stark-debug", feature = "cuda"))]
                let split_t0 = std::time::Instant::now();
                let detailed_profile =
                    std::env::var("OPENVM_RVR_PREFLIGHT_PROFILE_DETAIL").as_deref() == Ok("1");
                let capacities = self.preflight_capacities(trace_heights);
                let capacities_finished = std::time::Instant::now();
                let pc_to_air_idx = rvr_preflight.pc_to_air_idx();
                // R4: stage arena-native targets so the C writes those airs'
                // full records directly into their final arena backings.
                let arena_native = rvr_preflight.arena_native_airs().to_vec();
                let mut staged: Vec<(usize, ArenaNativeGeometry, VB::RecordArena)> = Vec::new();
                let mut targets = std::collections::BTreeMap::new();
                for &(air, geometry) in &arena_native {
                    let (height, width) = capacities[air];
                    let (arena, buf) =
                        <VB::RecordArena as RvrArenaNativeTarget>::stage_arena_native_pooled(
                            height,
                            width,
                            &geometry,
                            air,
                            rvr_preflight.buffer_pool(),
                        );
                    targets.insert(air, buf);
                    staged.push((air, geometry, arena));
                }
                let arena_staging_finished = std::time::Instant::now();
                // G2: stage compact WIRE targets for the airs the builder
                // requests — the C writes packed wire records straight into
                // the adopted backing (one alloc, no copy); the chips decode
                // them (GPU on-device). A requested air must be compiled
                // compact: wire-staging a fused-compiled air would hand the C
                // a wire-stride descriptor for full-record emission, so the
                // toggle mismatch fails here instead of corrupting records.
                let wire_airs = rvr_preflight.wire_airs();
                let mut staged_wire: Vec<(usize, usize, VB::RecordArena)> = Vec::new();
                if !wire_airs.is_empty() {
                    let inline_wire = rvr_preflight.inline_wire_airs().to_vec();
                    for &air in wire_airs {
                        if arena_native
                            .iter()
                            .any(|&(native_air, _)| native_air == air)
                        {
                            return Err(ExecutionError::RvrExecution(format!(
                                "air {air} is compiled arena-native but the builder requested \
                                 compact wire staging; recompile with the compact opt-in \
                                 (OPENVM_RVR_GPU_RECORDS=compact) or drop the request"
                            )));
                        }
                        let &(_, wire_size) = inline_wire
                            .iter()
                            .find(|&&(wire_air, _)| wire_air == air)
                            .ok_or_else(|| {
                                ExecutionError::RvrExecution(format!(
                                    "builder requested wire staging for air {air} but the \
                                     compiled library emits no inline records for it"
                                ))
                            })?;
                        let (arena, buf) =
                            <VB::RecordArena as RvrArenaNativeTarget>::stage_rvr_wire_pooled(
                                capacities[air].0,
                                wire_size,
                                air,
                                rvr_preflight.buffer_pool(),
                            );
                        targets.insert(air, buf);
                        staged_wire.push((air, wire_size, arena));
                    }
                }
                let wire_staging_finished = std::time::Instant::now();
                let staged_count = staged.len();
                let staged_wire_count = staged_wire.len();
                let staged_capacity_bytes = targets
                    .values()
                    .map(|target| u64::from(target.cap))
                    .sum::<u64>();
                let mut rvr_output = rvr_preflight.execute(
                    exe,
                    state,
                    num_insns,
                    Some(trace_heights),
                    (!targets.is_empty()).then_some(&targets),
                )?;
                let execute_call_finished = std::time::Instant::now();
                #[cfg(any(feature = "stark-debug", feature = "cuda"))]
                let split_t1 = std::time::Instant::now();
                #[cfg(any(feature = "stark-debug", feature = "cuda"))]
                let split_t2 = std::time::Instant::now();
                #[cfg(feature = "stark-debug")]
                if std::env::var("OPENVM_STARK_DEBUG_TRACE_ONLY").as_deref() == Ok("1")
                    && std::env::var("OPENVM_STARK_DEBUG_RVR_AIR_ROUTES").as_deref() == Ok("1")
                {
                    let mut routed = vec![[0u64; 3]; self.pk.per_air.len()];
                    let arena_native = rvr_output
                        .arena_native_written
                        .iter()
                        .map(|&(air, _)| air)
                        .collect::<std::collections::BTreeSet<_>>();
                    let pc_base = u64::from(exe.program.pc_base);
                    let pc_step = u64::from(openvm_instructions::program::DEFAULT_PC_STEP);
                    for entry in &rvr_output.raw_logs.program_log {
                        let Some(slot) = u64::from(entry.pc())
                            .checked_sub(pc_base)
                            .map(|offset| (offset / pc_step) as usize)
                        else {
                            continue;
                        };
                        let Some(air) = pc_to_air_idx.get(slot).copied().flatten() else {
                            continue;
                        };
                        let route = if rvr_output
                            .inline_pc_slots
                            .get(slot)
                            .copied()
                            .unwrap_or(false)
                        {
                            usize::from(!arena_native.contains(&air))
                        } else {
                            2
                        };
                        routed[air][route] += 1;
                    }
                    for (air, [native, compact, verbose]) in routed.into_iter().enumerate() {
                        if native + compact + verbose == 0 {
                            continue;
                        }
                        eprintln!(
                            "OPENVM_STARK_DEBUG_RVR_AIR_ROUTE air={air} name={:?} native={} \
                             compact={} verbose={}",
                            self.pk.per_air[air].air_name, native, compact, verbose
                        );
                    }
                }
                let generic_assembly_started = std::time::Instant::now();
                let rvr_g2_segment_id = rvr_output
                    .g2_segment
                    .as_ref()
                    .map(|segment| segment.header_acquire().map(|header| header.segment_id))
                    .transpose()?;
                let mut record_arenas = self
                    .builder
                    .generate_rvr_record_arenas_from_logs(
                        self.config(),
                        exe,
                        &mut rvr_output,
                        &capacities,
                        pc_to_air_idx,
                    )?
                    .ok_or_else(|| {
                        ExecutionError::RvrExecution(
                            "rvr log-native tracegen is not implemented for this VM builder"
                                .to_string(),
                        )
                    })?;
                let generic_assembly_finished = std::time::Instant::now();
                for (air, geometry, mut arena) in staged {
                    let written = rvr_output
                        .arena_native_written
                        .iter()
                        .find(|&&(written_air, _)| written_air == air)
                        .map(|&(_, count)| count as usize)
                        .ok_or_else(|| {
                            ExecutionError::RvrExecution(format!(
                                "arena-native air {air} reported no written count"
                            ))
                        })?;
                    let written_bytes = rvr_output
                        .arena_native_written_bytes
                        .iter()
                        .find(|&&(written_air, _)| written_air == air)
                        .map(|&(_, bytes)| bytes as usize)
                        .ok_or_else(|| {
                            ExecutionError::RvrExecution(format!(
                                "arena-native air {air} reported no written byte cursor"
                            ))
                        })?;
                    arena.finish_arena_native_sized(written, written_bytes, &geometry);
                    if let Some(segment_id) = rvr_g2_segment_id {
                        arena.set_rvr_g2_segment_id(segment_id);
                    }
                    record_arenas[air] = arena;
                }
                let arena_finish_finished = std::time::Instant::now();
                for (air, wire_size, mut arena) in staged_wire {
                    let written = rvr_output
                        .arena_native_written
                        .iter()
                        .find(|&&(written_air, _)| written_air == air)
                        .map(|&(_, count)| count as usize)
                        .ok_or_else(|| {
                            ExecutionError::RvrExecution(format!(
                                "wire-staged air {air} reported no written count"
                            ))
                        })?;
                    arena.finish_rvr_wire(written, wire_size);
                    record_arenas[air] = arena;
                }
                let wire_finish_finished = std::time::Instant::now();
                #[cfg(any(feature = "stark-debug", feature = "cuda"))]
                if std::env::var("OPENVM_STARK_DEBUG_TRACE_ONLY").as_deref() == Ok("1")
                    || std::env::var("OPENVM_GPU_E2E_PROFILE").as_deref() == Ok("1")
                {
                    let split_t3 = std::time::Instant::now();
                    eprintln!(
                        "OPENVM_STARK_DEBUG_RVR_PREFLIGHT_SPLIT cexec_us={} setup_us={} \
                         assembly_us={}",
                        (split_t1 - split_t0).as_micros(),
                        (split_t2 - split_t1).as_micros(),
                        (split_t3 - split_t2).as_micros()
                    );
                }

                // The arenas hold expanded records; the raw logs and compact
                // record bytes are dead — return them to the executor's pool
                // so the next segment skips their fresh-mapping fault cost.
                let RvrPreflightOutput {
                    system_records,
                    to_state,
                    raw_logs,
                    access_aux,
                    inline_records,
                    delta_records,
                    ..
                } = rvr_output;
                let recycle_started = std::time::Instant::now();
                rvr_preflight.recycle_access_aux(access_aux);
                rvr_preflight.recycle_segment_buffers(raw_logs, inline_records, delta_records);
                let recycle_finished = std::time::Instant::now();
                if detailed_profile {
                    eprintln!(
                        "OPENVM_RVR_WRAPPER_DETAIL capacities_us={} arena_stage_us={} \
                         wire_stage_us={} execute_call_us={} generic_assembly_us={} \
                         arena_finish_us={} wire_finish_us={} recycle_us={} staged_airs={} \
                         staged_wire_airs={} staged_capacity_bytes={}",
                        (capacities_finished - detail_wrapper_started).as_micros(),
                        (arena_staging_finished - capacities_finished).as_micros(),
                        (wire_staging_finished - arena_staging_finished).as_micros(),
                        (execute_call_finished - wire_staging_finished).as_micros(),
                        (generic_assembly_finished - generic_assembly_started).as_micros(),
                        (arena_finish_finished - generic_assembly_finished).as_micros(),
                        (wire_finish_finished - arena_finish_finished).as_micros(),
                        (recycle_finished - recycle_started).as_micros(),
                        staged_count,
                        staged_wire_count,
                        staged_capacity_bytes,
                    );
                }

                Ok(PreflightExecutionOutput {
                    system_records,
                    record_arenas,
                    to_state,
                })
            }
            CachedRvrPreflight::Interpreter => {
                self.execute_preflight_inner(interpreter, state, num_insns, trace_heights)
            }
        }
    }

    fn preflight_capacities(&self, trace_heights: &[u32]) -> Vec<(usize, usize)> {
        // TODO[jpw]: figure out how to compute RA specific main_widths
        let main_widths = self
            .pk
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.main_width())
            .collect_vec();
        zip_eq(trace_heights, main_widths)
            .map(|(&h, w)| (h as usize, w))
            .collect()
    }

    #[cfg(feature = "rvr")]
    /// Maps defined program instruction slots to AIR IDs for rvr log-native record routing.
    pub fn pc_to_air_idx(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<Vec<Option<usize>>, StaticProgramError> {
        let terminate_opcode = SystemOpcode::TERMINATE.global_opcode();
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        exe.program
            .instructions_and_debug_infos
            .iter()
            .map(|slot| {
                let Some((inst, _)) = slot else {
                    return Ok(None);
                };
                if inst.opcode == terminate_opcode {
                    return Ok(None);
                }
                let executor_idx = *self
                    .executor
                    .inventory
                    .instruction_lookup
                    .get(&inst.opcode)
                    .ok_or(StaticProgramError::ExecutorNotFound {
                        opcode: inst.opcode,
                    })? as usize;
                Ok(Some(executor_idx_to_air_idx[executor_idx]))
            })
            .collect()
    }

    /// Calls [`VmState::initial`] but sets more information for
    /// performance metrics when feature "perf-metrics" is enabled.
    #[instrument(name = "vm.create_initial_state", level = "debug", skip_all)]
    pub fn create_initial_state(
        &self,
        exe: &VmExe<Val<E::SC>>,
        inputs: impl Into<Streams>,
    ) -> VmState<GuestMemory> {
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
        #[cfg(feature = "metrics")]
        {
            state.metrics.set_pk_air_names(&self.pk);
        }
        #[cfg(feature = "perf-metrics")]
        {
            state.metrics.set_pk_trace_info(&self.pk);
            state.metrics.num_sys_airs = self.config().as_ref().num_airs();
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
        // main tracegen call:
        let ctx = self
            .chip_complex
            .generate_proving_ctx(system_records, record_arenas)?;

        // ==== Defensive checks that the trace heights satisfy the linear constraints: ====
        let idx_trace_heights = ctx
            .per_trace
            .iter()
            .map(|(air_idx, ctx)| (*air_idx, ctx.common_main.height()))
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
        // OPENVM_SKIP_DEBUG=1 skips debug-mode constraint checking, consistent with its
        // meaning in `stark_utils::air_test`; the checker otherwise dominates trace-only
        // timing runs (~tens of seconds per segment at reth scale).
        #[cfg(feature = "stark-debug")]
        if std::env::var("OPENVM_SKIP_DEBUG").as_deref() != Ok("1") {
            self.debug_proving_ctx(&ctx);
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
        interpreter: &mut PreflightInterpretedInstance2<Val<E::SC>, VB::VmConfig>,
        state: VmState<GuestMemory>,
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
        } = self.execute_preflight_inner(interpreter, state, num_insns, trace_heights)?;
        // drop final memory unless this is a terminal segment and the exit code is success
        let final_memory =
            (system_records.exit_code == Some(ExitCode::Success as u32)).then_some(to_state.memory);
        let ctx = self.generate_proving_ctx(system_records, record_arenas)?;
        let proof = self.engine.prove(&self.pk, ctx).unwrap();

        Ok((proof, final_memory))
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
        let rm_trace = generate_cached_trace(program);
        let cm_trace = ColMajorMatrix::from_row_major(&rm_trace);
        let d_trace = self.engine.device().transport_matrix_to_device(&cm_trace);
        let (commitment, pcs) = self
            .engine
            .device()
            .commit(std::slice::from_ref(&&d_trace))
            .unwrap();
        CommittedTraceData {
            commitment,
            trace: d_trace,
            data: Arc::new(pcs),
        }
    }

    /// Loads cached program trace into the VM.
    pub fn load_program(&mut self, cached_program_trace: CommittedTraceData<E::PB>) {
        self.chip_complex.system.load_program(cached_program_trace);
    }

    #[instrument(name = "vm.transport_init_memory", skip_all)]
    pub fn transport_init_memory_to_device(&mut self, memory: &GuestMemory) {
        self.chip_complex
            .system
            .transport_init_memory_to_device(memory);
    }

    /// Consume the device-replayed write-only page bitmap into the carried
    /// state. The segment loop calls this immediately after trace generation,
    /// before any subsequent initial-memory H2D can observe the state.
    pub fn merge_device_continuation_dirty_pages(&mut self, memory: &mut GuestMemory) {
        self.chip_complex
            .system
            .merge_device_continuation_dirty_pages(memory);
    }

    /// See [`SystemChipComplex::memory_top_tree`].
    pub fn memory_top_tree(&self) -> Option<&[[Val<E::SC>; VM_DIGEST_WIDTH]]> {
        self.chip_complex.system.memory_top_tree()
    }

    pub fn executor_idx_to_air_idx(&self) -> Vec<usize> {
        let ret = self.chip_complex.inventory.executor_idx_to_air_idx();
        tracing::debug!("executor_idx_to_air_idx: {:?}", ret);
        assert_eq!(self.executor().inventory.executors().len(), ret.len());
        ret
    }

    /// Convenience method to construct a [MeteredCtx] using data from the stored proving key.
    pub fn build_metered_ctx(&self, exe: &VmExe<Val<E::SC>>) -> MeteredCtx
    where
        Val<E::SC>: PrimeField32,
    {
        let program_len = exe.program.num_defined_instructions();

        let (mut constant_trace_heights, air_names, widths, interactions, need_rot): (
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
        ) = self
            .pk
            .per_air
            .iter()
            .map(|pk| {
                let constant_trace_height = pk.preprocessed_data.as_ref().map(|cd| cd.height());
                let air_names = pk.air_name.clone();
                let width = pk.vk.params.width.total_width();
                let num_interactions = pk.vk.symbolic_constraints.interactions.len();
                let need_rot = pk.vk.params.need_rot;
                (
                    constant_trace_height,
                    air_names,
                    width,
                    num_interactions,
                    need_rot,
                )
            })
            .multiunzip();

        // Program trace is the same for all segments
        constant_trace_heights[PROGRAM_AIR_ID] = Some(program_len);
        // VmConnectorAir always has a constant trace height of 2
        constant_trace_heights[CONNECTOR_AIR_ID] = Some(2);
        // Merge in constant heights reported by chips (e.g., lookup table chips).
        for (air_id, chip_height) in self
            .chip_complex
            .inventory
            .constant_trace_heights()
            .into_iter()
            .enumerate()
        {
            if constant_trace_heights[air_id].is_none() {
                constant_trace_heights[air_id] = chip_height;
            }
        }

        let log_stacked_height = self
            .engine
            .params()
            .log_stacked_height()
            .try_into()
            .expect("log_stacked_height must fit in u8");
        let mut ctx = self.executor().build_metered_ctx(
            MeteredCtxInputs {
                constant_trace_heights: &constant_trace_heights,
                air_names: &air_names,
                widths: &widths,
                interactions: &interactions,
                need_rot: &need_rot,
                segmentation_limits: SegmentationLimits {
                    max_trace_height_bits: log_stacked_height,
                    max_memory: self.config().as_ref().segmentation_max_memory,
                    max_interactions: <Val<E::SC> as PrimeField32>::ORDER_U32,
                },
            },
            self.engine.proving_memory_config(),
        );
        ctx.seed_initial_memory(&exe.init_memory);
        ctx
    }

    /// Convenience method to construct a [MeteredCostCtx] using data from the stored proving key.
    pub fn build_metered_cost_ctx(&self) -> MeteredCostCtx {
        let widths: Vec<_> = self
            .pk
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.total_width())
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
        debug_proving_ctx(self, ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::{SystemConfig, VirtualMachine, CONNECTOR_AIR_ID, PROGRAM_AIR_ID};
    use crate::{system::SystemCpuBuilder, utils::test_cpu_engine};

    #[test]
    fn keygen_marks_required_airs_for_continuations() {
        let engine = test_cpu_engine();
        let config = SystemConfig::default();
        let merkle_air_id = config.memory_merkle_air_id();
        let boundary_air_id = config.memory_boundary_air_id();

        let (_vm, pk) = VirtualMachine::new_with_keygen(engine, SystemCpuBuilder, config).unwrap();

        assert!(pk.per_air[PROGRAM_AIR_ID].vk.is_required);
        assert!(pk.per_air[CONNECTOR_AIR_ID].vk.is_required);
        assert!(pk.per_air[merkle_air_id].vk.is_required);
        assert!(pk.per_air[boundary_air_id].vk.is_required);
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Com<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>"
))]
pub struct ContinuationVmProof<SC: StarkProtocolConfig> {
    pub per_segment: Vec<Proof<SC>>,
    pub user_public_values: UserPublicValuesProof<{ VM_DIGEST_WIDTH }, Val<SC>>,
}

/// Prover for a specific exe in a specific continuation VM using a specific Stark config.
pub trait ContinuationVmProver<SC: StarkProtocolConfig> {
    fn prove(
        &mut self,
        input: impl Into<Streams>,
    ) -> Result<ContinuationVmProof<SC>, VirtualMachineError>;
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
    #[cfg(feature = "rvr")]
    rvr_metered: Option<Box<dyn CachedRvrMeteredExecutor>>,
    #[cfg(feature = "rvr")]
    rvr_preflight: Option<CachedRvrPreflight<Val<E::SC>>>,
    #[cfg(feature = "rvr")]
    rvr_preflight_engine: Option<RvrPreflightEngine>,
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    rvr_cuda_module_prewarmed: bool,
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    rvr_cuda_paths_prewarmed: bool,
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    rvr_cuda_pool_reserve_bytes: usize,
    #[getset(get = "pub")]
    program_commitment: <E::PB as ProverBackend>::Commitment,
    #[getset(get = "pub")]
    exe: Arc<VmExe<Val<E::SC>>>,
    #[getset(get = "pub", get_mut = "pub")]
    state: Option<VmState<GuestMemory>>,
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
        let program_commitment = cached_program_trace.commitment;
        vm.load_program(cached_program_trace);
        let interpreter = vm.preflight_interpreter(&exe)?;
        let state = vm.create_initial_state(&exe, vec![]);
        Ok(Self {
            vm,
            interpreter,
            #[cfg(feature = "rvr")]
            rvr_metered: None,
            #[cfg(feature = "rvr")]
            rvr_preflight: None,
            #[cfg(feature = "rvr")]
            rvr_preflight_engine: None,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            rvr_cuda_module_prewarmed: false,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            rvr_cuda_paths_prewarmed: false,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            rvr_cuda_pool_reserve_bytes: 0,
            program_commitment,
            exe,
            state: Some(state),
        })
    }

    /// Per-instance preflight engine override, strongest in the resolution
    /// order (instance override > `OPENVM_RVR_PREFLIGHT_ENGINE` > the
    /// builder's backend-keyed default). `None` restores default resolution.
    ///
    /// Must be called before proving: the routed engine is cached at the
    /// first proven segment and reused for the instance's lifetime.
    #[cfg(feature = "rvr")]
    pub fn set_rvr_preflight_engine(&mut self, engine: Option<RvrPreflightEngine>) {
        assert!(
            self.rvr_preflight.is_none(),
            "preflight engine already resolved for this instance; set the override before proving"
        );
        self.rvr_preflight_engine = engine;
    }

    /// Compile and cache the program-specialized RVR executors used by app
    /// proving. Callers can invoke this before entering a timed proving span;
    /// repeated calls are no-ops.
    #[cfg(feature = "rvr")]
    pub fn warm_rvr_proving(&mut self) -> Result<(), VirtualMachineError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
            MeteredExecutor<Val<E::SC>> + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
    {
        #[cfg(all(feature = "cuda", feature = "rvr"))]
        let cuda_module_prewarm = (!self.rvr_cuda_module_prewarmed
            && rvr_cuda_g2_module_prewarm_enabled())
        .then(|| self.vm.builder.rvr_cuda_device_prewarm_task())
        .flatten()
        .map(std::thread::spawn);
        if self.rvr_metered.is_none() {
            let metered = self.vm.metered_interpreter(&self.exe)?.into_owned();
            self.rvr_metered = Some(Box::new(metered));
        }
        if self.rvr_preflight.is_some() {
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            if let Some(task) = cuda_module_prewarm {
                task.join()
                    .map_err(|_| {
                        ExecutionError::RvrExecution(
                            "CUDA device prewarm worker panicked".to_string(),
                        )
                    })?
                    .map_err(ExecutionError::RvrExecution)?;
                self.rvr_cuda_module_prewarmed = true;
            }
            return Ok(());
        }

        let vm = &mut self.vm;
        // Engine resolution: per-instance override, then the
        // OPENVM_RVR_PREFLIGHT_ENGINE environment override, then the
        // builder's backend-keyed default (CPU → interpreter, GPU → rvr;
        // rationale on `RvrPreflightEngine`). `Rvr` remains subject to
        // opcode-capability routing.
        let engine = self
            .rvr_preflight_engine
            .or_else(rvr_preflight_engine_env_override)
            .unwrap_or_else(|| vm.builder.default_rvr_preflight_engine());
        let gpu_records_default =
            (vm.builder.default_rvr_preflight_engine() == RvrPreflightEngine::Rvr).then_some("g2");
        let cached = match engine {
            RvrPreflightEngine::Interpreter => CachedRvrPreflight::Interpreter,
            RvrPreflightEngine::Rvr => match vm.preflight_routed_instance_with_gpu_records_default(
                &self.exe,
                gpu_records_default,
            )? {
                RvrPreflightRoute::Rvr(RvrPreflightInstance {
                    runtime_hooks,
                    compiled,
                    chip_counts_len,
                    pool,
                    ..
                }) => {
                    let pc_to_air_idx = vm.pc_to_air_idx(&self.exe)?;
                    let mut wire_airs = vm
                        .builder
                        .rvr_wire_record_airs(
                            vm.config(),
                            &self.exe,
                            &pc_to_air_idx,
                            compiled.inline_records(),
                        )
                        .into_iter()
                        .collect::<Vec<_>>();
                    // Compiler-scope proof, including mixed-AIR taint: every
                    // defined routed slot must both emit inline and belong to
                    // a whole AIR owned by the device or an arena-native
                    // consumer. Otherwise retain the complete host access
                    // replay and fail-closed generic assembly path.
                    let fully_direct_delta = compiled.inline_records().fully_direct_delta;
                    let g2 = compiled.inline_records().g2.is_some();
                    // The AOT classification is backend-neutral. CPU builders
                    // intentionally advertise no delta-wire AIRs and must
                    // retain the full host replay. The only safe empty-wire
                    // exception is an all-custom arena route, whose AOT decode
                    // map is empty because no delta record can reach
                    // finalization.
                    let has_device_delta_route = !wire_airs.is_empty();
                    let all_custom_arena = compiled
                        .inline_records()
                        .delta_decode
                        .as_ref()
                        .is_some_and(|precomputed| precomputed.kind_to_air.is_empty());
                    if compiled.inline_records().delta_records || g2 {
                        // Stage-2 uses one cross-AIR chronological backing,
                        // not the per-AIR G2 wire targets.
                        wire_airs.clear();
                    }
                    let compact_delta_memory = fully_direct_delta && has_device_delta_route;
                    let device_touched_memory =
                        (fully_direct_delta || g2) && has_device_delta_route;
                    wire_airs.sort_unstable();
                    CachedRvrPreflight::Rvr(Box::new(CachedRvrCompiledPreflight {
                        compiled,
                        runtime_hooks,
                        chip_counts_len,
                        pool,
                        pc_to_air_idx,
                        wire_airs,
                        build_access_aux: !((fully_direct_delta || g2)
                            && (has_device_delta_route || all_custom_arena)),
                        compact_delta_memory,
                        device_touched_memory,
                    }))
                }
                RvrPreflightRoute::Interpreter(_) => CachedRvrPreflight::Interpreter,
            },
        };
        self.rvr_preflight = Some(cached);
        #[cfg(all(feature = "cuda", feature = "rvr"))]
        if let Some(task) = cuda_module_prewarm {
            task.join()
                .map_err(|_| {
                    ExecutionError::RvrExecution("CUDA device prewarm worker panicked".to_string())
                })?
                .map_err(ExecutionError::RvrExecution)?;
            self.rvr_cuda_module_prewarmed = true;
        }
        Ok(())
    }

    #[instrument(name = "vm.reset_state", level = "debug", skip_all)]
    pub fn reset_state(&mut self, inputs: impl Into<Streams>) {
        let state = self.state.as_mut().unwrap();
        state.reset(&self.exe.init_memory, self.exe.pc_start, inputs);

        #[cfg(all(feature = "metrics", any(feature = "perf-metrics", debug_assertions)))]
        {
            state.metrics.fn_bounds = self.exe.fn_bounds.clone();
            state.metrics.debug_infos = self.exe.program.debug_infos();
        }
    }
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn rvr_cuda_device_prewarm_depth() -> usize {
    std::env::var("OPENVM_RVR_CUDA_DEVICE_PREWARM_DEPTH")
        .or_else(|_| std::env::var("OPENVM_RVR_CUDA_G2_PREWARM_DEPTH"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        // Prime the exact pinned-host and trace-generation shapes used by the first three
        // segments. This is no longer a proof replay: it registers fresh arena allocations in the
        // foreground, launches trace kernels, and drains buffers before real segment zero.
        .unwrap_or(3)
        .min(8)
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn rvr_cuda_g2_module_prewarm_enabled() -> bool {
    let g2_route = std::env::var("OPENVM_RVR_GPU_RECORDS").map_or(true, |records| records == "g2");
    g2_route
        && std::env::var("OPENVM_RVR_CUDA_MODULE_PREWARM").map_or(true, |enabled| enabled != "0")
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn rvr_cuda_device_pool_prewarm_bytes(max_g2_capacity_bytes: usize) -> usize {
    std::env::var("OPENVM_RVR_CUDA_DEVICE_POOL_PREWARM_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        // The decoder concurrently owns its compact inputs, chronology, and
        // radix-sort workspace. Keep the default shape-scaled while leaving a
        // direct override for deployments whose G2 mix needs a different
        // retention target.
        .unwrap_or_else(|| max_g2_capacity_bytes.saturating_mul(3))
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
impl<E, VB> VmInstance<E, VB>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    VB: VmBuilder<E>,
    <E::PD as ProverDevice<E::PB, E::TS>>::DeviceCtx: 'static,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
        + MeteredExecutor<Val<E::SC>>
        + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
{
    fn prewarm_cuda_device_paths(
        &mut self,
        input: Streams,
        segments: &[Segment],
        selected: &[usize],
        device_pool_reserve_bytes: usize,
    ) -> Result<(), VirtualMachineError> {
        if selected.is_empty() {
            self.vm
                .builder
                .finish_rvr_cuda_device_prewarm(device_pool_reserve_bytes)
                .map_err(ExecutionError::RvrExecution)?;
            self.rvr_cuda_paths_prewarmed = true;
            self.rvr_cuda_pool_reserve_bytes = device_pool_reserve_bytes;
            return Ok(());
        }
        let selected_set = selected
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let last_selected = *selected_set
            .last()
            .expect("non-empty CUDA warm selection has a final segment");
        let started = std::time::Instant::now();
        let mut state = self.state.take();
        for (seg_idx, segment) in segments.iter().enumerate() {
            let from_state = state.take().expect("CUDA warm pass state missing");
            let warm_this_segment = selected_set.contains(&seg_idx);
            if warm_this_segment {
                self.vm.transport_init_memory_to_device(&from_state.memory);
            }
            let rvr_preflight = self
                .rvr_preflight
                .as_ref()
                .expect("RVR preflight executor must be warmed before CUDA device warmup");
            let execute = || {
                if let CachedRvrPreflight::Rvr(rvr) = rvr_preflight {
                    rvr.prepare_wire_backings(&segment.trace_heights);
                }
                self.vm.execute_rvr_preflight_for_proving(
                    rvr_preflight,
                    &mut self.interpreter,
                    &self.exe,
                    from_state,
                    Some(segment.num_insns),
                    &segment.trace_heights,
                )
            };
            let PreflightExecutionOutput {
                system_records,
                record_arenas,
                to_state,
            } = if warm_this_segment {
                crate::arch::cuda::pinned::with_eager_registration(execute)?
            } else {
                execute()?
            };
            state = Some(to_state);
            if warm_this_segment {
                // Run trace generation, but deliberately stop before STARK proving. This launches
                // the non-G2 system/extension trace kernels and materializes their exact device
                // buffer shapes; the direct G2 kernel preloader alone cannot cover those paths.
                let ctx = self
                    .vm
                    .generate_proving_ctx(system_records, record_arenas)?;
                // Trace warmup exercises the same device-owned predecessor/touched-memory path
                // as a real continuation. Consume its dirty-page result before binding the next
                // warm segment; reset_state below restores the production initial state after the
                // selected prefix has finished.
                self.vm.merge_device_continuation_dirty_pages(
                    &mut state
                        .as_mut()
                        .expect("CUDA trace prewarm produced no continuation state")
                        .memory,
                );
                openvm_cuda_common::stream::device_synchronize().map_err(|error| {
                    ExecutionError::RvrExecution(format!(
                        "CUDA trace prewarm synchronization failed: {error:?}"
                    ))
                })?;
                drop(ctx);
                if !crate::arch::cuda::pinned::drain_returns(std::time::Duration::from_secs(30)) {
                    return Err(ExecutionError::RvrExecution(
                        "CUDA pinned-host prewarm timed out draining returned arenas".to_string(),
                    )
                    .into());
                }
            } else {
                drop(system_records);
                drop(record_arenas);
            }
            if seg_idx == last_selected {
                break;
            }
        }
        self.state = state;
        self.reset_state(input);
        self.vm
            .builder
            .finish_rvr_cuda_device_prewarm(device_pool_reserve_bytes)
            .map_err(ExecutionError::RvrExecution)?;
        self.rvr_cuda_paths_prewarmed = true;
        self.rvr_cuda_pool_reserve_bytes = device_pool_reserve_bytes;
        eprintln!(
            "OPENVM_RVR_CUDA_DEVICE_PREWARM phase=paths elapsed_ms={} selected={:?} \
             pinned_host_pool_warm=1 trace_warm=1 proof_warm=0",
            started.elapsed().as_millis(),
            selected,
        );
        Ok(())
    }
}

impl<E, VB> ContinuationVmProver<E::SC> for VmInstance<E, VB>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    VB: VmBuilder<E>,
    <E::PD as ProverDevice<E::PB, E::TS>>::DeviceCtx: 'static,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
        + MeteredExecutor<Val<E::SC>>
        + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
{
    /// First performs metered execution to determine segments. Then sequentially proves each
    /// segment. The proof for each segment uses the specified [ProverBackend], but the proof for
    /// the next segment does not start before the current proof finishes.
    fn prove(
        &mut self,
        input: impl Into<Streams>,
    ) -> Result<ContinuationVmProof<E::SC>, VirtualMachineError> {
        self.prove_continuations(input, |_, _| {})
    }
}

impl<E, VB> VmInstance<E, VB>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    VB: VmBuilder<E>,
    <E::PD as ProverDevice<E::PB, E::TS>>::DeviceCtx: 'static,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
        + MeteredExecutor<Val<E::SC>>
        + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
{
    /// For internal use to resize trace matrices before proving.
    ///
    /// The closure `modify_ctx(seg_idx, &mut ctx)` is called sequentially for each segment.
    pub fn prove_continuations(
        &mut self,
        input: impl Into<Streams>,
        mut modify_ctx: impl FnMut(usize, &mut ProvingContext<E::PB>),
    ) -> Result<ContinuationVmProof<E::SC>, VirtualMachineError> {
        let input = input.into();
        self.reset_state(input.clone());
        #[cfg(feature = "rvr")]
        self.warm_rvr_proving()?;
        #[cfg(all(feature = "cuda", feature = "rvr"))]
        let device_prewarm_depth = rvr_cuda_device_prewarm_depth();
        let vm = &mut self.vm;
        let metered_ctx = vm.build_metered_ctx(&self.exe);
        #[cfg(feature = "rvr")]
        let metered_interpreter = self
            .rvr_metered
            .as_ref()
            .expect("RVR metered executor must be warmed before proving");
        #[cfg(not(feature = "rvr"))]
        let metered_interpreter = vm.metered_interpreter(&self.exe)?;
        let (segments, _) = metered_interpreter.execute_metered(input.clone(), metered_ctx)?;
        let num_segments = segments.len();
        #[cfg(all(feature = "cuda", feature = "rvr"))]
        let mut max_g2_capacity_bytes = 0usize;
        #[cfg(all(feature = "cuda", feature = "rvr"))]
        if let Some(CachedRvrPreflight::Rvr(rvr)) = self.rvr_preflight.as_ref() {
            let num_airs = segments
                .first()
                .map_or(0, |segment| segment.trace_heights.len());
            let mut max_trace_heights = vec![0u32; num_airs];
            for segment in &segments {
                assert_eq!(
                    segment.trace_heights.len(),
                    num_airs,
                    "metered segments disagree on AIR count"
                );
                for (maximum, &height) in max_trace_heights.iter_mut().zip(&segment.trace_heights) {
                    *maximum = (*maximum).max(height);
                }
            }
            max_g2_capacity_bytes = segments
                .iter()
                .filter_map(|segment| {
                    rvr.g2_capacity_bytes(&segment.trace_heights, segment.num_insns)
                })
                .max()
                .unwrap_or(0);
            rvr.prepare_arena_native_backings(
                &max_trace_heights,
                (max_g2_capacity_bytes != 0).then_some(max_g2_capacity_bytes),
            );
        }
        #[cfg(all(feature = "cuda", feature = "rvr"))]
        if device_prewarm_depth != 0 && !self.rvr_cuda_paths_prewarmed {
            let selected = if let Some(CachedRvrPreflight::Rvr(rvr)) = self.rvr_preflight.as_ref() {
                let g2_airs = rvr.g2_air_indices();
                if g2_airs.is_empty() {
                    Vec::new()
                } else {
                    // The direct module preloader covers the G2 per-kind kernels. Prefix trace
                    // passes cover the decoder, the remaining VM trace kernels, and synchronously
                    // pinned arena shapes; retain the configured depth because later early
                    // segments can introduce a larger size class.
                    let selected =
                        (0..segments.len().min(device_prewarm_depth)).collect::<Vec<_>>();
                    let mut uncovered = g2_airs
                        .into_iter()
                        .filter(|&air| {
                            segments.iter().any(|segment| {
                                segment.trace_heights.get(air).copied().unwrap_or(0) != 0
                            })
                        })
                        .collect::<std::collections::BTreeSet<_>>();
                    for &index in &selected {
                        uncovered.retain(|&air| {
                            segments[index].trace_heights.get(air).copied().unwrap_or(0) == 0
                        });
                    }
                    eprintln!(
                        "OPENVM_RVR_CUDA_DEVICE_PREWARM phase=select depth={} selected={:?} uncovered_active_g2_airs={:?}",
                        device_prewarm_depth,
                        selected,
                        uncovered,
                    );
                    selected
                }
            } else {
                Vec::new()
            };
            let _ = vm;
            let device_pool_reserve_bytes =
                rvr_cuda_device_pool_prewarm_bytes(max_g2_capacity_bytes);
            self.prewarm_cuda_device_paths(
                input.clone(),
                &segments,
                &selected,
                device_pool_reserve_bytes,
            )?;
        } else if !self.rvr_cuda_paths_prewarmed {
            // Populate the async device pool without replaying workload segments when no pinned
            // host-pool prefix was requested.
            let device_pool_reserve_bytes =
                rvr_cuda_device_pool_prewarm_bytes(max_g2_capacity_bytes);
            self.prewarm_cuda_device_paths(
                input.clone(),
                &segments,
                &[],
                device_pool_reserve_bytes,
            )?;
        } else {
            let device_pool_reserve_bytes =
                rvr_cuda_device_pool_prewarm_bytes(max_g2_capacity_bytes);
            if device_pool_reserve_bytes > self.rvr_cuda_pool_reserve_bytes {
                self.vm
                    .builder
                    .finish_rvr_cuda_device_prewarm(device_pool_reserve_bytes)
                    .map_err(ExecutionError::RvrExecution)?;
                self.rvr_cuda_pool_reserve_bytes = device_pool_reserve_bytes;
            }
        }
        let vm = &mut self.vm;
        #[cfg(feature = "stark-debug")]
        if std::env::var("OPENVM_STARK_DEBUG_TRACE_ONLY").as_deref() == Ok("1") {
            eprintln!("OPENVM_STARK_DEBUG_TOTAL_SEGMENTS={num_segments}");
        }
        let mut proofs = Vec::with_capacity(num_segments);
        let mut state = self.state.take();
        for (seg_idx, segment) in segments.into_iter().enumerate() {
            let _segment_span = info_span!("prove_segment", segment = seg_idx).entered();
            // We need a separate span so the metric label includes "segment" from _segment_span
            let _prove_span = info_span!("total_proof").entered();
            let Segment {
                num_insns,
                trace_heights,
                ..
            } = segment;
            let from_state = Option::take(&mut state).unwrap();
            #[cfg(feature = "cuda")]
            let gpu_e2e_profile = std::env::var("OPENVM_GPU_E2E_PROFILE").as_deref() == Ok("1");
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            let device_pool_entry = gpu_e2e_profile
                .then(|| vm.builder.rvr_cuda_device_pool_stats().unwrap())
                .flatten();
            #[cfg(feature = "cuda")]
            if gpu_e2e_profile {
                openvm_cuda_common::stream::device_synchronize().unwrap();
            }
            #[cfg(feature = "cuda")]
            let init_h2d_started = std::time::Instant::now();
            #[cfg(feature = "cuda")]
            let init_memory_span =
                gpu_e2e_profile.then(|| info_span!("gpu_e2e_init_memory_h2d_build").entered());
            vm.transport_init_memory_to_device(&from_state.memory);
            #[cfg(feature = "cuda")]
            if gpu_e2e_profile {
                openvm_cuda_common::stream::device_synchronize().unwrap();
            }
            #[cfg(feature = "cuda")]
            drop(init_memory_span);
            #[cfg(feature = "cuda")]
            let init_memory_h2d_build_elapsed = init_h2d_started.elapsed();
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            let device_pool_after_init = gpu_e2e_profile
                .then(|| vm.builder.rvr_cuda_device_pool_stats().unwrap())
                .flatten();
            #[cfg(feature = "rvr")]
            let rvr_preflight = self
                .rvr_preflight
                .as_ref()
                .expect("RVR preflight executor must be warmed before proving");
            #[cfg(feature = "rvr")]
            if let CachedRvrPreflight::Rvr(rvr) = rvr_preflight {
                rvr.prepare_wire_backings(&trace_heights);
            }
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            let arena_pool_before = crate::arch::cuda::pinned::stats_enabled()
                .then(crate::arch::cuda::pinned::PoolStatsSnapshot::capture);
            #[cfg(any(feature = "stark-debug", feature = "cuda"))]
            let preflight_started = std::time::Instant::now();
            #[cfg(feature = "rvr")]
            let PreflightExecutionOutput {
                system_records,
                record_arenas,
                to_state,
            } = vm.execute_rvr_preflight_for_proving(
                rvr_preflight,
                &mut self.interpreter,
                &self.exe,
                from_state,
                Some(num_insns),
                &trace_heights,
            )?;
            #[cfg(not(feature = "rvr"))]
            let PreflightExecutionOutput {
                system_records,
                record_arenas,
                to_state,
            } = vm.execute_preflight_for(
                &mut self.interpreter,
                from_state,
                num_insns,
                &trace_heights,
            )?;
            #[cfg(any(feature = "stark-debug", feature = "cuda"))]
            let preflight_elapsed = preflight_started.elapsed();
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            let device_pool_after_preflight = gpu_e2e_profile
                .then(|| vm.builder.rvr_cuda_device_pool_stats().unwrap())
                .flatten();
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            if let Some(before) = arena_pool_before {
                crate::arch::cuda::pinned::emit_segment_stats(seg_idx, before);
            }
            state = Some(to_state);

            #[cfg(any(feature = "stark-debug", feature = "cuda"))]
            let tracegen_started = std::time::Instant::now();
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            let tracegen_gpu_timer =
                crate::arch::rvr::gpu_profile::CudaStageTimer::start_from_device_ctx(
                    vm.engine.device().device_ctx(),
                );
            let mut ctx = vm.generate_proving_ctx(system_records, record_arenas)?;
            vm.merge_device_continuation_dirty_pages(
                &mut state
                    .as_mut()
                    .expect("preflight produced no continuation state")
                    .memory,
            );
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            vm.builder.release_rvr_cuda_device_trace_sources();
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            if let Some(timer) = tracegen_gpu_timer {
                timer.finish(
                    "tracegen",
                    u32::try_from(seg_idx).expect("G2 tracegen segment index exceeds u32"),
                    0,
                );
            }
            #[cfg(feature = "cuda")]
            if gpu_e2e_profile {
                openvm_cuda_common::stream::device_synchronize().unwrap();
            }
            #[cfg(any(feature = "stark-debug", feature = "cuda"))]
            let tracegen_elapsed = tracegen_started.elapsed();
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            let device_pool_after_tracegen = gpu_e2e_profile
                .then(|| vm.builder.rvr_cuda_device_pool_stats().unwrap())
                .flatten();
            modify_ctx(seg_idx, &mut ctx);
            #[cfg(feature = "stark-debug")]
            if std::env::var("OPENVM_STARK_DEBUG_TRACE_ONLY").as_deref() == Ok("1") {
                eprintln!(
                    "OPENVM_STARK_DEBUG_SEGMENT_TIMING seg={seg_idx} insns={num_insns} \
                     preflight_us={} tracegen_us={}",
                    preflight_elapsed.as_micros(),
                    tracegen_elapsed.as_micros()
                );
                eprintln!("OPENVM_STARK_DEBUG_SEGMENT_BALANCED={seg_idx}");
                let stop_after = std::env::var("OPENVM_STARK_DEBUG_STOP_AFTER_SEGMENT")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                // Trace-only mode never produces proofs, so exit cleanly after the last
                // segment (or an explicit early stop) instead of falling through to the
                // proof-consuming code with an empty proof vector.
                if stop_after == Some(seg_idx) || seg_idx + 1 == num_segments {
                    eprintln!("OPENVM_STARK_DEBUG_CHECKED_SEGMENTS={}", seg_idx + 1);
                    eprintln!("OPENVM_STARK_DEBUG_TRACE_ONLY_COMPLETE=1");
                    std::process::exit(0);
                }
                continue;
            }
            #[cfg(feature = "cuda")]
            let prove_started = std::time::Instant::now();
            let proof = vm.engine.prove(vm.pk(), ctx).unwrap();
            #[cfg(feature = "cuda")]
            if gpu_e2e_profile {
                openvm_cuda_common::stream::device_synchronize().unwrap();
                let prove_elapsed = prove_started.elapsed();
                #[cfg(feature = "rvr")]
                if let (
                    Some(entry),
                    Some(after_init),
                    Some(after_preflight),
                    Some(after_tracegen),
                ) = (
                    device_pool_entry,
                    device_pool_after_init,
                    device_pool_after_preflight,
                    device_pool_after_tracegen,
                ) {
                    let after_prove = vm
                        .builder
                        .rvr_cuda_device_pool_stats()
                        .unwrap()
                        .expect("CUDA builder omitted device-pool profiling stats");
                    eprintln!(
                        "OPENVM_GPU_E2E_DEVICE_POOL seg={seg_idx} \
                         entry_reserved={} init_reserved={} preflight_reserved={} \
                         tracegen_reserved={} prove_reserved={} reserved_high={} \
                         entry_used={} init_used={} preflight_used={} tracegen_used={} \
                         prove_used={} used_high={} release_threshold={}",
                        entry[0],
                        after_init[0],
                        after_preflight[0],
                        after_tracegen[0],
                        after_prove[0],
                        after_prove[1],
                        entry[2],
                        after_init[2],
                        after_preflight[2],
                        after_tracegen[2],
                        after_prove[2],
                        after_prove[3],
                        after_prove[4],
                    );
                }
                eprintln!(
                    "OPENVM_GPU_E2E_SEGMENT_TIMING seg={seg_idx} insns={num_insns} \
                     init_memory_h2d_build_us={} preflight_us={} \
                     tracegen_including_h2d_us={} prove_us={}",
                    init_memory_h2d_build_elapsed.as_micros(),
                    preflight_elapsed.as_micros(),
                    tracegen_elapsed.as_micros(),
                    prove_elapsed.as_micros()
                );
            }
            proofs.push(proof);
        }
        let to_state = state.unwrap();
        let final_memory = &to_state.memory.memory;
        let final_memory_top_tree = vm.memory_top_tree().expect("memory top tree should exist");
        let user_public_values = UserPublicValuesProof::compute(
            vm.config().as_ref(),
            &vm_poseidon2_hasher(),
            final_memory,
            final_memory_top_tree,
        );
        self.state = Some(to_state);
        Ok(ContinuationVmProof {
            per_segment: proofs,
            user_public_values,
        })
    }
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
    pub exe_commit: [F; VM_DIGEST_WIDTH],
    /// The Merkle root of the final memory state.
    pub final_memory_root: [F; VM_DIGEST_WIDTH],
}

/// Verify segment proofs with boundary condition checks for continuation between segments.
///
/// Assumption:
/// - `vk` is a valid verifying key of a VM circuit.
///
/// Returns:
/// - The commitment to the VM executable extracted from `proofs`. It is the responsibility of the
///   caller to check that the returned commitment matches the VM executable that the VM was
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
) -> Result<VerifiedExecutionPayload<Val<E::SC>>, VmVerificationError<E::SC>>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    Com<E::SC>: Into<[Val<E::SC>; VM_DIGEST_WIDTH]>,
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
        let mut boundary_air_present = false;
        let mut merkle_air_present = false;

        // Check public values.
        for (air_idx, (vdata, pvs)) in proof
            .trace_vdata
            .iter()
            .zip(proof.public_values.iter())
            .enumerate()
        {
            let air_vk = &vk.inner.per_air[air_idx];
            if air_idx == PROGRAM_AIR_ID {
                program_air_present = true;
                let vdata = vdata.as_ref().unwrap();
                if i == 0 {
                    program_commit = Some(vdata.cached_commitments[PROGRAM_CACHED_TRACE_INDEX]);
                } else if program_commit.unwrap()
                    != vdata.cached_commitments[PROGRAM_CACHED_TRACE_INDEX]
                {
                    return Err(VmVerificationError::ProgramCommitMismatch { index: i });
                }
            } else if air_idx == CONNECTOR_AIR_ID {
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
                if pvs.is_terminate != PrimeCharacteristicRing::from_bool(expected_is_terminate) {
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
                if pvs.exit_code != PrimeCharacteristicRing::from_u32(expected_exit_code) {
                    return Err(VmVerificationError::ExitCodeMismatch {
                        expected: expected_exit_code,
                        actual: pvs.exit_code.as_canonical_u32(),
                    });
                }
            } else if air_idx == BOUNDARY_AIR_ID {
                boundary_air_present = vdata.is_some();
                if !pvs.is_empty() {
                    return Err(VmVerificationError::UnexpectedPvs {
                        expected: 0,
                        actual: pvs.len(),
                    });
                }
            } else if air_idx == MERKLE_AIR_ID {
                merkle_air_present = true;
                let pvs: &MemoryMerklePvs<_, VM_DIGEST_WIDTH> = pvs.as_slice().borrow();

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
        if !boundary_air_present {
            return Err(VmVerificationError::SystemAirMissing {
                air_id: BOUNDARY_AIR_ID,
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
        &program_commit.unwrap().into(),
        initial_memory_root.as_ref().unwrap(),
        start_pc.unwrap(),
    );
    Ok(VerifiedExecutionPayload {
        exe_commit,
        final_memory_root: prev_final_memory_root.unwrap(),
    })
}

impl<SC: StarkProtocolConfig> Clone for ContinuationVmProof<SC>
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
pub fn debug_proving_ctx<E, VB>(vm: &VirtualMachine<E, VB>, ctx: &ProvingContext<E::PB>)
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    let air_inv = vm.config().create_airs().unwrap();
    let global_airs: Vec<AirRef<E::SC>> = air_inv.into_airs().map(|a| a as AirRef<_>).collect();
    vm.engine.debug(&global_airs, ctx);
}
