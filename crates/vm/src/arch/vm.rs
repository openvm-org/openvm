//! [VmExecutor] is the struct that can execute an _arbitrary_ program, provided in the form of a
//! [VmExe](openvm_instructions::exe::VmExe), for a fixed set of OpenVM instructions
//! corresponding to a [VmExecutionConfig].
//! Internally once it is given a program, it will preprocess the program to rewrite it into a more
//! optimized format for runtime execution. This **instance** of the executor will be a separate
//! struct specialized to running a _fixed_ program on different program inputs.
//!
//! [VirtualMachine] will similarly be the struct that has done all the setup so it can
//! execute+prove an arbitrary program for a fixed config - it will internally still hold VmExecutor
#[cfg(feature = "cuda")]
use std::any::Any;
#[cfg(feature = "metrics")]
use std::collections::BTreeMap;
#[cfg(feature = "rvr")]
use std::path::Path;
use std::{any::TypeId, borrow::Borrow, collections::VecDeque, sync::Arc};

use getset::{Getters, MutGetters, Setters, WithSetters};
use itertools::Itertools;
use openvm_circuit::system::program::trace::compute_exe_commit;
use openvm_cpu_backend::CpuBackend;
#[cfg(all(feature = "cuda", feature = "metrics"))]
use openvm_cuda_backend::prelude::F as CudaField;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend};
#[cfg(feature = "cuda")]
use openvm_cuda_common::memory_manager::MemTracker;
#[cfg(all(feature = "cuda", feature = "metrics"))]
use openvm_instructions::VmOpcode;
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    program::Program,
    VM_DIGEST_WIDTH,
};
#[cfg(feature = "metrics")]
use openvm_instructions::{LocalOpcode, SystemOpcode};
#[cfg(feature = "perf-metrics")]
use openvm_instructions::{PhantomDiscriminant, SysPhantom};
#[cfg(feature = "cuda")]
use openvm_stark_backend::prover::AirProvingContext;
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
use rvr_openvm_lift::RvrExtensions;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info_span, instrument};

#[cfg(feature = "cuda")]
use super::cuda::postflight::{
    GpuPostflightBoundary, GpuPostflightContext, GpuPostflightError, GpuPostflightPlan,
    GpuPostflightProgram, GpuPostflightTranscript,
};
#[cfg(any(not(feature = "rvr"), feature = "test-utils"))]
use super::execution_mode::PreflightCtx;
#[cfg(feature = "rvr")]
use super::rvr::{
    bridge::map_rvr_compile_error, build_pc_to_chip, compile, compile::compile_preflight,
    compile_metered, compile_metered_cost, compile_metered_segment_boundary,
    compile_with_instret_tracking, load_compiled_from_path, ChipMapping, GuestDebugMap,
    PreflightInstance, RvrExecutionKind, RvrInitialImage, RvrMeteredCostInstance,
    RvrMeteredInstance, RvrMeteredSegmentInstance, RvrPureInstance,
    RvrPureWithInstretTrackingInstance,
};
#[cfg(feature = "cuda")]
use super::ExecutionState;
#[cfg(feature = "metrics")]
use super::InterpreterExecutor;
#[cfg(feature = "perf-metrics")]
use super::PreflightProgramEvent;
use super::{
    execution_mode::{
        ExecutionCtx, MeteredCostCtx, MeteredCtx, MeteredCtxInputs, Segment, SegmentationLimits,
    },
    hasher::poseidon2::vm_poseidon2_hasher,
    hint_stream::HintStream,
    interpreter::{InterpretedInstance, PreflightInterpretedInstance},
    AirInventoryError, ChipInventoryError, ExecutionError, Executor, ExecutorInventory,
    ExecutorInventoryError, MemoryConfig, MeteredExecutor, Postflight, PostflightProgramIndex,
    PreflightOutput, StaticProgramError, SystemConfig, VmBuilder, VmChipComplex, VmCircuitConfig,
    VmExecutionConfig, VmState, BOUNDARY_AIR_ID, CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID,
    PROGRAM_CACHED_TRACE_INDEX,
};
#[cfg(feature = "cuda")]
use crate::system::cuda::SystemChipInventoryGPU;
use crate::{
    arch::deferral::DeferralState,
    system::{
        connector::{VmConnectorPvs, DEFAULT_SUSPEND_EXIT_CODE},
        memory::{
            merkle::{
                public_values::{UserPublicValuesProof, UserPublicValuesProofError},
                MemoryMerklePvs,
            },
            online::GuestMemory,
            AddressMap,
        },
        program::trace::generate_cached_trace,
        SystemChipComplex, SystemChipInventory, SystemWithFixedTraceHeights,
    },
};

#[cfg(all(feature = "rvr", feature = "test-utils"))]
mod testing;

/// Canonical field bound for VM execution/circuit code.
pub const BABYBEAR_S_BOX_DEGREE: u64 = 7;

pub trait VmField: PrimeField32 + InjectiveMonomial<BABYBEAR_S_BOX_DEGREE> {}
impl<T> VmField for T where T: PrimeField32 + InjectiveMonomial<BABYBEAR_S_BOX_DEGREE> {}

#[cfg(feature = "cuda")]
fn with_gpu_memory_metrics<T, E>(
    name: &'static str,
    f: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    let memory = MemTracker::start_and_reset_peak(name);
    let result = f();
    memory.emit_metrics();
    result
}

#[derive(Error, Debug)]
pub enum GenerationError {
    #[error("extension trace generation failed: {0}")]
    ExtensionTracegen(String),
    #[error("VM prover cannot be reused after an incomplete or failed proving session")]
    ProverPoisoned,
    #[error("proof generation failed: {0}")]
    Proving(String),
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

/// Converts immutable preflight history into a backend-specific proving context.
///
/// Implementations may prepare backend-owned fixed-program data once and reuse it across all
/// segments. CPU builders replay the history through their chip inventory, while GPU builders
/// expand it into the transcript consumed by their trace-generation kernels. The concrete
/// coordinator owns the complete trace-generation session, including final validation: a failed
/// or incomplete session must leave the VM poisoned.
pub trait PostflightTracegen<E: StarkEngine>: VmBuilder<E> {
    type Prepared;

    /// Prepares fixed-program data. CPU preparation indexes `program`; GPU preparation uses `vm`
    /// to upload it.
    fn prepare_postflight(
        vm: &VirtualMachine<E, Self>,
        program: &Program,
    ) -> Result<Self::Prepared, GenerationError>;

    /// Builds one segment's proving context. CPU trace generation reads instructions from
    /// `host_program`; GPU implementations receive it through this shared trait but read the
    /// uploaded `prepared` program instead.
    fn generate_proving_ctx(
        vm: &mut VirtualMachine<E, Self>,
        host_program: &Program,
        prepared: &Self::Prepared,
        output: &PreflightOutput,
    ) -> Result<ProvingContext<E::PB>, GenerationError>;
}

impl<SC, E, VB> PostflightTracegen<E> for VB
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>>,
    Val<SC>: VmField,
    VB: VmBuilder<E, SystemChipInventory = SystemChipInventory<SC>>,
    <VB::VmConfig as VmExecutionConfig<Val<SC>>>::Executor: Executor<Val<SC>>,
{
    type Prepared = PostflightProgramIndex;

    fn prepare_postflight(
        _vm: &VirtualMachine<E, Self>,
        program: &Program,
    ) -> Result<Self::Prepared, GenerationError> {
        PostflightProgramIndex::new(program)
            .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))
    }

    fn generate_proving_ctx(
        vm: &mut VirtualMachine<E, Self>,
        host_program: &Program,
        prepared: &Self::Prepared,
        output: &PreflightOutput,
    ) -> Result<ProvingContext<E::PB>, GenerationError> {
        begin_preflight_tracegen_session(&mut vm.preflight_tracegen_poisoned)?;
        let postflight = Postflight::new_prepared(
            host_program,
            prepared,
            &output.history,
            &vm.config().as_ref().memory_config,
            output.exit_code,
        )
        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        #[cfg(feature = "metrics")]
        crate::metrics::emit_opcode_counts(
            &output.state.metrics,
            vm.postflight_opcode_counts(&postflight),
        );
        let result = {
            let _span = info_span!("trace_gen").entered();
            vm.chip_complex
                .generate_proving_ctx_from_postflight(&postflight)
                .and_then(|ctx| vm.validate_proving_ctx(ctx))
        };
        if result.is_ok() {
            vm.preflight_tracegen_poisoned = false;
        }
        result
    }
}

#[cfg(feature = "cuda")]
pub fn prepare_gpu_postflight<VB>(
    vm: &VirtualMachine<BabyBearPoseidon2GpuEngine, VB>,
    program: &Program,
) -> Result<GpuPostflightProgram, GenerationError>
where
    VB: VmBuilder<BabyBearPoseidon2GpuEngine>,
{
    GpuPostflightProgram::upload(
        program,
        &vm.config().as_ref().memory_config,
        &vm.engine.device().device_ctx,
    )
    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))
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

/// Preflight interpreter specialized by a VM execution config.
type PreflightInterpreter<F, VC> =
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
    /// Creates an interpreter instance specialized for append-only preflight.
    #[cfg(not(feature = "rvr"))]
    pub fn preflight_instance(
        &self,
        exe: &VmExe,
    ) -> Result<InterpretedInstance<'_, PreflightCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_preflight", backend = "interpreter").entered();
        InterpretedInstance::new::<F, _>(&self.inventory, exe)
    }

    /// Creates an instance of the interpreter specialized for pure execution, without metering, of
    /// the given `exe`.
    ///
    /// For metered execution, use the [`metered_instance`](Self::metered_instance) constructor.
    #[cfg(not(feature = "rvr"))]
    pub fn instance(
        &self,
        exe: &VmExe,
    ) -> Result<InterpretedInstance<'_, ExecutionCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_pure", backend = "interpreter").entered();
        InterpretedInstance::new::<F, _>(&self.inventory, exe)
    }

    #[cfg(feature = "rvr")]
    pub fn interpreter_instance(
        &self,
        exe: &VmExe,
    ) -> Result<InterpretedInstance<'_, ExecutionCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_pure", backend = "interpreter").entered();
        InterpretedInstance::new::<F, _>(&self.inventory, exe)
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
    pub fn instance(&self, exe: &VmExe) -> Result<RvrPureInstance<'_>, StaticProgramError> {
        self.instance_with_debug_map(exe, None)
    }

    pub fn instance_with_debug_map(
        &self,
        exe: &VmExe,
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<RvrPureInstance<'_>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span = tracing::info_span!("compile_pure", backend = "compiled").entered();
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
    pub fn instret_tracking_instance(
        &self,
        exe: &VmExe,
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<RvrPureWithInstretTrackingInstance<'_>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span = tracing::info_span!("compile_pure", backend = "compiled").entered();
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

    /// Compile the compact preflight executor.
    ///
    /// The compact transcript is the serial input to record-free GPU replay.
    pub fn preflight_instance(
        &self,
        exe: &VmExe,
    ) -> Result<PreflightInstance<'_>, StaticProgramError> {
        self.preflight_instance_with_debug_map(exe, None)
    }

    pub fn preflight_instance_with_debug_map(
        &self,
        exe: &VmExe,
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<PreflightInstance<'_>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_preflight", backend = "compiled").entered();
        let extensions = self.build_rvr_extensions(None);
        let compiled = compile_preflight(exe, extensions.lifters(), guest_debug_map)
            .map_err(map_rvr_compile_error)?;
        Ok(PreflightInstance::new(
            self.inventory.config(),
            RvrInitialImage::from(exe),
            compiled,
            extensions.into_runtime_hooks(),
        ))
    }

    /// Load a previously saved preflight artifact.
    pub fn load_preflight_instance(
        &self,
        lib_path: &Path,
        exe: &VmExe,
    ) -> Result<PreflightInstance<'_>, StaticProgramError> {
        let extensions = self.build_rvr_extensions(None);
        let compiled = load_compiled_from_path(lib_path).map_err(map_rvr_compile_error)?;
        compiled
            .require_execution_kind(&[RvrExecutionKind::Preflight])
            .map_err(map_rvr_compile_error)?;
        Ok(PreflightInstance::new(
            self.inventory.config(),
            RvrInitialImage::from(exe),
            compiled,
            extensions.into_runtime_hooks(),
        ))
    }

    /// Load a previously saved unlimited-pure artifact.
    pub fn load_instance(
        &self,
        lib_path: &Path,
        exe: &VmExe,
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
        lib_path: &Path,
        exe: &VmExe,
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
        exe: &VmExe,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<InterpretedInstance<'_, MeteredCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered", backend = "interpreter").entered();
        InterpretedInstance::new_metered::<F, _>(&self.inventory, exe, executor_idx_to_air_idx)
    }

    #[cfg(feature = "rvr")]
    pub fn metered_interpreter_instance(
        &self,
        exe: &VmExe,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<InterpretedInstance<'_, MeteredCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered", backend = "interpreter").entered();
        InterpretedInstance::new_metered::<F, _>(&self.inventory, exe, executor_idx_to_air_idx)
    }

    /// Creates an instance of the interpreter specialized for cost metering execution of the given
    /// `exe`.
    #[cfg(not(feature = "rvr"))]
    pub fn metered_cost_instance(
        &self,
        exe: &VmExe,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<InterpretedInstance<'_, MeteredCostCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered_cost", backend = "interpreter").entered();
        InterpretedInstance::new_metered::<F, _>(&self.inventory, exe, executor_idx_to_air_idx)
    }

    #[cfg(feature = "rvr")]
    pub fn metered_cost_interpreter_instance(
        &self,
        exe: &VmExe,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<InterpretedInstance<'_, MeteredCostCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered_cost", backend = "interpreter").entered();
        InterpretedInstance::new_metered::<F, _>(&self.inventory, exe, executor_idx_to_air_idx)
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
        exe: &VmExe,
        executor_idx_to_air_idx: &[usize],
        num_airs: usize,
    ) -> Result<RvrMeteredInstance<'_>, StaticProgramError> {
        self.metered_instance_with_debug_map(exe, executor_idx_to_air_idx, num_airs, None)
    }

    pub fn metered_instance_with_debug_map(
        &self,
        exe: &VmExe,
        executor_idx_to_air_idx: &[usize],
        num_airs: usize,
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<RvrMeteredInstance<'_>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered", backend = "compiled").entered();
        let extensions = self.build_rvr_extensions(Some(executor_idx_to_air_idx));
        let chips = ChipMapping {
            num_airs,
            pc_to_chip: build_pc_to_chip(exe, &self.inventory, executor_idx_to_air_idx)
                .map_err(map_rvr_compile_error)?,
            chip_widths: None,
        };
        let uses_deferral_address_space = extensions.lifters().uses_deferral_address_space();
        let compiled = compile_metered(exe, extensions.lifters(), &chips, guest_debug_map)
            .map_err(map_rvr_compile_error)?;
        let runtime_hooks = extensions.into_runtime_hooks();

        Ok(RvrMeteredInstance::new(
            self.inventory.config(),
            RvrInitialImage::from(exe),
            runtime_hooks,
            compiled,
            uses_deferral_address_space,
        ))
    }

    pub fn metered_segment_instance(
        &self,
        exe: &VmExe,
        executor_idx_to_air_idx: &[usize],
        num_airs: usize,
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<RvrMeteredSegmentInstance<'_>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered_segment", backend = "compiled").entered();
        let extensions = self.build_rvr_extensions(Some(executor_idx_to_air_idx));
        let chips = ChipMapping {
            num_airs,
            pc_to_chip: build_pc_to_chip(exe, &self.inventory, executor_idx_to_air_idx)
                .map_err(map_rvr_compile_error)?,
            chip_widths: None,
        };
        let uses_deferral_address_space = extensions.lifters().uses_deferral_address_space();
        let compiled =
            compile_metered_segment_boundary(exe, extensions.lifters(), &chips, guest_debug_map)
                .map_err(map_rvr_compile_error)?;
        let runtime_hooks = extensions.into_runtime_hooks();

        Ok(RvrMeteredSegmentInstance::new(
            self.inventory.config(),
            RvrInitialImage::from(exe),
            runtime_hooks,
            compiled,
            uses_deferral_address_space,
        ))
    }

    pub fn metered_cost_instance(
        &self,
        exe: &VmExe,
        executor_idx_to_air_idx: &[usize],
        widths: &[usize],
    ) -> Result<RvrMeteredCostInstance<'_>, StaticProgramError> {
        self.metered_cost_instance_with_debug_map(exe, executor_idx_to_air_idx, widths, None)
    }

    /// Load a previously saved metered-mode artifact. Its generated execution
    /// kind is validated; the caller supplies matching `exe` and
    /// `executor_idx_to_air_idx`.
    pub fn load_metered_instance(
        &self,
        lib_path: &Path,
        exe: &VmExe,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<RvrMeteredInstance<'_>, StaticProgramError> {
        let extensions = self.build_rvr_extensions(Some(executor_idx_to_air_idx));
        let uses_deferral_address_space = extensions.lifters().uses_deferral_address_space();
        let runtime_hooks = extensions.into_runtime_hooks();
        let compiled = load_compiled_from_path(lib_path).map_err(map_rvr_compile_error)?;
        compiled
            .require_execution_kind(&[RvrExecutionKind::Metered])
            .map_err(map_rvr_compile_error)?;

        Ok(RvrMeteredInstance::new(
            self.inventory.config(),
            RvrInitialImage::from(exe),
            runtime_hooks,
            compiled,
            uses_deferral_address_space,
        ))
    }

    /// Load a previously saved segment-boundary metered artifact. Its generated
    /// execution kind is validated; the caller supplies matching `exe` and
    /// `executor_idx_to_air_idx`.
    pub fn load_metered_segment_instance(
        &self,
        lib_path: &Path,
        exe: &VmExe,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<RvrMeteredSegmentInstance<'_>, StaticProgramError> {
        let extensions = self.build_rvr_extensions(Some(executor_idx_to_air_idx));
        let uses_deferral_address_space = extensions.lifters().uses_deferral_address_space();
        let runtime_hooks = extensions.into_runtime_hooks();
        let compiled = load_compiled_from_path(lib_path).map_err(map_rvr_compile_error)?;
        compiled
            .require_execution_kind(&[RvrExecutionKind::MeteredSegment])
            .map_err(map_rvr_compile_error)?;

        Ok(RvrMeteredSegmentInstance::new(
            self.inventory.config(),
            RvrInitialImage::from(exe),
            runtime_hooks,
            compiled,
            uses_deferral_address_space,
        ))
    }

    /// Load a saved metered-cost artifact and check its execution kind and chip
    /// widths. The caller must provide matching `exe`,
    /// `executor_idx_to_air_idx`, and `widths`.
    pub fn load_metered_cost_instance(
        &self,
        lib_path: &Path,
        exe: &VmExe,
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

    pub fn metered_cost_instance_with_debug_map(
        &self,
        exe: &VmExe,
        executor_idx_to_air_idx: &[usize],
        widths: &[usize],
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<RvrMeteredCostInstance<'_>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered_cost", backend = "compiled").entered();
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

    #[error("initial pc index mismatch (initial: {initial}, prev_final: {prev_final})")]
    InitialPcIdxMismatch { initial: u32, prev_final: u32 },

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

fn begin_preflight_tracegen_session(poisoned: &mut bool) -> Result<(), GenerationError> {
    if *poisoned {
        return Err(GenerationError::ProverPoisoned);
    }
    *poisoned = true;
    Ok(())
}

/// The [VirtualMachine] struct contains the API to generate proofs for _arbitrary_ programs for a
/// fixed set of OpenVM instructions and a fixed VM circuit corresponding to those instructions. The
/// API is specific to a particular [StarkEngine], which specifies a fixed [StarkProtocolConfig] and
/// [ProverBackend] via associated types.
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
    chip_complex: VmChipComplex<E::SC, E::PB, VB::SystemChipInventory>,
    /// Preflight trace generation mutates shared lookup histograms. Once a segment
    /// starts, the VM remains poisoned until its outermost coordinator has
    /// validated every producer and explicitly completes the session.
    preflight_tracegen_poisoned: bool,
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
            preflight_tracegen_poisoned: false,
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

    #[cfg(not(feature = "rvr"))]
    pub fn preflight_instance(
        &self,
        exe: &VmExe,
    ) -> Result<InterpretedInstance<'_, PreflightCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        self.executor().preflight_instance(exe)
    }

    #[cfg(feature = "rvr")]
    pub fn preflight_instance(
        &self,
        exe: &VmExe,
    ) -> Result<PreflightInstance<'_>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        self.executor().preflight_instance(exe)
    }

    /// Pure execution instance.
    #[cfg(not(feature = "rvr"))]
    pub fn instance(
        &self,
        exe: &VmExe,
    ) -> Result<InterpretedInstance<'_, ExecutionCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        self.executor().instance(exe)
    }

    #[cfg(feature = "rvr")]
    pub fn instance(&self, exe: &VmExe) -> Result<RvrPureInstance<'_>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        self.executor().instance(exe)
    }

    // Interpreter access when RVR is enabled.
    #[cfg(feature = "rvr")]
    pub fn interpreter_instance(
        &self,
        exe: &VmExe,
    ) -> Result<InterpretedInstance<'_, ExecutionCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        self.executor().interpreter_instance(exe)
    }

    #[cfg(not(feature = "rvr"))]
    pub fn metered_instance(
        &self,
        exe: &VmExe,
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
    pub fn metered_instance(
        &self,
        exe: &VmExe,
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
    pub fn metered_segment_instance(
        &self,
        exe: &VmExe,
    ) -> Result<RvrMeteredSegmentInstance<'_>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor().metered_segment_instance(
            exe,
            &executor_idx_to_air_idx,
            self.num_airs(),
            None,
        )
    }

    #[cfg(feature = "rvr")]
    pub fn load_metered_instance(
        &self,
        lib_path: &Path,
        exe: &VmExe,
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
        lib_path: &Path,
        exe: &VmExe,
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
    pub fn metered_interpreter_instance(
        &self,
        exe: &VmExe,
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
    pub fn metered_cost_instance(
        &self,
        exe: &VmExe,
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
    pub fn metered_cost_instance(
        &self,
        exe: &VmExe,
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
        self.executor().metered_cost_instance_with_debug_map(
            exe,
            &executor_idx_to_air_idx,
            &widths,
            None,
        )
    }

    #[cfg(feature = "rvr")]
    pub fn load_metered_cost_instance(
        &self,
        lib_path: &Path,
        exe: &VmExe,
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

    /// Builds the interpreter preflight instance for `exe`.
    pub fn preflight_interpreter(
        &self,
        exe: &VmExe,
    ) -> Result<PreflightInterpreter<Val<E::SC>, VB::VmConfig>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>> + 'static,
    {
        PreflightInterpretedInstance::new(exe, self.executor.inventory.clone())
    }

    fn prove_segment_inner(
        &mut self,
        interpreter: &PreflightInterpreter<Val<E::SC>, VB::VmConfig>,
        program: &Program,
        prepared: &VB::Prepared,
        state: VmState<GuestMemory>,
        segment: &Segment,
    ) -> Result<(Proof<E::SC>, PreflightOutput), VirtualMachineError>
    where
        Val<E::SC>: VmField,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>> + 'static,
        VB: PostflightTracegen<E>,
    {
        self.transport_init_memory_to_device(&state.memory);
        let output = interpreter.execute_segment(state, segment)?;
        #[cfg(feature = "perf-metrics")]
        let mut output = output;
        let ctx = self.generate_proving_ctx(program, prepared, &output)?;
        #[cfg(feature = "perf-metrics")]
        info_span!("guest_profile").in_scope(|| {
            self.emit_guest_instruction_metrics(
                program,
                &output.history.program,
                &mut output.state.metrics,
            )
        })?;
        let proof = self
            .engine
            .prove(self.pk(), ctx)
            .map_err(|error| GenerationError::Proving(error.to_string()))?;
        Ok((proof, output))
    }

    /// Calls [`VmState::initial`] but sets more information for
    /// performance metrics when feature "perf-metrics" is enabled.
    #[instrument(name = "vm.create_initial_state", level = "debug", skip_all)]
    pub fn create_initial_state(
        &self,
        exe: &VmExe,
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

    pub(crate) fn generate_proving_ctx(
        &mut self,
        program: &Program,
        prepared: &VB::Prepared,
        output: &PreflightOutput,
    ) -> Result<ProvingContext<E::PB>, GenerationError>
    where
        VB: PostflightTracegen<E>,
    {
        VB::generate_proving_ctx(self, program, prepared, output)
    }

    fn validate_proving_ctx(
        &self,
        ctx: ProvingContext<E::PB>,
    ) -> Result<ProvingContext<E::PB>, GenerationError> {
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
        #[cfg(feature = "stark-debug")]
        self.debug_proving_ctx(&ctx);

        Ok(ctx)
    }

    /// Transforms the program into a cached trace and commits it _on device_ using the proof system
    /// polynomial commitment scheme.
    ///
    /// Returns the cached program trace.
    /// Note that [`load_program`](Self::load_program) must be called separately to load the cached
    /// program trace into the VM itself.
    pub fn commit_program_on_device(&self, program: &Program) -> CommittedTraceData<E::PB>
    where
        Val<E::SC>: PrimeField32,
    {
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

    #[cfg(feature = "metrics")]
    fn postflight_opcode_counts(
        &self,
        postflight: &Postflight<'_, Val<E::SC>>,
    ) -> BTreeMap<(usize, String), u64>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        postflight
            .executed_opcodes()
            .filter(|&opcode| opcode != SystemOpcode::TERMINATE.global_opcode())
            .filter_map(|opcode| {
                let executor_idx = *self.executor.inventory.instruction_lookup.get(&opcode)?;
                let air_idx = *executor_idx_to_air_idx.get(executor_idx as usize)?;
                let executor = self
                    .executor
                    .inventory
                    .executors
                    .get(executor_idx as usize)?;
                Some((
                    (air_idx, executor.get_opcode_name(opcode.as_usize())),
                    postflight.opcode_count(opcode),
                ))
            })
            .collect()
    }

    #[cfg(feature = "perf-metrics")]
    fn emit_guest_instruction_metrics(
        &self,
        program: &Program,
        program_log: &[PreflightProgramEvent],
        metrics: &mut crate::metrics::VmMetrics,
    ) -> Result<(), GenerationError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        for pair in program_log.windows(2) {
            let [current, next] = pair else {
                unreachable!("windows(2) always returns two elements")
            };
            let Some(program_offset) = current.pc.checked_sub(program.pc_base) else {
                return Err(GenerationError::ExtensionTracegen(format!(
                    "guest metric PC {:#x} precedes program base {:#x}",
                    current.pc, program.pc_base
                )));
            };
            if !program_offset.is_multiple_of(openvm_instructions::program::DEFAULT_PC_STEP) {
                return Err(GenerationError::ExtensionTracegen(format!(
                    "guest metric PC {:#x} is not instruction-aligned",
                    current.pc
                )));
            }
            let program_index =
                program_offset as usize / openvm_instructions::program::DEFAULT_PC_STEP as usize;
            let Some((instruction, _)) = program.get_instruction_and_debug_info(program_index)
            else {
                return Err(GenerationError::ExtensionTracegen(format!(
                    "guest metric PC {:#x} does not resolve to an instruction",
                    current.pc
                )));
            };
            if instruction.opcode == SystemOpcode::TERMINATE.global_opcode() {
                continue;
            }

            let executor_idx = *self
                .executor
                .inventory
                .instruction_lookup
                .get(&instruction.opcode)
                .ok_or_else(|| {
                    GenerationError::ExtensionTracegen(format!(
                        "guest metric opcode {} has no executor",
                        instruction.opcode.as_usize()
                    ))
                })?;
            let executor = self
                .executor
                .inventory
                .executors
                .get(executor_idx as usize)
                .ok_or_else(|| {
                    GenerationError::ExtensionTracegen(format!(
                        "guest metric executor index {executor_idx} is out of bounds"
                    ))
                })?;
            let debug_info = metrics.debug_infos.get(current.pc);

            let system_phantom = if instruction.opcode == SystemOpcode::PHANTOM.global_opcode() {
                let phantom = PhantomDiscriminant(instruction.c.as_u32() as u16);
                SysPhantom::from_repr(phantom.0)
            } else {
                None
            };

            metrics.record_replayed_instruction(
                executor.get_opcode_name(instruction.opcode.as_usize()),
                debug_info.as_ref().map(|info| info.dsl_instruction.clone()),
                system_phantom,
                next.pc,
            );
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", feature = "perf-metrics"))]
    pub fn emit_gpu_guest_instruction_metrics(
        &self,
        program: &Program,
        transcript: &GpuPostflightTranscript,
        metrics: &mut crate::metrics::VmMetrics,
    ) -> Result<(), GenerationError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        let _span = info_span!("guest_profile").entered();
        let program_log = transcript
            .copy_program_log()
            .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        self.emit_guest_instruction_metrics(program, &program_log, metrics)
    }

    /// Convenience method to construct a [MeteredCtx] using data from the stored proving key.
    pub fn build_metered_ctx(&self, exe: &VmExe) -> MeteredCtx
    where
        Val<E::SC>: PrimeField32,
    {
        let program_len = exe.program.num_defined_instructions();

        let (
            mut constant_trace_heights,
            air_names,
            widths,
            interactions,
            need_rot,
            constraint_eval_buffers,
        ): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) = self
            .pk
            .per_air
            .iter()
            .map(|pk| {
                let constant_trace_height = pk.preprocessed_data.as_ref().map(|cd| cd.height());
                let air_names = pk.air_name.clone();
                let width = pk.vk.params.width.total_width();
                let num_interactions = pk.vk.symbolic_constraints.interactions.len();
                let need_rot = pk.vk.params.need_rot;
                let constraint_eval_buffer = E::PB::constraint_eval_buffer_size(pk);
                (
                    constant_trace_height,
                    air_names,
                    width,
                    num_interactions,
                    need_rot,
                    constraint_eval_buffer,
                )
            })
            .multiunzip();

        #[cfg(feature = "metrics")]
        let bus_names = self
            .chip_complex
            .inventory
            .airs()
            .bus_names()
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>();
        #[cfg(feature = "metrics")]
        let bus_interactions = self
            .pk
            .per_air
            .iter()
            .map(|pk| {
                let mut by_bus = BTreeMap::new();
                for interaction in &pk.vk.symbolic_constraints.interactions {
                    *by_bus.entry(interaction.bus_index).or_insert(0) += 1;
                }
                by_bus.into_iter().collect()
            })
            .collect::<Vec<_>>();

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
                #[cfg(feature = "metrics")]
                bus_names: &bus_names,
                #[cfg(feature = "metrics")]
                bus_interactions: &bus_interactions,
                widths: &widths,
                interactions: &interactions,
                need_rot: &need_rot,
                constraint_eval_buffers: &constraint_eval_buffers,
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

#[cfg(feature = "cuda")]
impl<VB> VirtualMachine<BabyBearPoseidon2GpuEngine, VB>
where
    VB: VmBuilder<BabyBearPoseidon2GpuEngine, SystemChipInventory = SystemChipInventoryGPU>,
{
    /// Validates and borrows the fixed GPU program and segment-start memory for postflight.
    pub fn gpu_postflight_context<'a>(
        &'a self,
        program: &'a GpuPostflightProgram,
    ) -> Result<GpuPostflightContext<'a>, GpuPostflightError> {
        let system = &self.chip_complex.system;
        GpuPostflightContext::new(
            program,
            &system.program.device_ctx,
            &system.memory_inventory.device_ctx,
            &system.memory_inventory.initial_memory,
        )
    }

    /// Derives the standard GPU replay indexes from history produced by
    /// interpreter preflight.
    pub fn postflight_history(
        &self,
        program: &GpuPostflightProgram,
        output: &PreflightOutput,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let context = self.gpu_postflight_context(program)?;
        let from = output.history.program.first().ok_or_else(|| {
            GpuPostflightError::InvalidTranscript(
                "preflight history must contain a program event".to_string(),
            )
        })?;
        let to = output.history.program.last().unwrap();
        context.upload_history(
            &output.history,
            GpuPostflightBoundary::new(
                ExecutionState::new(from.pc, from.timestamp),
                ExecutionState::new(to.pc, to.timestamp),
                output.exit_code,
            ),
        )
    }

    #[cfg(feature = "metrics")]
    #[doc(hidden)]
    pub fn emit_preflight_opcode_counts(&self, replay_plan: &GpuPostflightPlan)
    where
        <VB::VmConfig as VmExecutionConfig<CudaField>>::Executor: Executor<CudaField>,
    {
        let executor_idx_to_air_idx = self.chip_complex.inventory.executor_idx_to_air_idx();
        for opcode in replay_plan.executed_opcodes() {
            let opcode = VmOpcode::from_usize(opcode as usize);
            if opcode == SystemOpcode::TERMINATE.global_opcode() {
                continue;
            }
            let Some(&executor_idx) = self.executor.inventory.instruction_lookup.get(&opcode)
            else {
                continue;
            };
            let Some(&air_idx) = executor_idx_to_air_idx.get(executor_idx as usize) else {
                continue;
            };
            let Some(executor) = self.executor.inventory.executors.get(executor_idx as usize)
            else {
                continue;
            };
            let Some(air_name) = self.pk.per_air.get(air_idx).map(|pk| pk.air_name.clone()) else {
                continue;
            };
            let labels = [
                ("air_name", air_name),
                ("air_id", air_idx.to_string()),
                ("opcode", executor.get_opcode_name(opcode.as_usize())),
            ];
            metrics::counter!("opcode_count", &labels)
                .absolute(replay_plan.opcode_range(opcode).len() as u64);
        }
    }

    /// Generates one preflight proving context and leaves the VM reusable only
    /// after the concrete producer has validated complete opcode coverage.
    #[doc(hidden)]
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_preflight_proving_ctx<P>(
        &mut self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
        producer: P,
        mut generate_extension: impl FnMut(
            &mut P,
            &dyn Any,
        )
            -> Result<AirProvingContext<GpuBackend>, GenerationError>,
        finish: impl FnOnce(P) -> Result<(), GenerationError>,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError> {
        begin_preflight_tracegen_session(&mut self.preflight_tracegen_poisoned)?;
        // Reset once around the complete segment tracegen phase. Nested CUDA
        // components may report their own deltas, but must not reset this
        // phase-wide high-water mark. The allocator's logical peak is the
        // source of truth for live buffers; reserved pool pages are reported
        // separately and may remain mapped after a correct drop.
        let mut producer = producer;
        let result = with_gpu_memory_metrics("tracegen", || {
            let ctx = self.chip_complex.generate_proving_ctx_from_postflight(
                program,
                transcript,
                replay_plan,
                |chip| generate_extension(&mut producer, chip),
            );

            // Every system and extension kernel above uses raw views borrowed from
            // `transcript` and `replay_plan`. The error read fences their common
            // stream, with an explicit synchronization fallback if that copy fails.
            let replay_error = transcript.finish_replay();
            if replay_error.is_ok() {
                // The boundary trace kernel has completed on this stream. Its
                // merged input records are not part of the proving context;
                // the trace and Poseidon2 outputs own separate buffers.
                self.chip_complex
                    .system
                    .memory_inventory
                    .boundary
                    .release_records();
            }
            let replay_error =
                replay_error.map_err(|error| GenerationError::ExtensionTracegen(error.to_string()));
            let ctx = ctx?;
            let replay_error = replay_error?;
            if replay_error != 0 {
                return Err(GenerationError::ExtensionTracegen(format!(
                    "preflight GPU trace generation rejected transcript with code {replay_error}"
                )));
            }
            let ctx = self.validate_proving_ctx(ctx)?;
            finish(producer)?;
            Ok(ctx)
        });
        if result.is_ok() {
            self.preflight_tracegen_poisoned = false;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::{
        begin_preflight_tracegen_session, GenerationError, SystemConfig, VirtualMachine,
        CONNECTOR_AIR_ID, PROGRAM_AIR_ID,
    };
    use crate::{system::SystemCpuBuilder, utils::test_cpu_engine};

    #[test]
    fn late_preflight_coverage_failure_poison_rejects_retry() {
        let mut poisoned = false;
        begin_preflight_tracegen_session(&mut poisoned).unwrap();
        // A late coverage failure deliberately does not complete the session.
        let retry = begin_preflight_tracegen_session(&mut poisoned).unwrap_err();
        assert!(matches!(retry, GenerationError::ProverPoisoned));
    }

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

/// Constructs the continuation proving driver for a VM builder.
///
/// Builders explicitly choose their proving path. A prepared continuation is tied to the exact
/// fixed-program [`VmInstance`] passed to [`Self::prepare_continuation`]. Implementations must not
/// reuse it with another instance.
pub trait ContinuationProverBuilder<E: StarkEngine>: VmBuilder<E> {
    type PreparedContinuation;

    fn prepare_continuation(
        instance: &VmInstance<E, Self>,
    ) -> Result<Self::PreparedContinuation, VirtualMachineError>;

    fn prove_continuation(
        prepared: &mut Self::PreparedContinuation,
        instance: &mut VmInstance<E, Self>,
        input: Streams,
    ) -> Result<ContinuationVmProof<E::SC>, VirtualMachineError>;
}

impl<SC, E, VB> ContinuationProverBuilder<E> for VB
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>>,
    Val<SC>: VmField,
    VB: VmBuilder<E, SystemChipInventory = SystemChipInventory<SC>> + PostflightTracegen<E>,
    <VB::VmConfig as VmExecutionConfig<Val<SC>>>::Executor:
        Executor<Val<SC>> + MeteredExecutor<Val<SC>> + 'static,
{
    type PreparedContinuation = (PreflightInterpreter<Val<SC>, VB::VmConfig>, VB::Prepared);

    fn prepare_continuation(
        instance: &VmInstance<E, Self>,
    ) -> Result<Self::PreparedContinuation, VirtualMachineError> {
        let preflight = instance.vm.preflight_interpreter(instance.exe())?;
        let prepared = VB::prepare_postflight(&instance.vm, &instance.exe().program)?;
        Ok((preflight, prepared))
    }

    fn prove_continuation(
        (preflight, prepared): &mut Self::PreparedContinuation,
        instance: &mut VmInstance<E, Self>,
        input: Streams,
    ) -> Result<ContinuationVmProof<SC>, VirtualMachineError> {
        instance.prove_continuations(preflight, prepared, input)
    }
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
    #[getset(get = "pub")]
    program_commitment: <E::PB as ProverBackend>::Commitment,
    #[getset(get = "pub")]
    exe: Arc<VmExe>,
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
        exe: Arc<VmExe>,
        cached_program_trace: CommittedTraceData<E::PB>,
    ) -> Result<Self, StaticProgramError> {
        let program_commitment = cached_program_trace.commitment;
        vm.load_program(cached_program_trace);
        let state = vm.create_initial_state(&exe, vec![]);
        Ok(Self {
            vm,
            program_commitment,
            exe,
            state: Some(state),
        })
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

/// Fixed-program prover for independently scheduled segments using immutable preflight history.
///
/// The prover owns the VM used to prepare its interpreter, so compiled program
/// data cannot be paired with another executable or proving key.
pub struct SegmentProver<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E> + PostflightTracegen<E>,
    Val<E::SC>: PrimeField32,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>> + 'static,
{
    preflight: PreflightInterpreter<Val<E::SC>, VB::VmConfig>,
    prepared: VB::Prepared,
    exe: Arc<VmExe>,
    instance: VmInstance<E, VB>,
}

impl<E, VB> SegmentProver<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E> + PostflightTracegen<E>,
    Val<E::SC>: VmField,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>> + 'static,
{
    pub fn new(instance: VmInstance<E, VB>) -> Result<Self, VirtualMachineError> {
        let preflight = instance.vm.preflight_interpreter(instance.exe())?;
        let exe = Arc::clone(instance.exe());
        let prepared = VB::prepare_postflight(&instance.vm, &exe.program)?;
        Ok(Self {
            preflight,
            prepared,
            exe,
            instance,
        })
    }

    /// Proves one segment from an arbitrary segment-start state.
    ///
    /// Final memory is returned only when the segment terminates successfully.
    pub fn prove(
        &mut self,
        state: VmState<GuestMemory>,
        segment: &Segment,
    ) -> Result<(Proof<E::SC>, Option<GuestMemory>), VirtualMachineError> {
        let (proof, output) = self.instance.vm.prove_segment_inner(
            &self.preflight,
            &self.exe.program,
            &self.prepared,
            state,
            segment,
        )?;
        let final_memory =
            (output.exit_code == Some(ExitCode::Success as u32)).then_some(output.state.memory);
        Ok((proof, final_memory))
    }

    pub fn vm(&self) -> &VirtualMachine<E, VB> {
        &self.instance.vm
    }
}

impl<E, VB> ContinuationVmProver<E::SC> for VmInstance<E, VB>
where
    E: StarkEngine,
    Val<E::SC>: VmField,
    VB: VmBuilder<E> + PostflightTracegen<E>,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
        Executor<Val<E::SC>> + MeteredExecutor<Val<E::SC>> + 'static,
{
    /// First performs metered execution to determine segments. Then sequentially proves each
    /// segment. The proof for each segment uses the specified [ProverBackend], but the proof for
    /// the next segment does not start before the current proof finishes.
    fn prove(
        &mut self,
        input: impl Into<Streams>,
    ) -> Result<ContinuationVmProof<E::SC>, VirtualMachineError> {
        let preflight = self.vm.preflight_interpreter(&self.exe)?;
        let prepared = VB::prepare_postflight(&self.vm, &self.exe.program)?;
        self.prove_continuations(&preflight, &prepared, input.into())
    }
}

impl<E, VB> VmInstance<E, VB>
where
    E: StarkEngine,
    Val<E::SC>: VmField,
    VB: VmBuilder<E>,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
        Executor<Val<E::SC>> + MeteredExecutor<Val<E::SC>> + 'static,
    VB: PostflightTracegen<E>,
{
    pub(crate) fn prove_continuations(
        &mut self,
        preflight: &PreflightInterpreter<Val<E::SC>, VB::VmConfig>,
        prepared: &VB::Prepared,
        input: Streams,
    ) -> Result<ContinuationVmProof<E::SC>, VirtualMachineError> {
        if self.state.is_none() {
            return Err(GenerationError::ProverPoisoned.into());
        }
        self.reset_state(input.clone());
        let vm = &mut self.vm;
        let metered_ctx = vm.build_metered_ctx(&self.exe);
        let metered_instance = vm.metered_instance(&self.exe)?;
        let (segments, _) = metered_instance.execute_metered(input, metered_ctx)?;
        let mut proofs = Vec::with_capacity(segments.len());
        let mut state = self.state.take();
        for (seg_idx, segment) in segments.into_iter().enumerate() {
            let _segment_span = info_span!("prove_segment", segment = seg_idx).entered();
            // We need a separate span so the metric label includes "segment" from _segment_span
            let _prove_span = info_span!("total_proof").entered();
            let from_state = Option::take(&mut state).unwrap();
            let (proof, output) = vm.prove_segment_inner(
                preflight,
                &self.exe.program,
                prepared,
                from_state,
                &segment,
            )?;
            proofs.push(proof);
            state = Some(output.state);
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
    let mut prev_final_pc_idx = None;
    let mut start_pc_idx = None;
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
                    // Check the initial PC index against the previous final PC index.
                    if pvs.initial_pc_idx != prev_final_pc_idx.unwrap() {
                        return Err(VmVerificationError::InitialPcIdxMismatch {
                            initial: pvs.initial_pc_idx.as_canonical_u32(),
                            prev_final: prev_final_pc_idx.unwrap().as_canonical_u32(),
                        });
                    }
                } else {
                    start_pc_idx = Some(pvs.initial_pc_idx);
                }
                prev_final_pc_idx = Some(pvs.final_pc_idx);

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
        start_pc_idx.unwrap(),
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
