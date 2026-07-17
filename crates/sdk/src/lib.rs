#![cfg_attr(feature = "tco", allow(incomplete_features))]

#[cfg(feature = "rvr")]
use std::path::PathBuf;
use std::{
    fs::read,
    path::Path,
    sync::{Arc, OnceLock},
};

use config::AppConfig;
use getset::Getters;
use keygen::{AppProvingKey, AppVerifyingKey};
use openvm_build::{
    build_guest_package, find_unique_executable, get_package, GuestOptions, TargetFilter,
};
// Re-exports
pub use openvm_build::{cargo_command, get_rustup_toolchain_name};
#[cfg(feature = "rvr")]
pub use openvm_circuit::arch::rvr::{RvrTrackedExecution, RvrTrackedExecutionOutcome};
#[cfg(feature = "rvr")]
use openvm_circuit::arch::{
    execution_mode::MeteredCtx,
    instructions::program::DEFAULT_PC_STEP,
    rvr::{default_addr2line_cmd, GuestDebugMap},
};
pub use openvm_circuit::{self, arch::ExecutionOutcome};
use openvm_circuit::{
    arch::{
        execution_mode::Segment, instructions::exe::VmExe, Executor, InitFileGenerator,
        MeteredExecutor, PreflightExecutor, VirtualMachineError, VmBuilder, VmExecutionConfig,
        VmExecutor, U16_CELL_SIZE,
    },
    system::memory::merkle::public_values::extract_public_values,
};
use openvm_continuations::CommitBytes;
use openvm_sdk_config::{SdkVmConfig, SdkVmCpuBuilder, TranspilerConfig};
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey, prover::ProverDevice, StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2CpuEngine as BabyBearPoseidon2Engine, Digest,
};
#[cfg(feature = "evm-prove")]
use openvm_static_verifier::StaticVerifierShape;
use openvm_transpiler::{
    elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler, FromElf,
};
use openvm_verify_stark_host::{
    verify_vm_stark_proof_decoded,
    vk::{VerificationBaseline, VmStarkVerifyingKey},
    VmStarkProof,
};
pub use types::{ExecutableFormat, ExecutableInput};

#[cfg(feature = "rvr")]
use crate::compiled::load_metered_artifact_metadata;
use crate::{
    config::{AggregationConfig, AggregationSystemParams, AggregationTreeConfig},
    keygen::{AggPrefixProvingKey, AggProvingKey, SdkCachedProvingKey},
    prover::{AggProver, AppProver, DeferralAggProver, DeferralHookCommits, StarkProver},
    types::AppExecutionCommit,
};
#[cfg(feature = "evm-prove")]
use crate::{halo2_params::CacheHalo2ParamsReader, keygen::Halo2ProvingKey, prover::Halo2Prover};
#[cfg(feature = "root-prover")]
use crate::{
    keygen::{dummy::compute_root_proof_heights, RootProvingKey},
    prover::{EvmProver, RootProver},
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_sdk_config::SdkVmGpuBuilder;
        use openvm_cuda_backend::BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine;
        pub use GpuSdk as Sdk;
        pub type DefaultStarkEngine = GpuBabyBearPoseidon2Engine;
    } else {
        pub use CpuSdk as Sdk;
        pub type DefaultStarkEngine = BabyBearPoseidon2Engine;
    }
}

pub use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config as SC, F};

pub mod builder;
pub mod compiled;
pub mod config;
pub mod fs;
#[cfg(feature = "evm-prove")]
pub mod halo2_params;
pub mod keygen;
pub mod prover;
#[cfg(feature = "evm-verify")]
mod solidity;
pub mod types;
pub mod util;

#[cfg(test)]
mod tests;

mod error;
mod stdin;
#[cfg(feature = "rvr")]
pub use compiled::CompiledExePureWithInstretTracking;
pub use compiled::{CompiledExeMetered, CompiledExeMeteredCost, CompiledExePure};
pub use error::SdkError;
pub use stdin::*;

pub const OPENVM_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION_MAJOR"),
    ".",
    env!("CARGO_PKG_VERSION_MINOR")
);

struct CompileInput {
    executable: ExecutableFormat,
    #[cfg(feature = "rvr")]
    elf_path: Option<PathBuf>,
}

// The SDK is only generic in the engine for the non-root SC. The root SC is fixed to
// BabyBearPoseidon2RootEngine right now.
/// The SDK provides convenience methods and constructors for provers.
///
/// A built SDK is an immutable proving environment: user-supplied config, params, and pre-generated
/// keys are fixed after construction. Use [`builder`](Self::builder) for advanced initialization,
/// or [`new`](Self::new) / [`new_without_transpiler`](Self::new_without_transpiler) for the
/// common config-driven paths.
///
/// Internally, the SDK lazily caches proving state that depends only on the app VM config,
/// aggregation config, root params, and optional pre-generated keys. It does not cache any state
/// that depends on the program executable.
///
/// Some commonly used methods are:
/// - [`compile_and_execute`](Self::compile_and_execute)
/// - [`prove`](Self::prove)
/// - [`verify_proof`](Self::verify_proof)
#[derive(Getters)]
pub struct GenericSdk<E, VB>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E>,
    VB::VmConfig: VmExecutionConfig<F>,
{
    #[getset(get = "pub")]
    app_config: AppConfig<VB::VmConfig>,
    #[getset(get = "pub")]
    agg_config: AggregationConfig,
    #[getset(get = "pub")]
    agg_tree_config: AggregationTreeConfig,
    #[cfg(feature = "root-prover")]
    #[getset(get = "pub")]
    root_params: SystemParams,
    #[cfg(feature = "evm-prove")]
    #[getset(get = "pub")]
    halo2_shape: StaticVerifierShape,
    #[cfg(feature = "evm-prove")]
    #[getset(get = "pub")]
    halo2_config: config::Halo2Config,

    #[getset(get = "pub")]
    app_vm_builder: VB,

    transpiler: Option<Transpiler<F>>,

    /// The `executor` may be used to construct different types of interpreters, given the program,
    /// for more specific execution purposes. By default, it is recommended to use the
    /// [`execute`](GenericSdk::execute) method.
    #[getset(get = "pub")]
    executor: VmExecutor<F, VB::VmConfig>,

    app_pk: OnceLock<AppProvingKey<VB::VmConfig>>,
    agg_prover: OnceLock<Arc<AggProver>>,
    #[cfg(feature = "root-prover")]
    root_prover: OnceLock<Arc<RootProver>>,

    deferral_setup: DeferralSetup,

    #[cfg(feature = "evm-prove")]
    #[getset(get = "pub")]
    halo2_params_reader: CacheHalo2ParamsReader,
    #[cfg(feature = "evm-prove")]
    halo2_prover: OnceLock<Halo2Prover>,
}

#[derive(Clone, Default)]
pub enum DeferralSetup {
    #[default]
    /// Deferrals are disabled for this GenericSdk. The STARK, root, and halo2 vks are deferral-
    /// unaware, and trying to prove with def_inputs set causes an error.
    Disabled,
    /// The STARK, root, and halo2 vks are deferral-aware (i.e. are equivalent to Active), but
    /// trying to prove with def_inputs set still causes an error. Use this when a program that
    /// doesn't use deferrals must be verified by a deferral-aware vk.
    Aware(DeferralHookCommits),
    /// The STARK, root, and halo2 vks are deferral-aware (i.e. are equivalent to Aware), and
    /// deferral inputs can be used and proved.
    Active(Arc<DeferralAggProver>),
}

impl DeferralSetup {
    pub fn hook_cached_commit(&self) -> Option<Digest> {
        match self {
            Self::Disabled => None,
            Self::Aware(commits) => Some(commits.hook_cached_commit),
            Self::Active(prover) => Some(prover.def_hook_cached_commit()),
        }
    }

    pub fn hook_commit(&self) -> Option<Digest> {
        match self {
            Self::Disabled => None,
            Self::Aware(commits) => Some(commits.hook_commit),
            Self::Active(prover) => Some(prover.def_hook_commit()),
        }
    }

    pub fn prover(&self) -> Option<Arc<DeferralAggProver>> {
        match self {
            Self::Active(prover) => Some(prover.clone()),
            Self::Disabled | Self::Aware(_) => None,
        }
    }
}

pub type CpuSdk = GenericSdk<BabyBearPoseidon2Engine, SdkVmCpuBuilder>;

#[cfg(feature = "cuda")]
pub type GpuSdk = GenericSdk<GpuBabyBearPoseidon2Engine, SdkVmGpuBuilder>;

impl<E, VB> GenericSdk<E, VB>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E, VmConfig = SdkVmConfig> + Clone + Default,
{
    /// Creates SDK with a standard configuration that includes a set of default VM extensions
    /// loaded.
    ///
    /// **Note**: To use this configuration, your `openvm.toml` must match, including the order of
    /// the moduli and elliptic curve parameters of the respective extensions:
    /// The `app_vm_config` field of your `openvm.toml` must exactly match the following:
    ///
    /// ```toml
    #[doc = include_str!("../../sdk-config/src/openvm_standard.toml")]
    /// ```
    pub fn standard(app_params: SystemParams, agg_params: AggregationSystemParams) -> Self {
        GenericSdk::new(AppConfig::standard(app_params), agg_params).unwrap()
    }

    /// Creates SDK with a configuration with RISC-V RV64IM and IO VM extensions loaded.
    ///
    /// **Note**: To use this configuration, your `openvm.toml` must exactly match the following:
    ///
    /// ```toml
    #[doc = include_str!("../../sdk-config/src/openvm_riscv64.toml")]
    /// ```
    pub fn riscv64(app_params: SystemParams, agg_params: AggregationSystemParams) -> Self {
        GenericSdk::new(AppConfig::riscv64(app_params), agg_params).unwrap()
    }
}

impl<E, VB> GenericSdk<E, VB>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E>,
{
    /// Creates SDK custom to the given [AppConfig], with a RISC-V transpiler.
    pub fn new(
        app_config: AppConfig<VB::VmConfig>,
        agg_params: AggregationSystemParams,
    ) -> Result<Self, SdkError>
    where
        VB: Default,
        VB::VmConfig: TranspilerConfig<F>,
    {
        Self::builder()
            .app_config(app_config)
            .agg_params(agg_params)
            .build()
    }

    /// Creates an SDK custom to the given [AppConfig] without configuring a transpiler.
    ///
    /// **Note**: This function does not set the transpiler, which must be done separately to
    /// support RISC-V ELFs.
    pub fn new_without_transpiler(
        app_config: AppConfig<VB::VmConfig>,
        agg_params: AggregationSystemParams,
    ) -> Result<Self, SdkError>
    where
        VB: Default,
    {
        Self::builder()
            .app_config(app_config)
            .agg_params(agg_params)
            .build_without_transpiler()
    }

    /// Returns the def_hook_prover cached commit.
    pub fn def_hook_cached_commit(&self) -> Option<Digest> {
        self.deferral_setup.hook_cached_commit()
    }

    /// Returns the deferral hook commit derived from the deferral aggregation path.
    pub fn def_hook_commit(&self) -> Option<Digest> {
        self.deferral_setup.hook_commit()
    }

    /// Returns the deferral aggregation prover when this SDK can prove non-empty deferral inputs.
    pub fn deferral_agg_prover(&self) -> Option<Arc<DeferralAggProver>> {
        self.deferral_setup.prover()
    }

    /// Derives the cached commits that the deferral circuit at `def_idx` expects callers to fold
    /// into its input commit.
    pub fn deferral_circuit_cached_commits(
        &self,
        def_idx: usize,
    ) -> Result<Vec<CommitBytes>, SdkError> {
        let deferral_prover = self.deferral_setup.prover().ok_or_else(|| {
            SdkError::Other(eyre::eyre!(
                "deferral circuit cached commits require an active deferral prover"
            ))
        })?;
        deferral_prover
            .multi_deferral_circuit_prover
            .single_circuit_provers
            .get(def_idx)
            .map(|prover| prover.def_circuit_prover.cached_commits())
            .ok_or_else(|| {
                SdkError::Other(eyre::eyre!(
                    "deferral circuit index {def_idx} is not configured"
                ))
            })
    }

    /// Returns serde-serializable proving keys for this SDK.
    ///
    /// This errors if `app_pk` or `agg_pk` have not already been generated or seeded into the SDK.
    /// Optional keys are returned only if they are already cached or were seeded into the SDK.
    ///
    /// Halo2 proving keys are intentionally excluded; use `write_halo2_pk_to_file` and
    /// `read_halo2_pk_from_file` for those.
    pub fn cached_proving_key(&self) -> Result<SdkCachedProvingKey<VB::VmConfig>, SdkError> {
        let app_pk = self
            .app_pk
            .get()
            .ok_or_else(|| SdkError::Other(eyre::eyre!("app_pk is not generated")))?
            .clone();
        let agg_prover = self
            .agg_prover
            .get()
            .ok_or_else(|| SdkError::Other(eyre::eyre!("agg_pk is not generated")))?;
        let deferral_prover = self.deferral_setup.prover();
        Ok(SdkCachedProvingKey {
            app_pk,
            agg_pk: AggProvingKey {
                prefix: AggPrefixProvingKey {
                    leaf: agg_prover.leaf_prover.get_pk(),
                    internal_for_leaf: agg_prover.internal_for_leaf_prover.get_pk(),
                },
                internal_recursive: agg_prover.internal_recursive_prover.get_pk(),
            },
            deferral_pk: deferral_prover
                .as_ref()
                .map(|def_agg_prover| def_agg_prover.multi_deferral_circuit_prover.get_pk()),
            deferral_agg_pk: deferral_prover
                .as_ref()
                .map(|def_agg_prover| def_agg_prover.get_pk()),
            #[cfg(feature = "root-prover")]
            root_pk: self.root_prover.get().map(|root_prover| RootProvingKey {
                root_pk: root_prover.0.get_pk(),
                trace_heights: root_prover.0.get_trace_heights().unwrap_or_default(),
            }),
        })
    }

    /// Builds the guest package located at `pkg_dir`. This function requires that the build target
    /// is unique and errors otherwise. Returns the built ELF file decoded in the [Elf] type.
    pub fn build<P: AsRef<Path>>(
        &self,
        guest_opts: GuestOptions,
        pkg_dir: P,
        target_filter: &Option<TargetFilter>,
        init_file_name: Option<&str>, // If None, we use "openvm-init.rs"
    ) -> Result<Elf, SdkError> {
        self.app_config
            .app_vm_config
            .write_to_init_file(pkg_dir.as_ref(), init_file_name)?;
        let pkg = get_package(pkg_dir.as_ref());
        let target_dir = match build_guest_package(&pkg, &guest_opts, None, target_filter) {
            Ok(target_dir) => target_dir,
            Err(Some(code)) => {
                return Err(SdkError::BuildFailedWithCode(code));
            }
            Err(None) => {
                return Err(SdkError::BuildFailed);
            }
        };

        let elf_path =
            find_unique_executable(pkg_dir, target_dir, target_filter).map_err(SdkError::Other)?;
        let data = read(&elf_path)?;
        Elf::decode(&data, MEM_SIZE as u32).map_err(SdkError::Other)
    }

    /// Transpiler for transpiling RISC-V ELF to OpenVM executable.
    pub fn transpiler(&self) -> Result<&Transpiler<F>, SdkError> {
        self.transpiler
            .as_ref()
            .ok_or(SdkError::TranspilerNotAvailable)
    }

    /// Normalizes an ELF or executable handle into a shared [`VmExe`].
    pub fn convert_to_exe(
        &self,
        executable: impl Into<ExecutableFormat>,
    ) -> Result<Arc<VmExe<F>>, SdkError> {
        let executable = executable.into();
        let exe = match executable {
            ExecutableFormat::Elf(elf) => {
                let transpiler = self.transpiler()?.clone();
                Arc::new(VmExe::from_elf(elf, transpiler)?)
            }
            ExecutableFormat::VmExe(exe) => Arc::new(exe),
            ExecutableFormat::SharedVmExe(exe) => exe,
        };
        Ok(exe)
    }

    fn compile_input(
        &self,
        executable: impl Into<ExecutableInput>,
    ) -> Result<CompileInput, SdkError> {
        let executable = executable.into();
        match executable {
            ExecutableInput::Format(format) => Ok(CompileInput {
                executable: format,
                #[cfg(feature = "rvr")]
                elf_path: None,
            }),
            ExecutableInput::ElfFile(path) => {
                let bytes = read(&path)?;
                let elf = Elf::decode(&bytes, MEM_SIZE as u32)?;
                Ok(CompileInput {
                    executable: ExecutableFormat::Elf(elf),
                    #[cfg(feature = "rvr")]
                    elf_path: Some(path),
                })
            }
            #[cfg(feature = "rvr")]
            ExecutableInput::WithElfPath {
                executable,
                elf_path,
            } => Ok(CompileInput {
                executable,
                elf_path: Some(elf_path),
            }),
        }
    }

    #[cfg(feature = "rvr")]
    fn guest_debug_map(&self, elf_path: &Path, exe: &VmExe<F>) -> Result<GuestDebugMap, SdkError> {
        let pcs = exe
            .program
            .instructions_and_debug_infos
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| {
                slot.as_ref()
                    .map(|_| exe.program.pc_base + (index as u32) * DEFAULT_PC_STEP)
            })
            .collect::<Vec<_>>();
        GuestDebugMap::from_elf(elf_path, &pcs, &default_addr2line_cmd())
            .map_err(|err| SdkError::Other(eyre::eyre!(err)))
    }
}

// The SDK is only functional for SC = BabyBearPoseidon2Config because that is what recursive
// aggregation supports.
impl<E, VB> GenericSdk<E, VB>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E> + Clone,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor:
        Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, VB::RecordArena>,
{
    /// Compile `app_exe` and execute it, returning the user public values as bytes.
    pub fn compile_and_execute(
        &self,
        app_exe: impl Into<ExecutableInput>,
        inputs: StdIn,
    ) -> Result<Vec<u8>, SdkError> {
        let compiled = self.compile(app_exe)?;
        self.execute(&compiled, inputs)
    }

    /// Compile `app_exe` for pure execution.
    #[tracing::instrument(name = "sdk.compile", level = "info", skip_all)]
    pub fn compile(
        &self,
        app_exe: impl Into<ExecutableInput>,
    ) -> Result<CompiledExePure<'_>, SdkError> {
        let input = self.compile_input(app_exe)?;
        let exe = self.convert_to_exe(input.executable)?;
        #[cfg(feature = "rvr")]
        {
            let guest_debug_map = input
                .elf_path
                .as_deref()
                .map(|elf_path| self.guest_debug_map(elf_path, &exe))
                .transpose()?;
            self.executor
                .rvr_instance(&exe, guest_debug_map.as_ref())
                .map(CompiledExePure::new)
                .map_err(VirtualMachineError::from)
                .map_err(SdkError::from)
        }
        #[cfg(not(feature = "rvr"))]
        self.executor
            .instance(&exe)
            .map(CompiledExePure::new)
            .map_err(VirtualMachineError::from)
            .map_err(SdkError::from)
    }

    /// Compile `app_exe` for pure RVR execution with instret tracking and block-boundary
    /// suspension.
    #[cfg(feature = "rvr")]
    #[tracing::instrument(name = "sdk.compile_with_instret_tracking", level = "info", skip_all)]
    pub fn compile_with_instret_tracking(
        &self,
        app_exe: impl Into<ExecutableInput>,
    ) -> Result<CompiledExePureWithInstretTracking<'_>, SdkError> {
        let input = self.compile_input(app_exe)?;
        let exe = self.convert_to_exe(input.executable)?;
        let guest_debug_map = input
            .elf_path
            .as_deref()
            .map(|elf_path| self.guest_debug_map(elf_path, &exe))
            .transpose()?;
        self.executor
            .rvr_instret_tracking_instance(&exe, guest_debug_map.as_ref())
            .map(CompiledExePureWithInstretTracking::new)
            .map_err(VirtualMachineError::from)
            .map_err(SdkError::from)
    }

    /// Load a previously saved pure-mode RVR artifact.
    ///
    /// Its generated execution kind is validated from the artifact marker,
    /// but no general `exe`, config, or code-version validation is performed.
    #[cfg(feature = "rvr")]
    pub fn load_compiled(
        &self,
        lib_path: &Path,
        app_exe: impl Into<ExecutableFormat>,
    ) -> Result<CompiledExePure<'_>, SdkError> {
        let exe = self.convert_to_exe(app_exe)?;
        self.executor
            .load_instance(lib_path, &exe)
            .map(CompiledExePure::new)
            .map_err(VirtualMachineError::from)
            .map_err(SdkError::from)
    }

    /// Load a previously saved pure RVR artifact with instret tracking.
    #[cfg(feature = "rvr")]
    pub fn load_compiled_with_instret_tracking(
        &self,
        lib_path: &Path,
        app_exe: impl Into<ExecutableFormat>,
    ) -> Result<CompiledExePureWithInstretTracking<'_>, SdkError> {
        let exe = self.convert_to_exe(app_exe)?;
        self.executor
            .load_instret_tracking_instance(lib_path, &exe)
            .map(CompiledExePureWithInstretTracking::new)
            .map_err(VirtualMachineError::from)
            .map_err(SdkError::from)
    }

    /// Run a [`CompiledExePure`] against `inputs` and extract the user public values.
    #[tracing::instrument(name = "sdk.execute", level = "info", skip_all)]
    pub fn execute(
        &self,
        compiled: &CompiledExePure<'_>,
        inputs: StdIn,
    ) -> Result<Vec<u8>, SdkError> {
        let final_memory = compiled
            .execute(inputs)
            .map_err(VirtualMachineError::from)?
            .memory;
        let public_values = extract_public_values(
            self.executor.config.as_ref().num_public_values * U16_CELL_SIZE,
            &final_memory.memory,
        );
        Ok(public_values)
    }

    /// Executes with segmentation for proof generation.
    /// Returns both user public values and segments with instruction counts and trace heights.
    pub fn compile_and_execute_metered(
        &self,
        app_exe: impl Into<ExecutableInput>,
        inputs: StdIn,
    ) -> Result<(Vec<u8>, Vec<Segment>), SdkError> {
        let compiled = self.compile_metered(app_exe)?;
        self.execute_metered(&compiled, inputs)
    }

    /// Compile `app_exe` for metered execution. The returned [`CompiledExeMetered`] bundles
    /// a precomputed `MeteredCtx` so subsequent runs just clone it.
    #[tracing::instrument(name = "sdk.compile_metered", level = "info", skip_all)]
    pub fn compile_metered(
        &self,
        app_exe: impl Into<ExecutableInput>,
    ) -> Result<CompiledExeMetered<'_>, SdkError> {
        let input = self.compile_input(app_exe)?;
        let app_prover = self.app_prover(input.executable)?;

        let vm = app_prover.vm();
        let exe = app_prover.exe();

        let ctx = vm.build_metered_ctx(&exe);
        let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
        #[cfg(feature = "rvr")]
        let guest_debug_map = input
            .elf_path
            .as_deref()
            .map(|elf_path| self.guest_debug_map(elf_path, &exe))
            .transpose()?;
        #[cfg(feature = "rvr")]
        let instance = self
            .executor
            .metered_rvr_instance(
                &exe,
                &executor_idx_to_air_idx,
                ctx.trace_heights.len(),
                guest_debug_map.as_ref(),
            )
            .map_err(VirtualMachineError::from)?;
        #[cfg(not(feature = "rvr"))]
        let instance = self
            .executor
            .metered_instance(&exe, &executor_idx_to_air_idx)
            .map_err(VirtualMachineError::from)?;
        Ok(CompiledExeMetered {
            instance,
            ctx,
            #[cfg(feature = "rvr")]
            executor_idx_to_air_idx,
        })
    }

    /// Load a previously saved metered-mode artifact. The `MeteredCtx`
    /// is rebuilt. Caller supplies `app_exe`; no compatibility validation is performed.
    #[cfg(feature = "rvr")]
    pub fn load_compiled_metered(
        &self,
        lib_path: &Path,
        app_exe: impl Into<ExecutableFormat>,
    ) -> Result<CompiledExeMetered<'_>, SdkError> {
        let metadata = load_metered_artifact_metadata(lib_path).map_err(SdkError::Other)?;
        let exe = self.convert_to_exe(app_exe)?;
        let mut ctx = MeteredCtx::from_config(
            metadata.metered_ctx_config,
            metadata.segmentation_config,
            self.executor.config.as_ref(),
        );
        ctx.seed_initial_memory(&exe.init_memory);
        let instance = self
            .executor
            .load_metered_instance(lib_path, &exe, &metadata.executor_idx_to_air_idx)
            .map_err(VirtualMachineError::from)?;
        Ok(CompiledExeMetered {
            instance,
            ctx,
            executor_idx_to_air_idx: metadata.executor_idx_to_air_idx,
        })
    }

    /// Run a [`CompiledExeMetered`] against `inputs`.
    #[tracing::instrument(name = "sdk.execute_metered", level = "info", skip_all)]
    pub fn execute_metered(
        &self,
        compiled: &CompiledExeMetered<'_>,
        inputs: StdIn,
    ) -> Result<(Vec<u8>, Vec<Segment>), SdkError> {
        let (segments, final_state) = compiled
            .instance
            .execute_metered(inputs, compiled.ctx.clone())
            .map_err(VirtualMachineError::from)?;
        let public_values = extract_public_values(
            self.executor.config.as_ref().num_public_values * U16_CELL_SIZE,
            &final_state.memory.memory,
        );

        Ok((public_values, segments))
    }

    /// Executes with cost metering to measure computational cost in trace cells.
    /// Returns both user public values, and cost along with instruction count.
    pub fn compile_and_execute_metered_cost(
        &self,
        app_exe: impl Into<ExecutableInput>,
        inputs: StdIn,
    ) -> Result<(Vec<u8>, (u64, u64)), SdkError> {
        let compiled = self.compile_metered_cost(app_exe)?;
        self.execute_metered_cost(&compiled, inputs)
    }

    /// Compile `app_exe` for metered-cost execution. See [`Self::compile_metered`].
    #[tracing::instrument(name = "sdk.compile_metered_cost", level = "info", skip_all)]
    pub fn compile_metered_cost(
        &self,
        app_exe: impl Into<ExecutableInput>,
    ) -> Result<CompiledExeMeteredCost<'_>, SdkError> {
        let input = self.compile_input(app_exe)?;
        let app_prover = self.app_prover(input.executable)?;

        let vm = app_prover.vm();
        let exe = app_prover.exe();

        let ctx = vm.build_metered_cost_ctx();
        let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
        #[cfg(feature = "rvr")]
        let guest_debug_map = input
            .elf_path
            .as_deref()
            .map(|elf_path| self.guest_debug_map(elf_path, &exe))
            .transpose()?;
        #[cfg(feature = "rvr")]
        let instance = self
            .executor
            .metered_cost_rvr_instance(
                &exe,
                &executor_idx_to_air_idx,
                &ctx.widths,
                guest_debug_map.as_ref(),
            )
            .map_err(VirtualMachineError::from)?;
        #[cfg(not(feature = "rvr"))]
        let instance = self
            .executor
            .metered_cost_instance(&exe, &executor_idx_to_air_idx)
            .map_err(VirtualMachineError::from)?;
        Ok(CompiledExeMeteredCost { instance, ctx })
    }

    /// Load a previously saved metered-cost-mode artifact. The `MeteredCostCtx` is
    /// rebuilt. Caller supplies `app_exe`; no compatibility validation is performed.
    #[cfg(feature = "rvr")]
    pub fn load_compiled_metered_cost(
        &self,
        lib_path: &Path,
        app_exe: impl Into<ExecutableFormat>,
    ) -> Result<CompiledExeMeteredCost<'_>, SdkError> {
        let app_prover = self.app_prover(app_exe)?;
        let vm = app_prover.vm();
        let exe = app_prover.exe();

        let ctx = vm.build_metered_cost_ctx();
        let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
        let instance = self
            .executor
            .load_metered_cost_instance(lib_path, &exe, &executor_idx_to_air_idx, &ctx.widths)
            .map_err(VirtualMachineError::from)?;
        Ok(CompiledExeMeteredCost { instance, ctx })
    }

    /// Run a [`CompiledExeMeteredCost`] against `inputs`.
    #[tracing::instrument(name = "sdk.execute_metered_cost", level = "info", skip_all)]
    pub fn execute_metered_cost(
        &self,
        compiled: &CompiledExeMeteredCost<'_>,
        inputs: StdIn,
    ) -> Result<(Vec<u8>, (u64, u64)), SdkError> {
        let (ctx, final_state) = compiled
            .instance
            .execute_metered_cost(inputs, compiled.ctx.clone())
            .map_err(VirtualMachineError::from)?;
        let instret = ctx.instret;
        let cost = ctx.cost;

        let public_values = extract_public_values(
            self.executor.config.as_ref().num_public_values * U16_CELL_SIZE,
            &final_state.memory.memory,
        );

        Ok((public_values, (cost, instret)))
    }

    // ======================== Proving Methods ============================

    /// Generates a single aggregate STARK proof of the full program execution of the given
    /// `app_exe` with program inputs `inputs`.\
    ///
    /// For convenience, this function also returns the [VerificationBaseline], which is a full
    /// commitment to the App [VmExe] and aggregation verifiers. It does **not** depend on the
    /// `inputs`. It can be generated separately from the proof by creating a
    /// [`prover`](Self::prover) and calling
    /// [`app_vm_commit`](StarkProver::app_vm_commit).
    ///
    /// If STARK aggregation is not needed and a proof whose size may grow linearly with the length
    /// of the program runtime is desired, create an [`app_prover`](Self::app_prover) and call
    /// [`app_prover.prove(inputs)`](AppProver::prove).
    pub fn prove(
        &self,
        app_exe: impl Into<ExecutableFormat>,
        inputs: StdIn,
        def_inputs: &[DeferralInput],
    ) -> Result<(VmStarkProof, VerificationBaseline), SdkError>
    where
        <E::PD as ProverDevice<E::PB, E::TS>>::DeviceCtx: 'static,
    {
        let mut prover = self.prover(app_exe)?;
        let proof = prover.prove(inputs, def_inputs)?.0;
        let baseline = prover.generate_baseline();
        Ok((proof, baseline))
    }

    #[cfg(feature = "evm-prove")]
    /// Generates an EVM-verifiable proof for the given executable and inputs.
    pub fn prove_evm(
        &self,
        app_exe: impl Into<ExecutableFormat>,
        inputs: StdIn,
        def_inputs: &[DeferralInput],
    ) -> Result<types::EvmProof, SdkError>
    where
        <E::PD as ProverDevice<E::PB, E::TS>>::DeviceCtx: 'static,
    {
        let app_exe = self.convert_to_exe(app_exe)?;
        let mut evm_prover = self.evm_prover(app_exe)?;
        let evm_proof = evm_prover.prove_evm(inputs, def_inputs)?;
        Ok(evm_proof)
    }

    // ========================= Prover Constructors =========================

    /// This constructor is for generating app proofs that do not require a single aggregate STARK
    /// proof of the full program execution. For a single STARK proof, use the
    /// [`prove`](Self::prove) method instead.
    ///
    /// Creates an app prover instance specific to the provided exe.
    /// This function will generate the [AppProvingKey] if it doesn't already exist and use it to
    /// construct the [AppProver].
    pub fn app_prover(
        &self,
        exe: impl Into<ExecutableFormat>,
    ) -> Result<AppProver<E, VB>, SdkError> {
        let exe = self.convert_to_exe(exe)?;
        let app_pk = self.app_pk();
        let prover = AppProver::<E, VB>::new(self.app_vm_builder.clone(), &app_pk.app_vm_pk, exe)?;
        Ok(prover)
    }

    /// Constructs a new [StarkProver] instance for the given executable.
    /// This function will generate the [AppProvingKey] if it does not already
    /// exist.
    pub fn prover(
        &self,
        app_exe: impl Into<ExecutableFormat>,
    ) -> Result<StarkProver<E, VB>, SdkError> {
        let app_exe = self.convert_to_exe(app_exe)?;
        let app_pk = self.app_pk();
        let stark_prover = StarkProver::<E, _>::new(
            self.app_vm_builder.clone(),
            &app_pk.app_vm_pk,
            app_exe,
            self.agg_prover(),
            self.deferral_setup.clone(),
        )?;
        Ok(stark_prover)
    }

    #[cfg(feature = "root-prover")]
    /// Constructs an [`EvmProver`] for the given executable with only the root prover, generating
    /// prerequisite keys, lazily
    pub fn evm_prover_without_halo2(
        &self,
        app_exe: impl Into<ExecutableFormat>,
    ) -> Result<EvmProver<E, VB>, SdkError> {
        let app_exe = self.convert_to_exe(app_exe)?;
        let app_pk = self.app_pk();
        let evm_prover = EvmProver::<E, _>::new(
            self.app_vm_builder.clone(),
            &app_pk.app_vm_pk,
            app_exe,
            self.agg_prover(),
            self.deferral_setup.clone(),
            self.root_prover(),
            #[cfg(feature = "evm-prove")]
            None,
        )?;
        Ok(evm_prover)
    }

    #[cfg(all(feature = "root-prover", feature = "evm-prove"))]
    /// Constructs an [`EvmProver`] for the given executable, generating prerequisite keys lazily.
    pub fn evm_prover(
        &self,
        app_exe: impl Into<ExecutableFormat>,
    ) -> Result<EvmProver<E, VB>, SdkError>
    where
        <E::PD as ProverDevice<E::PB, E::TS>>::DeviceCtx: 'static,
    {
        let mut evm_prover = self.evm_prover_without_halo2(app_exe)?;
        evm_prover.halo2_prover = Some(self.halo2_prover());
        Ok(evm_prover)
    }

    #[cfg(all(feature = "root-prover", not(feature = "evm-prove")))]
    /// Constructs an [`EvmProver`] for the given executable, generating prerequisite keys lazily.
    pub fn evm_prover(
        &self,
        app_exe: impl Into<ExecutableFormat>,
    ) -> Result<EvmProver<E, VB>, SdkError> {
        self.evm_prover_without_halo2(app_exe)
    }

    // ===================== Component Prover Constructors =====================

    /// Returns the cached aggregation prover, generating it on first use if needed.
    pub fn agg_prover(&self) -> Arc<AggProver> {
        let app_pk = self.app_pk();
        self.agg_prover
            .get_or_init(|| {
                Arc::new(AggProver::new(
                    Arc::new(app_pk.app_vm_pk.vm_pk.get_vk()),
                    self.agg_config.clone(),
                    self.agg_tree_config,
                    self.def_hook_cached_commit(),
                ))
            })
            .clone()
    }

    #[cfg(feature = "root-prover")]
    /// Returns the cached root prover, generating it on first use if needed.
    pub fn root_prover(&self) -> Arc<RootProver> {
        self.root_prover
            .get_or_init(|| {
                let system_config = self.app_config.app_vm_config.as_ref();
                let root_params = self.root_params.clone();
                let agg_prover = self.agg_prover();

                let (trace_heights, root_pk) = compute_root_proof_heights(
                    system_config.clone(),
                    self.agg_config.params.clone(),
                    self.agg_tree_config,
                    root_params.clone(),
                    self.deferral_setup.clone(),
                )
                .expect("Trace heights did not generate properly");

                let memory_dimensions = system_config.memory_config.memory_dimensions();
                let num_user_pvs = system_config.num_public_values;

                Arc::new(RootProver::from_pk(
                    agg_prover.internal_recursive_prover.get_vk(),
                    agg_prover
                        .internal_recursive_prover
                        .get_self_vk_pcs_data()
                        .unwrap()
                        .commitment
                        .into(),
                    root_pk,
                    memory_dimensions,
                    num_user_pvs,
                    self.def_hook_commit(),
                    Some(trace_heights),
                ))
            })
            .clone()
    }

    #[cfg(feature = "evm-prove")]
    /// Returns the cached Halo2 prover, generating it on first use if needed.
    pub fn halo2_prover(&self) -> Halo2Prover
    where
        <E::PD as ProverDevice<E::PB, E::TS>>::DeviceCtx: 'static,
    {
        self.halo2_prover
            .get_or_init(|| {
                use crate::keygen::static_verifier::keygen_halo2;

                let root_prover = self.root_prover();
                let root_vk = root_prover.0.get_vk().as_ref().clone();
                let agg_prover = self.agg_prover();

                // Generate a dummy root proof by running a trivial program through the pipeline
                let dummy_root_proof = keygen::dummy::generate_dummy_root_proof::<E, _>(
                    self.app_vm_builder.clone(),
                    &self.app_pk().app_vm_pk,
                    agg_prover.clone(),
                    self.deferral_setup.clone(),
                    root_prover,
                );

                let halo2_pk = keygen_halo2(
                    &self.halo2_config,
                    &self.halo2_params_reader,
                    self.halo2_shape,
                    &agg_prover.internal_recursive_prover.get_vk(),
                    &root_vk,
                    &dummy_root_proof,
                );

                Halo2Prover::new(self.halo2_params_reader(), halo2_pk)
            })
            .clone()
    }

    // ======================== Keygen Related Methods ========================

    /// Generates the app proving key once and caches it. Future calls will return the cached key.
    ///
    /// # Panics
    /// This function will panic if the app keygen fails.
    pub fn app_keygen(&self) -> (AppProvingKey<VB::VmConfig>, AppVerifyingKey) {
        let pk = self.app_pk().clone();
        let vk = pk.get_app_vk();
        (pk, vk)
    }

    /// Generates the app proving key once and caches it. Future calls will return the cached key.
    ///
    /// # Panics
    /// This function will panic if the app keygen fails.
    pub fn app_pk(&self) -> &AppProvingKey<VB::VmConfig> {
        // TODO[jpw]: use `get_or_try_init` once it is stable
        self.app_pk.get_or_init(|| {
            AppProvingKey::keygen(self.app_config.clone()).expect("app_keygen failed")
        })
    }

    /// Returns the app verifying key derived from the cached app proving key.
    pub fn app_vk(&self) -> AppVerifyingKey {
        self.app_pk().get_app_vk()
    }

    /// Generates or retrieves the aggregation proving and verifying keys as a pair.
    pub fn agg_keygen(&self) -> (AggProvingKey, MultiStarkVerifyingKey<SC>) {
        let pk = self.agg_pk();
        let vk = self.agg_vk().as_ref().clone();
        (pk, vk)
    }

    /// Generates or retrieves the aggregation prefix proving key without the internal-recursive
    /// key.
    pub fn agg_prefix_pk(&self) -> AggPrefixProvingKey {
        if let Some(agg_prover) = self.agg_prover.get() {
            return AggPrefixProvingKey {
                leaf: agg_prover.leaf_prover.get_pk(),
                internal_for_leaf: agg_prover.internal_for_leaf_prover.get_pk(),
            };
        }

        let app_pk = self.app_pk();
        AggProver::keygen_prefix(
            Arc::new(app_pk.app_vm_pk.vm_pk.get_vk()),
            self.agg_config.clone(),
            self.def_hook_cached_commit(),
        )
    }

    /// Generates or retrieves the full aggregation proving key.
    pub fn agg_pk(&self) -> AggProvingKey {
        let agg_prover = self.agg_prover();
        AggProvingKey {
            prefix: AggPrefixProvingKey {
                leaf: agg_prover.leaf_prover.get_pk(),
                internal_for_leaf: agg_prover.internal_for_leaf_prover.get_pk(),
            },
            internal_recursive: agg_prover.internal_recursive_prover.get_pk(),
        }
    }

    /// Returns the aggregation verifying key for the recursive aggregation layer.
    pub fn agg_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        self.agg_prover().internal_recursive_prover.get_vk()
    }

    #[cfg(feature = "root-prover")]
    /// Generates or retrieves the root proving key and recorded trace heights.
    pub fn root_pk(&self) -> RootProvingKey {
        let root_prover = self.root_prover();
        RootProvingKey {
            root_pk: root_prover.0.get_pk(),
            trace_heights: root_prover.0.get_trace_heights().unwrap_or_default(),
        }
    }

    /// Generates the Halo2 (static verifier + wrapper) proving key once and caches it.
    ///
    /// The flow:
    /// 1. Get the root VK and internal recursive VK cached commit
    /// 2. Generate a dummy root proof via the EVM prover pipeline
    /// 3. Keygen the static verifier circuit
    /// 4. Generate a dummy snark from the verifier
    /// 5. Keygen the wrapper circuit (auto-tuned or fixed k)
    #[cfg(feature = "evm-prove")]
    pub fn halo2_pk(&self) -> Halo2ProvingKey
    where
        <E::PD as ProverDevice<E::PB, E::TS>>::DeviceCtx: 'static,
    {
        self.halo2_prover().pk()
    }

    /// Generates the [`AppExecutionCommit`] for the given executable.
    ///
    /// This function will generate the app_pk if it does not already exist.
    pub fn app_commit(
        &self,
        app_exe: impl Into<ExecutableFormat>,
    ) -> Result<AppExecutionCommit, SdkError> {
        let prover = self.prover(app_exe)?;
        Ok(AppExecutionCommit {
            app_exe_commit: CommitBytes::from(prover.generate_baseline().app_exe_commit),
            app_vm_commit: CommitBytes::from(prover.app_vm_commit()),
        })
    }

    // ======================== Verification Methods ========================

    /// Verifies aggregate STARK proof of VM execution.
    ///
    /// **Note**: This function does not have any reliance on `self` and does not depend on the app
    /// config set in the [Sdk].
    pub fn verify_proof(
        agg_vk: MultiStarkVerifyingKey<SC>,
        baseline: VerificationBaseline,
        proof: &VmStarkProof,
    ) -> Result<(), SdkError> {
        let vk = VmStarkVerifyingKey {
            mvk: agg_vk,
            baseline,
        };
        verify_vm_stark_proof_decoded(&vk, proof)?;
        Ok(())
    }

    #[cfg(feature = "evm-verify")]
    /// Generates Solidity verifier artifacts for the cached Halo2 proving key.
    pub fn generate_halo2_verifier_solidity(&self) -> Result<types::EvmHalo2Verifier, SdkError>
    where
        <E::PD as ProverDevice<E::PB, E::TS>>::DeviceCtx: 'static,
    {
        solidity::generate_halo2_verifier_solidity(&self.halo2_pk(), &self.halo2_params_reader)
    }

    #[cfg(feature = "evm-verify")]
    /// Generates Solidity verifier artifacts under `src/{version_name}` in the solc source map.
    ///
    /// Solidity embeds source metadata in bytecode, so `version_name` should match the directory
    /// where the generated verifier contracts will be written.
    pub fn generate_halo2_verifier_solidity_with_version_name(
        &self,
        version_name: &str,
    ) -> Result<types::EvmHalo2Verifier, SdkError>
    where
        <E::PD as ProverDevice<E::PB, E::TS>>::DeviceCtx: 'static,
    {
        solidity::generate_halo2_verifier_solidity_with_version_name(
            &self.halo2_pk(),
            &self.halo2_params_reader,
            version_name,
        )
    }

    #[cfg(feature = "evm-verify")]
    /// Uses the `verify(..)` interface of the `OpenVmHalo2Verifier` contract.
    ///
    /// Requires the `evm-verify` feature. Internally deploys the verifier bytecode in a local EVM
    /// and executes the verification call. If expected_app_commit is provided, it will check the
    /// proof's app_commit against it.
    pub fn verify_evm_halo2_proof(
        openvm_verifier: &types::EvmHalo2Verifier,
        evm_proof: types::EvmProof,
        expected_app_commit: Option<AppExecutionCommit>,
    ) -> Result<u64, SdkError> {
        if let Some(expected_app_commit) = expected_app_commit {
            if expected_app_commit != evm_proof.app_commit {
                return Err(
                    eyre::eyre!("EVM proof verification failed: mismatching app commits").into(),
                );
            }
        }
        solidity::verify_evm_halo2_proof(openvm_verifier, evm_proof)
    }
}
