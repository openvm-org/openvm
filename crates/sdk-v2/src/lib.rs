#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
use std::{
    fs::read,
    marker::PhantomData,
    path::Path,
    sync::{Arc, OnceLock},
};

use config::{AggregationTreeConfig, AppConfig};
use continuations_v2::aggregation::{AggregationProver, NonRootStarkProof};
use getset::{Getters, MutGetters, WithSetters};
use keygen::{AppProvingKey, AppVerifyingKey};
use openvm_build::{
    GuestOptions, TargetFilter, build_guest_package, find_unique_executable, get_package,
};
use openvm_circuit::{
    arch::{
        Executor, InitFileGenerator, MeteredExecutor, PreflightExecutor, VirtualMachineError,
        VmBuilder, VmExecutionConfig, VmExecutor, execution_mode::Segment,
        instructions::exe::VmExe,
    },
    system::memory::merkle::public_values::extract_public_values,
};
use openvm_transpiler::{
    FromElf, elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler,
};
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2 as BabyBearPoseidon2Engine, StarkWhirEngine, SystemParams,
    keygen::types::MultiStarkVerifyingKeyV2 as MultiStarkVerifyingKey,
};
use verify_stark::{VerificationBaseline, verify_vm_stark_proof};

use crate::{
    config::{
        AggregationConfig, AggregationSystemParams, SdkVmConfig, SdkVmCpuBuilder, TranspilerConfig,
    },
    prover::{AggProver, AppProver, StarkProver},
    types::ExecutableFormat,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use config::SdkVmGpuBuilder;
        use cuda_backend_v2::BabyBearPoseidon2GpuEngineV2 as GpuBabyBearPoseidon2Engine;
        pub use GpuSdk as Sdk;
        pub type DefaultStarkEngine = GpuBabyBearPoseidon2Engine;
    } else {
        pub use CpuSdk as Sdk;
        pub type DefaultStarkEngine = BabyBearPoseidon2Engine;
    }
}

pub use stark_backend_v2::{F, SC};

pub mod config;
pub mod fs;
pub mod keygen;
pub mod prover;
pub mod types;
pub mod util;

mod error;
mod stdin;
pub use error::SdkError;
pub use stdin::*;

pub const OPENVM_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION_MAJOR"),
    ".",
    env!("CARGO_PKG_VERSION_MINOR")
);

// The SDK is only generic in the engine for the non-root SC. The root SC is fixed to
// BabyBearPoseidon2RootEngine right now.
/// The SDK provides convenience methods and constructors for provers.
///
/// The SDK is stateful to cache results of computations that depend only on the App VM config and
/// aggregation config. The SDK will not cache any state that depends on the program executable.
///
/// Some commonly used methods are:
/// - [`execute`](Self::execute)
/// - [`prove`](Self::prove)
/// - [`verify_proof`](Self::verify_proof)
#[derive(Getters, MutGetters, WithSetters)]
pub struct GenericSdk<E, VB>
where
    E: StarkWhirEngine<SC = SC>,
    VB: VmBuilder<E>,
    VB::VmConfig: VmExecutionConfig<F>,
{
    #[getset(get = "pub", get_mut = "pub", set_with = "pub")]
    app_config: AppConfig<VB::VmConfig>,
    #[getset(get = "pub", get_mut = "pub", set_with = "pub")]
    agg_config: AggregationConfig,
    #[getset(get = "pub", get_mut = "pub", set_with = "pub")]
    agg_tree_config: AggregationTreeConfig,

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

    _phantom: PhantomData<E>,
}

pub type CpuSdk = GenericSdk<BabyBearPoseidon2Engine, SdkVmCpuBuilder>;

#[cfg(feature = "cuda")]
pub type GpuSdk = GenericSdk<GpuBabyBearPoseidon2Engine, SdkVmGpuBuilder>;

impl<E, VB> GenericSdk<E, VB>
where
    E: StarkWhirEngine<SC = SC>,
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
    #[doc = include_str!("./config/openvm_standard.toml")]
    /// ```
    pub fn standard(app_params: SystemParams, agg_params: AggregationSystemParams) -> Self {
        GenericSdk::new(AppConfig::standard(app_params), agg_params).unwrap()
    }

    /// Creates SDK with a configuration with RISC-V RV32IM and IO VM extensions loaded.
    ///
    /// **Note**: To use this configuration, your `openvm.toml` must exactly match the following:
    ///
    /// ```toml
    #[doc = include_str!("./config/openvm_riscv32.toml")]
    /// ```
    pub fn riscv32(app_params: SystemParams, agg_params: AggregationSystemParams) -> Self {
        GenericSdk::new(AppConfig::riscv32(app_params), agg_params).unwrap()
    }
}

impl<E, VB> GenericSdk<E, VB>
where
    E: StarkWhirEngine<SC = SC>,
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
        let transpiler = app_config.app_vm_config.transpiler();
        let sdk = Self::new_without_transpiler(app_config, agg_params)?.with_transpiler(transpiler);
        Ok(sdk)
    }

    /// **Note**: This function does not set the transpiler, which must be done separately to
    /// support RISC-V ELFs.
    pub fn new_without_transpiler(
        app_config: AppConfig<VB::VmConfig>,
        agg_params: AggregationSystemParams,
    ) -> Result<Self, SdkError>
    where
        VB: Default,
    {
        let system_config = app_config.app_vm_config.as_ref();
        let executor = VmExecutor::new(app_config.app_vm_config.clone())
            .map_err(|e| SdkError::Vm(e.into()))?;
        let agg_config = AggregationConfig {
            max_num_user_public_values: system_config.num_public_values,
            params: agg_params,
        };
        Ok(Self {
            app_config,
            agg_config,
            agg_tree_config: Default::default(),
            app_vm_builder: Default::default(),
            transpiler: None,
            executor,
            app_pk: OnceLock::new(),
            agg_prover: OnceLock::new(),
            _phantom: Default::default(),
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
    pub fn set_transpiler(&mut self, transpiler: Transpiler<F>) {
        self.transpiler = Some(transpiler);
    }
    pub fn with_transpiler(mut self, transpiler: Transpiler<F>) -> Self {
        self.set_transpiler(transpiler);
        self
    }

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
}

// The SDK is only functional for SC = BabyBearPoseidon2Config because that is what recursive
// aggregation supports.
impl<E, VB> GenericSdk<E, VB>
where
    E: StarkWhirEngine<SC = SC>,
    VB: VmBuilder<E> + Clone,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor:
        Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, VB::RecordArena>,
{
    /// Returns the user public values as field elements.
    pub fn execute(
        &self,
        app_exe: impl Into<ExecutableFormat>,
        inputs: StdIn,
    ) -> Result<Vec<u8>, SdkError> {
        let exe = self.convert_to_exe(app_exe)?;
        let instance = self
            .executor
            .instance(&exe)
            .map_err(VirtualMachineError::from)?;
        let final_memory = instance
            .execute(inputs, None)
            .map_err(VirtualMachineError::from)?
            .memory;
        let public_values = extract_public_values(
            self.executor.config.as_ref().num_public_values,
            &final_memory.memory,
        );
        Ok(public_values)
    }

    /// Executes with segmentation for proof generation.
    /// Returns both user public values and segments with instruction counts and trace heights.
    pub fn execute_metered(
        &self,
        app_exe: impl Into<ExecutableFormat>,
        inputs: StdIn,
    ) -> Result<(Vec<u8>, Vec<Segment>), SdkError> {
        let app_prover = self.app_prover(app_exe)?;

        let vm = app_prover.vm();
        let exe = app_prover.exe();

        let ctx = vm.build_metered_ctx(&exe);
        let interpreter = vm
            .metered_interpreter(&exe)
            .map_err(VirtualMachineError::from)?;

        let (segments, final_state) = interpreter
            .execute_metered(inputs, ctx)
            .map_err(VirtualMachineError::from)?;
        let public_values = extract_public_values(
            self.executor.config.as_ref().num_public_values,
            &final_state.memory.memory,
        );

        Ok((public_values, segments))
    }

    /// Executes with cost metering to measure computational cost in trace cells.
    /// Returns both user public values, and cost along with instruction count.
    pub fn execute_metered_cost(
        &self,
        app_exe: impl Into<ExecutableFormat>,
        inputs: StdIn,
    ) -> Result<(Vec<u8>, (u64, u64)), SdkError> {
        let app_prover = self.app_prover(app_exe)?;

        let vm = app_prover.vm();
        let exe = app_prover.exe();

        let ctx = vm.build_metered_cost_ctx();
        let interpreter = vm
            .metered_cost_interpreter(&exe)
            .map_err(VirtualMachineError::from)?;

        let (cost, final_state) = interpreter
            .execute_metered_cost(inputs, ctx)
            .map_err(VirtualMachineError::from)?;
        let instret = final_state.instret();

        let public_values = extract_public_values(
            self.executor.config.as_ref().num_public_values,
            &final_state.memory.memory,
        );

        Ok((public_values, (cost, instret)))
    }

    // ======================== Proving Methods ============================

    /// Generates a single aggregate STARK proof of the full program execution of the given
    /// `app_exe` with program inputs `inputs`.\
    ///
    /// For convenience, this function also returns the [VerificationBaseline], which is a full
    /// commitment to the App [VmExe] and aggregation verifiers. It does **not** depend on the `inputs`.
    /// It can be generated separately from the proof by creating a
    /// [`prover`](Self::prover) and calling
    /// [`app_commit`](StarkProver::app_commit).
    ///
    /// If STARK aggregation is not needed and a proof whose size may grow linearly with the length
    /// of the program runtime is desired, create an [`app_prover`](Self::app_prover) and call
    /// [`app_prover.prove(inputs)`](AppProver::prove).
    pub fn prove(
        &self,
        app_exe: impl Into<ExecutableFormat>,
        inputs: StdIn,
    ) -> Result<(NonRootStarkProof, VerificationBaseline), SdkError> {
        let mut prover = self.prover(app_exe)?;
        let proof = prover.prove(inputs)?;
        let baseline = prover.generate_baseline();
        Ok((proof, baseline))
    }

    // ========================= Prover Constructors =========================

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
        )?;
        Ok(stark_prover)
    }

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

    pub fn agg_prover(&self) -> Arc<AggProver> {
        let app_pk = self.app_pk();
        self.agg_prover
            .get_or_init(|| {
                Arc::new(AggProver::new(
                    Arc::new(app_pk.app_vm_pk.vm_pk.get_vk()),
                    self.agg_config,
                    self.agg_tree_config,
                ))
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
    /// Sets the app proving key. Returns `Ok(())` if app keygen has not been called and
    /// `Err(app_pk)` if keygen has already been called.
    pub fn set_app_pk(
        &self,
        app_pk: AppProvingKey<VB::VmConfig>,
    ) -> Result<(), AppProvingKey<VB::VmConfig>> {
        self.app_pk.set(app_pk)
    }

    /// See [`set_app_pk`](Self::set_app_pk). This should only be used in a constructor, and panics
    /// if app keygen has already been called.
    pub fn with_app_pk(self, app_pk: AppProvingKey<VB::VmConfig>) -> Self {
        let _ = self
            .set_app_pk(app_pk)
            .map_err(|_| panic!("app_pk already set"));
        self
    }

    pub fn agg_vk(&self) -> Arc<MultiStarkVerifyingKey> {
        let agg_prover = self.agg_prover();
        agg_prover.internal_recursive_prover.get_vk()
    }

    // ======================== Verification Methods ========================

    /// Verifies aggregate STARK proof of VM execution.
    ///
    /// **Note**: This function does not have any reliance on `self` and does not depend on the app
    /// config set in the [Sdk].
    pub fn verify_proof(
        agg_vk: &MultiStarkVerifyingKey,
        baseline: VerificationBaseline,
        proof: &NonRootStarkProof,
    ) -> Result<(), SdkError> {
        verify_vm_stark_proof(agg_vk, baseline, proof)?;
        Ok(())
    }
}
