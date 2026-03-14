#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]

use std::{
    fs::read,
    marker::PhantomData,
    path::Path,
    sync::{Arc, OnceLock},
};

use config::AppConfig;
use getset::{Getters, MutGetters, WithSetters};
use keygen::{AppProvingKey, AppVerifyingKey};
use openvm_build::{
    build_guest_package, find_unique_executable, get_package, GuestOptions, TargetFilter,
};
// Re-exports
pub use openvm_build::{cargo_command, get_rustup_toolchain_name};
pub use openvm_circuit;
use openvm_circuit::{
    arch::{
        execution_mode::Segment, instructions::exe::VmExe, Executor, InitFileGenerator,
        MeteredExecutor, PreflightExecutor, VirtualMachineError, VmBuilder, VmExecutionConfig,
        VmExecutor,
    },
    system::memory::merkle::public_values::extract_public_values,
};
use openvm_sdk_config::{SdkVmConfig, SdkVmCpuBuilder, TranspilerConfig};
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, StarkEngine, SystemParams};
use openvm_stark_sdk::config::{
    baby_bear_poseidon2::BabyBearPoseidon2CpuEngine as BabyBearPoseidon2Engine,
    root_params_with_100_bits_security,
};
use openvm_transpiler::{
    elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler, FromElf,
};
use openvm_verify_stark_host::{
    verify_vm_stark_proof_decoded,
    vk::{NonRootStarkVerifyingKey, VerificationBaseline},
    NonRootStarkProof,
};

use crate::{
    config::{AggregationConfig, AggregationSystemParams, AggregationTreeConfig},
    keygen::AggProvingKey,
    prover::{
        compute_root_proof_heights, AggProver, AppProver, DeferralPathProver, DeferralProver,
        EvmProver, RootProver, StarkProver,
    },
    types::ExecutableFormat,
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

pub mod config;
pub mod fs;
pub mod keygen;
pub mod prover;
pub mod types;
pub mod util;

#[cfg(test)]
mod tests;

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
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E>,
    VB::VmConfig: VmExecutionConfig<F>,
{
    #[getset(get = "pub", get_mut = "pub", set_with = "pub")]
    app_config: AppConfig<VB::VmConfig>,
    #[getset(get = "pub", get_mut = "pub", set_with = "pub")]
    agg_config: AggregationConfig,
    #[getset(get = "pub", get_mut = "pub", set_with = "pub")]
    agg_tree_config: AggregationTreeConfig,
    #[cfg(feature = "evm-prove")]
    #[getset(get = "pub", get_mut = "pub", set_with = "pub")]
    halo2_config: Halo2Config,

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
    root_prover: OnceLock<Arc<RootProver>>,

    def_path_prover: Option<Arc<DeferralPathProver>>,

    #[cfg(feature = "evm-prove")]
    #[getset(get = "pub", get_mut = "pub", set_with = "pub")]
    halo2_params_reader: CacheHalo2ParamsReader,
    #[cfg(feature = "evm-prove")]
    halo2_pk: OnceLock<Halo2ProvingKey>,

    _phantom: PhantomData<E>,
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

    /// Creates SDK with a configuration with RISC-V RV32IM and IO VM extensions loaded.
    ///
    /// **Note**: To use this configuration, your `openvm.toml` must exactly match the following:
    ///
    /// ```toml
    #[doc = include_str!("../../sdk-config/src/openvm_riscv32.toml")]
    /// ```
    pub fn riscv32(app_params: SystemParams, agg_params: AggregationSystemParams) -> Self {
        GenericSdk::new(AppConfig::riscv32(app_params), agg_params).unwrap()
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
        let executor = VmExecutor::new(app_config.app_vm_config.clone())
            .map_err(|e| SdkError::Vm(e.into()))?;
        let agg_config = AggregationConfig { params: agg_params };
        Ok(Self {
            app_config,
            agg_config,
            agg_tree_config: Default::default(),
            app_vm_builder: Default::default(),
            transpiler: None,
            executor,
            app_pk: OnceLock::new(),
            agg_prover: OnceLock::new(),
            root_prover: OnceLock::new(),
            def_path_prover: None,
            _phantom: Default::default(),
        })
    }

    /// Enables deferrals in this GenericSdk. The DeferralProver must be created ahead of time
    /// because the DeferralExtension should be created using DeferralProver::make_extension, as
    /// it has the capability to generate def_vk_commits.
    pub fn with_deferral_prover(mut self, deferral_prover: DeferralProver) -> Self {
        assert!(
            self.def_path_prover.is_none(),
            "Deferral prover already defined"
        );

        let deferral_tree_config = AggregationTreeConfig {
            num_children_leaf: 2,
            num_children_internal: 2,
        };
        let agg_prover = AggProver::new(
            deferral_prover.def_hook_prover.get_vk(),
            self.agg_config.clone(),
            deferral_tree_config,
            Some(deferral_prover.def_hook_prover.get_cached_commit()),
        );
        let def_path_prover = DeferralPathProver {
            deferral_prover: Arc::new(deferral_prover),
            agg_prover: Arc::new(agg_prover),
        };

        self.def_path_prover = Some(Arc::new(def_path_prover));
        self
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
    E: StarkEngine<SC = SC>,
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

        let (ctx, final_state) = interpreter
            .execute_metered_cost(inputs, ctx)
            .map_err(VirtualMachineError::from)?;
        let instret = ctx.instret;
        let cost = ctx.cost;

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
    /// commitment to the App [VmExe] and aggregation verifiers. It does **not** depend on the
    /// `inputs`. It can be generated separately from the proof by creating a
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
        def_inputs: &[DeferralInput],
    ) -> Result<(NonRootStarkProof, VerificationBaseline), SdkError> {
        let mut prover = self.prover(app_exe)?;
        let proof = prover.prove(inputs, def_inputs)?;
        let baseline = prover.generate_baseline();
        Ok((proof, baseline))
    }

    #[cfg(feature = "evm-prove")]
    pub fn prove_evm(
        &self,
        app_exe: impl Into<ExecutableFormat>,
        inputs: StdIn,
    ) -> Result<EvmProof, SdkError> {
        let app_exe = self.convert_to_exe(app_exe)?;
        let mut evm_prover = self.evm_prover(app_exe)?;
        let proof = evm_prover.prove_evm(inputs)?;
        Ok(proof)
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
            self.def_path_prover.clone(),
        )?;
        Ok(stark_prover)
    }

    // #[cfg(feature = "evm-prove")]
    pub fn evm_prover(
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
            self.root_prover(),
        )?;
        Ok(evm_prover)
    }

    // ===================== Component Prover Constructors =====================

    pub fn agg_prover(&self) -> Arc<AggProver> {
        let app_pk = self.app_pk();
        self.agg_prover
            .get_or_init(|| {
                let def_hook_commit = self
                    .def_path_prover
                    .as_ref()
                    .map(|p| p.deferral_prover.def_hook_prover.get_cached_commit());
                Arc::new(AggProver::new(
                    Arc::new(app_pk.app_vm_pk.vm_pk.get_vk()),
                    self.agg_config.clone(),
                    self.agg_tree_config,
                    def_hook_commit,
                ))
            })
            .clone()
    }

    pub fn root_prover(&self) -> Arc<RootProver> {
        self.root_prover
            .get_or_init(|| {
                // TODO[INT-6073]: store root_params
                let system_config = self.app_config.app_vm_config.as_ref();
                let root_params = root_params_with_100_bits_security();

                let trace_heights = compute_root_proof_heights(
                    system_config.clone(),
                    self.agg_config.params.clone(),
                    self.agg_tree_config,
                    root_params.clone(),
                )
                .expect("Trace heights did not generate properly");

                let memory_dimensions = system_config.memory_config.memory_dimensions();
                let num_user_pvs = system_config.num_public_values;

                let agg_prover = self.agg_prover();
                Arc::new(RootProver::new(
                    agg_prover.internal_recursive_prover.get_vk(),
                    agg_prover
                        .internal_recursive_prover
                        .get_self_vk_pcs_data()
                        .unwrap()
                        .commitment
                        .into(),
                    root_params,
                    memory_dimensions,
                    num_user_pvs,
                    Some(trace_heights),
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

    pub fn agg_pk(&self) -> AggProvingKey {
        let agg_prover = self.agg_prover();
        AggProvingKey {
            leaf_pk: agg_prover.leaf_prover.get_pk(),
            internal_for_leaf_pk: agg_prover.internal_for_leaf_prover.get_pk(),
            internal_recursive_pk: agg_prover.internal_recursive_prover.get_pk(),
        }
    }

    pub fn agg_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        self.agg_prover().internal_recursive_prover.get_vk()
    }

    #[cfg(feature = "evm-prove")]
    pub fn halo2_keygen(&self) -> Halo2ProvingKey {
        self.halo2_pk().clone()
    }

    #[cfg(feature = "evm-prove")]
    pub fn halo2_pk(&self) -> &Halo2ProvingKey {
        let (agg_pk, dummy_internal_proof) = self.agg_pk_and_dummy_internal_proof();
        // TODO[jpw]: use `get_or_try_init` once it is stable
        self.halo2_pk.get_or_init(|| {
            Halo2ProvingKey::keygen(
                self.halo2_config,
                self.halo2_params_reader(),
                &DefaultStaticVerifierPvHandler,
                agg_pk,
                dummy_internal_proof.clone(),
            )
            .expect("halo2_keygen failed")
        })
    }
    /// Sets the halo2 proving keys. Returns `Ok(())` if halo2 keygen has not been called and
    /// `Err(halo2_pk)` if keygen has already been called.
    #[cfg(feature = "evm-prove")]
    pub fn set_halo2_pk(&self, halo2_pk: Halo2ProvingKey) -> Result<(), Halo2ProvingKey> {
        self.halo2_pk.set(halo2_pk)
    }
    /// See [`set_halo2_pk`](Self::set_halo2_pk). This should only be used in a constructor, and
    /// panics if halo2 keygen has already been called.
    #[cfg(feature = "evm-prove")]
    pub fn with_halo2_pk(self, halo2_pk: Halo2ProvingKey) -> Self {
        let _ = self
            .set_halo2_pk(halo2_pk)
            .map_err(|_| panic!("halo2_pk already set"));
        self
    }

    #[cfg(feature = "evm-prove")]
    pub fn with_halo2_params_dir(mut self, params_dir: impl AsRef<Path>) -> Self {
        self.set_halo2_params_dir(params_dir);
        self
    }
    #[cfg(feature = "evm-prove")]
    pub fn set_halo2_params_dir(&mut self, params_dir: impl AsRef<Path>) {
        self.halo2_params_reader = CacheHalo2ParamsReader::new(params_dir);
    }

    // ======================== Verification Methods ========================

    /// Verifies aggregate STARK proof of VM execution.
    ///
    /// **Note**: This function does not have any reliance on `self` and does not depend on the app
    /// config set in the [Sdk].
    pub fn verify_proof(
        agg_vk: MultiStarkVerifyingKey<SC>,
        baseline: VerificationBaseline,
        proof: &NonRootStarkProof,
    ) -> Result<(), SdkError> {
        let vk = NonRootStarkVerifyingKey {
            mvk: agg_vk,
            baseline,
        };
        verify_vm_stark_proof_decoded(&vk, proof)?;
        Ok(())
    }

    #[cfg(feature = "evm-verify")]
    pub fn generate_halo2_verifier_solidity(&self) -> Result<types::EvmHalo2Verifier, SdkError> {
        use std::{
            fs::{create_dir_all, write},
            io::Write,
            process::{Command, Stdio},
        };

        use eyre::Context;
        use forge_fmt::{
            format, FormatterConfig, IntTypes, MultilineFuncHeaderStyle, NumberUnderscore,
            QuoteStyle, SingleLineBlockStyle,
        };
        use openvm_native_recursion::halo2::wrapper::EvmVerifierByteCode;
        use serde_json::{json, Value};
        use snark_verifier::halo2_base::halo2_proofs::poly::commitment::Params;
        use snark_verifier_sdk::SHPLONK;
        use tempfile::tempdir;
        use types::EvmHalo2Verifier;

        use crate::fs::{
            EVM_HALO2_VERIFIER_BASE_NAME, EVM_HALO2_VERIFIER_INTERFACE_NAME,
            EVM_HALO2_VERIFIER_PARENT_NAME,
        };

        let reader = self.halo2_params_reader();
        let halo2_pk = self.halo2_pk();

        let params = reader.read_params(halo2_pk.wrapper.pinning.metadata.config_params.k);
        let pinning = &halo2_pk.wrapper.pinning;

        assert_eq!(
            pinning.metadata.config_params.k as u32,
            params.k(),
            "Provided params don't match circuit config"
        );

        let halo2_verifier_code = gen_evm_verifier_sol_code::<AggregationCircuit, SHPLONK>(
            &params,
            pinning.pk.get_vk(),
            pinning.metadata.num_pvs.clone(),
        );

        let wrapper_pvs = halo2_pk.wrapper.pinning.metadata.num_pvs.clone();
        let pvs_length = match wrapper_pvs.first() {
            // We subtract 14 to exclude the KZG accumulator and the app exe
            // and vm commits.
            Some(v) => v
                .checked_sub(14)
                .expect("Unexpected number of static verifier wrapper public values"),
            _ => panic!("Unexpected amount of instance columns in the static verifier wrapper"),
        };

        assert!(
            pvs_length <= 8192,
            "OpenVM Halo2 verifier contract does not support more than 8192 public values"
        );

        // Fill out the public values length and OpenVM version in the template
        let openvm_verifier_code = EVM_HALO2_VERIFIER_TEMPLATE
            .replace("{PUBLIC_VALUES_LENGTH}", &pvs_length.to_string())
            .replace("{OPENVM_VERSION}", OPENVM_VERSION);

        let formatter_config = FormatterConfig {
            line_length: 120,
            tab_width: 4,
            bracket_spacing: true,
            int_types: IntTypes::Long,
            multiline_func_header: MultilineFuncHeaderStyle::AttributesFirst,
            quote_style: QuoteStyle::Double,
            number_underscore: NumberUnderscore::Thousands,
            single_line_statement_blocks: SingleLineBlockStyle::Preserve,
            override_spacing: false,
            wrap_comments: false,
            ignore: vec![],
            contract_new_lines: false,
            sort_imports: false,
            ..Default::default()
        };

        let formatted_interface = format(EVM_HALO2_VERIFIER_INTERFACE, formatter_config.clone())
            .into_result()
            .expect("Failed to format interface");
        let formatted_halo2_verifier_code = format(&halo2_verifier_code, formatter_config.clone())
            .into_result()
            .expect("Failed to format halo2 verifier code");
        let formatted_openvm_verifier_code = format(&openvm_verifier_code, formatter_config)
            .into_result()
            .expect("Failed to format openvm verifier code");

        // Create temp dir
        let temp_dir = tempdir()
            .wrap_err("Failed to create temp dir")
            .map_err(SdkError::Other)?;
        let temp_path = temp_dir.path();
        let root_path = Path::new("src").join(format!("v{OPENVM_VERSION}"));

        // Make interfaces dir
        let interfaces_path = root_path.join("interfaces");

        // This will also create the dir for root_path, so no need to explicitly
        // create it
        create_dir_all(temp_path.join(&interfaces_path))?;

        let interface_file_path = interfaces_path.join(EVM_HALO2_VERIFIER_INTERFACE_NAME);
        let parent_file_path = root_path.join(EVM_HALO2_VERIFIER_PARENT_NAME);
        let base_file_path = root_path.join(EVM_HALO2_VERIFIER_BASE_NAME);

        // Write the files to the temp dir. This is only for compilation
        // purposes.
        write(temp_path.join(&interface_file_path), &formatted_interface)?;
        write(
            temp_path.join(&parent_file_path),
            &formatted_halo2_verifier_code,
        )?;
        write(
            temp_path.join(&base_file_path),
            &formatted_openvm_verifier_code,
        )?;

        // Run solc from the temp dir
        let solc_input = json!({
            "language": "Solidity",
            "sources": {
                interface_file_path.to_str().unwrap(): {
                    "content": formatted_interface
                },
                parent_file_path.to_str().unwrap(): {
                    "content": formatted_halo2_verifier_code
                },
                base_file_path.to_str().unwrap(): {
                    "content": formatted_openvm_verifier_code
                }
            },
            "settings": {
                "remappings": ["forge-std/=lib/forge-std/src/"],
                "optimizer": {
                    "enabled": true,
                    "runs": 100000,
                    "details": {
                        "constantOptimizer": false,
                        "yul": false
                    }
                },
                "evmVersion": "paris",
                "viaIR": false,
                "outputSelection": {
                    "*": {
                        "*": ["metadata", "evm.bytecode.object"]
                    }
                }
            }
        });

        let mut child = Command::new("solc")
            .current_dir(temp_path)
            .arg("--standard-json")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to spawn solc");

        child
            .stdin
            .as_mut()
            .expect("Failed to open stdin")
            .write_all(solc_input.to_string().as_bytes())
            .expect("Failed to write to stdin");

        let output = child.wait_with_output().expect("Failed to read output");

        if !output.status.success() {
            return Err(SdkError::Other(eyre::eyre!(
                "solc exited with status {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let parsed: Value =
            serde_json::from_slice(&output.stdout).map_err(|e| SdkError::Other(e.into()))?;

        let bytecode = parsed
            .get("contracts")
            .expect("No 'contracts' field found")
            .get(format!("src/v{OPENVM_VERSION}/OpenVmHalo2Verifier.sol"))
            .unwrap_or_else(|| {
                panic!("No 'src/v{OPENVM_VERSION}/OpenVmHalo2Verifier.sol' field found")
            })
            .get("OpenVmHalo2Verifier")
            .expect("No 'OpenVmHalo2Verifier' field found")
            .get("evm")
            .expect("No 'evm' field found")
            .get("bytecode")
            .expect("No 'bytecode' field found")
            .get("object")
            .expect("No 'object' field found")
            .as_str()
            .expect("No 'object' field found");

        let bytecode = hex::decode(bytecode).expect("Invalid hex in Binary");

        let evm_verifier = EvmHalo2Verifier {
            halo2_verifier_code: formatted_halo2_verifier_code,
            openvm_verifier_code: formatted_openvm_verifier_code,
            openvm_verifier_interface: formatted_interface,
            artifact: EvmVerifierByteCode {
                sol_compiler_version: "0.8.19".to_string(),
                sol_compiler_options: solc_input.get("settings").unwrap().to_string(),
                bytecode,
            },
        };
        Ok(evm_verifier)
    }

    #[cfg(feature = "evm-verify")]
    /// Uses the `verify(..)` interface of the `OpenVmHalo2Verifier` contract.
    pub fn verify_evm_halo2_proof(
        openvm_verifier: &types::EvmHalo2Verifier,
        evm_proof: EvmProof,
    ) -> Result<u64, SdkError> {
        let calldata = evm_proof.verifier_calldata();
        let deployment_code = openvm_verifier.artifact.bytecode.clone();

        let gas_cost = snark_verifier::loader::evm::deploy_and_call(deployment_code, calldata)
            .map_err(|reason| {
                SdkError::Other(eyre::eyre!("Sdk::verify_openvm_evm_proof: {reason:?}"))
            })?;

        Ok(gas_cost)
    }
}
