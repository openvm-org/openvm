use std::{
    borrow::Borrow,
    fs::read,
    marker::PhantomData,
    path::Path,
    sync::{Arc, OnceLock},
};

#[cfg(feature = "evm-verify")]
use alloy_sol_types::sol;
use commit::AppExecutionCommit;
use config::{AggregationTreeConfig, AppConfig};
use getset::{Getters, MutGetters, WithSetters};
use keygen::{AppProvingKey, AppVerifyingKey};
use openvm_build::{
    build_guest_package, find_unique_executable, get_package, GuestOptions, TargetFilter,
};
use openvm_circuit::{
    arch::{
        hasher::{poseidon2::vm_poseidon2_hasher, Hasher},
        instructions::exe::VmExe,
        Executor, InitFileGenerator, MeteredExecutor, PreflightExecutor, VirtualMachineError,
        VmBuilder, VmExecutionConfig, VmExecutor, VmVerificationError, CONNECTOR_AIR_ID,
        PROGRAM_AIR_ID, PROGRAM_CACHED_TRACE_INDEX, PUBLIC_VALUES_AIR_ID,
    },
    system::{
        memory::{
            merkle::public_values::{extract_public_values, UserPublicValuesProofError},
            CHUNK,
        },
        program::trace::{compute_exe_commit, VmCommittedExe},
    },
};
#[cfg(feature = "evm-prove")]
pub use openvm_continuations::static_verifier::DefaultStaticVerifierPvHandler;
use openvm_continuations::verifier::{
    common::types::VmVerifierPvs,
    internal::types::{InternalVmVerifierPvs, VmStarkProof},
    root::RootVmVerifierConfig,
};
// Re-exports:
pub use openvm_continuations::{RootSC, C, F, SC};
use openvm_native_circuit::{NativeConfig, NativeCpuBuilder};
#[cfg(feature = "evm-prove")]
use openvm_native_recursion::halo2::utils::{CacheHalo2ParamsReader, Halo2ParamsReader};
use openvm_stark_backend::proof::Proof;
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    engine::{StarkEngine, StarkFriEngine},
};
use openvm_transpiler::{
    elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler, FromElf,
};
#[cfg(feature = "evm-verify")]
use snark_verifier_sdk::{evm::gen_evm_verifier_sol_code, halo2::aggregation::AggregationCircuit};

use crate::{
    config::{AggregationConfig, Halo2Config, SdkVmCpuBuilder},
    keygen::{asm::program_to_asm, AggProvingKey, AggVerifyingKey},
    prover::{AppProver, StarkProver},
};
#[cfg(feature = "evm-prove")]
use crate::{keygen::Halo2ProvingKey, prover::EvmHalo2Prover, types::EvmProof};

pub mod codec;
pub mod commit;
pub mod config;
pub mod fs;
pub mod keygen;
pub mod prover;
pub mod types;

mod error;
mod stdin;
pub use error::SdkError;
pub use stdin::*;

pub const EVM_HALO2_VERIFIER_INTERFACE: &str =
    include_str!("../contracts/src/IOpenVmHalo2Verifier.sol");
pub const EVM_HALO2_VERIFIER_TEMPLATE: &str =
    include_str!("../contracts/template/OpenVmHalo2Verifier.sol");
pub const OPENVM_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION_MAJOR"),
    ".",
    env!("CARGO_PKG_VERSION_MINOR")
);

#[cfg(feature = "evm-verify")]
sol! {
    IOpenVmHalo2Verifier,
    concat!(env!("CARGO_MANIFEST_DIR"), "/contracts/abi/IOpenVmHalo2Verifier.json"),
}

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
pub struct GenericSdk<E, VB, NativeBuilder>
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

    /// The `executor` may be used to construct different types of interpreters, given the program,
    /// for more specific execution purposes. By default, it is recommended to use the
    /// [`execute`](GenericSdk::execute) method.
    #[getset(get = "pub")]
    executor: VmExecutor<F, VB::VmConfig>,

    #[getset(get_mut = "pub")]
    app_pk: OnceLock<AppProvingKey<VB::VmConfig>>,
    /// STARK aggregation proving key and dummy internal proof. Dummy internal proof is saved for
    /// halo2 pkey generation usage.
    #[getset(get_mut = "pub")]
    agg_pk_and_dummy_internal_proof: OnceLock<(AggProvingKey, Proof<SC>)>,

    #[cfg(feature = "evm-prove")]
    #[getset(get = "pub", get_mut = "pub", set_with = "pub")]
    halo2_params_reader: CacheHalo2ParamsReader,
    #[cfg(feature = "evm-prove")]
    #[getset(get_mut = "pub")]
    halo2_pk: OnceLock<Halo2ProvingKey>,

    #[getset(get = "pub")]
    app_vm_builder: VB,
    #[getset(get = "pub")]
    native_builder: NativeBuilder,
    #[getset(get = "pub", get_mut = "pub", set_with = "pub")]
    transpiler: Option<Transpiler<F>>,

    _phantom: PhantomData<E>,
}

pub type Sdk = GenericSdk<BabyBearPoseidon2Engine, SdkVmCpuBuilder, NativeCpuBuilder>;

impl Sdk {
    pub fn standard() -> Self {
        let app_config = AppConfig::standard();
        let transpiler = app_config.app_vm_config.transpiler();
        GenericSdk::new(app_config)
            .expect("standard config is valid")
            .with_transpiler(Some(transpiler))
    }

    pub fn riscv32() -> Self {
        let app_config = AppConfig::riscv32();
        let transpiler = app_config.app_vm_config.transpiler();
        GenericSdk::new(app_config)
            .expect("riscv32 config is valid")
            .with_transpiler(Some(transpiler))
    }
}

// The SDK is only functional for SC = BabyBearPoseidon2Config because that is what recursive
// aggregation supports.
impl<E, VB, NativeBuilder> GenericSdk<E, VB, NativeBuilder>
where
    E: StarkFriEngine<SC = SC>,
    VB: VmBuilder<E> + Clone,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor:
        Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, VB::RecordArena>,
    NativeBuilder: VmBuilder<E, VmConfig = NativeConfig> + Clone,
    <NativeConfig as VmExecutionConfig<F>>::Executor:
        PreflightExecutor<F, <NativeBuilder as VmBuilder<E>>::RecordArena>,
{
    /// Creates SDK custom to the given [AppConfig].
    ///
    /// **Note**: This function does not set the transpiler, which must be done separately to
    /// support RISC-V ELFs.
    pub fn new(app_config: AppConfig<VB::VmConfig>) -> Result<Self, SdkError>
    where
        VB: Default,
        NativeBuilder: Default,
    {
        let profiling = app_config.app_vm_config.as_ref().profiling;
        let executor = VmExecutor::new(app_config.app_vm_config.clone())
            .map_err(|e| SdkError::Vm(e.into()))?;
        let agg_config = AggregationConfig {
            profiling,
            ..Default::default()
        };
        let halo2_config = Halo2Config {
            profiling,
            ..Default::default()
        };
        Ok(Self {
            app_config,
            agg_config,
            #[cfg(feature = "evm-prove")]
            halo2_config,
            agg_tree_config: Default::default(),
            app_vm_builder: Default::default(),
            native_builder: Default::default(),
            transpiler: None,
            executor,
            app_pk: OnceLock::new(),
            agg_pk_and_dummy_internal_proof: OnceLock::new(),
            #[cfg(feature = "evm-prove")]
            halo2_params_reader: CacheHalo2ParamsReader::new_with_default_params_dir(),
            #[cfg(feature = "evm-prove")]
            halo2_pk: OnceLock::new(),
            _phantom: PhantomData,
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

    /// Transpiles RISC-V ELF to OpenVM executable.
    pub fn transpile(&self, elf: Elf) -> Result<VmExe<F>, SdkError> {
        let transpiler = self
            .transpiler
            .clone()
            .ok_or(SdkError::TranspilerNotAvailable)?;
        let exe = VmExe::from_elf(elf, transpiler)?;
        Ok(exe)
    }

    /// Returns the user public values as field elements.
    pub fn execute(&self, exe: VmExe<F>, inputs: StdIn) -> Result<Vec<u8>, SdkError> {
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

    // ======================== Proving Methods ============================

    /// Generates a single aggregate STARK proof of the full program execution of the given
    /// `app_exe` with program inputs `inputs`.
    ///
    /// The returned STARK proof is not intended for EVM verification. For EVM verification, use the
    /// [`prove_evm`](Self::prove_evm) method, which requires the `"evm-prove"` feature to be
    /// enabled.
    pub fn prove(
        &self,
        app_committed_exe: Arc<VmCommittedExe<SC>>,
        inputs: StdIn,
    ) -> Result<VmStarkProof<SC>, SdkError> {
        let mut prover = self.prover(app_committed_exe)?;
        let proof = prover.prove(inputs)?;
        Ok(proof)
    }

    #[cfg(feature = "evm-prove")]
    pub fn prove_evm(
        &self,
        app_committed_exe: Arc<VmCommittedExe<SC>>,
        inputs: StdIn,
    ) -> Result<EvmProof, SdkError> {
        let mut evm_prover = self.evm_prover(app_committed_exe)?;
        let proof = evm_prover.prove_evm(inputs)?;
        Ok(proof)
    }

    // ========================= Prover Constructors =========================

    /// Constructs a new [StarkProver] instance for the given executable.
    /// This function will generate the [AppProvingKey] and [AggProvingKey] if they do not already
    /// exist.
    pub fn prover(
        &self,
        app_committed_exe: Arc<VmCommittedExe<SC>>,
    ) -> Result<StarkProver<E, VB, NativeBuilder>, SdkError> {
        let app_pk = self.app_pk().clone();
        let agg_pk = self.agg_pk().clone();
        let stark_prover = StarkProver::<E, _, _>::new(
            self.app_vm_builder.clone(),
            self.native_builder.clone(),
            app_pk,
            app_committed_exe,
            agg_pk,
            self.agg_tree_config,
        )?;
        Ok(stark_prover)
    }

    #[cfg(feature = "evm-prove")]
    pub fn evm_prover(
        &self,
        app_exe: Arc<VmCommittedExe<SC>>,
    ) -> Result<EvmHalo2Prover<E, VB, NativeBuilder>, SdkError> {
        let evm_prover = EvmHalo2Prover::<E, _, _>::new(
            self.halo2_params_reader(),
            self.app_vm_builder.clone(),
            self.native_builder.clone(),
            self.app_pk().clone(),
            app_exe,
            self.agg_pk().clone(),
            self.halo2_pk().clone(),
            self.agg_tree_config,
        )?;
        Ok(evm_prover)
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
        app_committed_exe: Arc<VmCommittedExe<SC>>,
    ) -> Result<AppProver<E, VB>, SdkError> {
        let vm_pk = self.app_pk().app_vm_pk.clone();
        let prover =
            AppProver::<E, VB>::new(self.app_vm_builder.clone(), vm_pk, app_committed_exe)?;
        Ok(prover)
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

    /// Generates the proving keys necessary for STARK aggregation. Generates the proving keys once
    /// and caches them. Future calls will return the cached key. This function does not include
    /// [`app_keygen`](Self::app_keygen), which is specific to the App VM config. The proving keys
    /// generated in this step are independent of the App VM config.
    ///
    /// # Panics
    /// This function will panic if the keygen fails.
    pub fn agg_keygen(&self) -> Result<(AggProvingKey, AggVerifyingKey), SdkError> {
        let agg_pk = self.agg_pk().clone();
        let agg_vk = agg_pk.get_agg_vk();
        Ok((agg_pk, agg_vk))
    }

    pub fn agg_pk(&self) -> &AggProvingKey {
        let (agg_pk, _) = self.agg_pk_and_dummy_internal_proof();
        agg_pk
    }
    fn agg_pk_and_dummy_internal_proof(&self) -> &(AggProvingKey, Proof<SC>) {
        // TODO[jpw]: use `get_or_try_init` once it is stable
        self.agg_pk_and_dummy_internal_proof.get_or_init(|| {
            AggProvingKey::dummy_proof_and_keygen(self.agg_config).expect("agg_keygen failed")
        })
    }

    pub fn generate_root_verifier_asm(&self, agg_stark_pk: &AggProvingKey) -> String {
        let kernel_asm = RootVmVerifierConfig {
            leaf_fri_params: agg_stark_pk.leaf_vm_pk.fri_params,
            internal_fri_params: agg_stark_pk.internal_vm_pk.fri_params,
            num_user_public_values: agg_stark_pk.num_user_public_values(),
            internal_vm_verifier_commit: agg_stark_pk
                .internal_committed_exe
                .get_program_commit()
                .into(),
            compiler_options: Default::default(),
        }
        .build_kernel_asm(
            &agg_stark_pk.leaf_vm_pk.vm_pk.get_vk(),
            &agg_stark_pk.internal_vm_pk.vm_pk.get_vk(),
        );
        program_to_asm(kernel_asm)
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

    // ======================== Verification Methods ========================

    /// Verifies aggregate STARK proof of VM execution.
    ///
    /// **Note**: This function does not have any reliance on `self` and does not depend on the app
    /// config set in the [Sdk].
    pub fn verify_proof(
        agg_vk: &AggVerifyingKey,
        expected_app_commit: AppExecutionCommit,
        proof: &VmStarkProof<SC>,
    ) -> Result<(), SdkError> {
        if proof.inner.per_air.len() < 3 {
            return Err(VmVerificationError::NotEnoughAirs(proof.inner.per_air.len()).into());
        } else if proof.inner.per_air[0].air_id != PROGRAM_AIR_ID {
            return Err(VmVerificationError::SystemAirMissing {
                air_id: PROGRAM_AIR_ID,
            }
            .into());
        } else if proof.inner.per_air[1].air_id != CONNECTOR_AIR_ID {
            return Err(VmVerificationError::SystemAirMissing {
                air_id: CONNECTOR_AIR_ID,
            }
            .into());
        } else if proof.inner.per_air[2].air_id != PUBLIC_VALUES_AIR_ID {
            return Err(VmVerificationError::SystemAirMissing {
                air_id: PUBLIC_VALUES_AIR_ID,
            }
            .into());
        }
        let public_values_air_proof_data = &proof.inner.per_air[2];

        let program_commit =
            proof.inner.commitments.main_trace[PROGRAM_CACHED_TRACE_INDEX].as_ref();
        let internal_commit: &[_; CHUNK] = &agg_vk.internal_verifier_program_commit.into();

        let (fri_params_final, vk_final, claimed_app_vm_commit) =
            if program_commit == internal_commit {
                let internal_pvs: &InternalVmVerifierPvs<_> = public_values_air_proof_data
                    .public_values
                    .as_slice()
                    .borrow();
                if internal_commit != &internal_pvs.extra_pvs.internal_program_commit {
                    tracing::debug!(
                        "Invalid internal program commit: expected {:?}, got {:?}",
                        internal_commit,
                        internal_pvs.extra_pvs.internal_program_commit
                    );
                    return Err(VmVerificationError::ProgramCommitMismatch { index: 0 }.into());
                }
                (
                    agg_vk.internal_fri_params,
                    &agg_vk.internal_vk,
                    internal_pvs.extra_pvs.leaf_verifier_commit,
                )
            } else {
                (agg_vk.leaf_fri_params, &agg_vk.leaf_vk, *program_commit)
            };
        let e = E::new(fri_params_final);
        e.verify(vk_final, &proof.inner)
            .map_err(VmVerificationError::from)?;

        let pvs: &VmVerifierPvs<_> =
            public_values_air_proof_data.public_values[..VmVerifierPvs::<u8>::width()].borrow();

        if let Some(exit_code) = pvs.connector.exit_code() {
            if exit_code != 0 {
                return Err(VmVerificationError::ExitCodeMismatch {
                    expected: 0,
                    actual: exit_code,
                }
                .into());
            }
        } else {
            return Err(VmVerificationError::IsTerminateMismatch {
                expected: true,
                actual: false,
            }
            .into());
        }

        let hasher = vm_poseidon2_hasher();
        let public_values_root = hasher.merkle_root(&proof.user_public_values);
        if public_values_root != pvs.public_values_commit {
            tracing::debug!(
                "Invalid public values root: expected {:?}, got {:?}",
                pvs.public_values_commit,
                public_values_root
            );
            return Err(VmVerificationError::UserPublicValuesError(
                UserPublicValuesProofError::UserPublicValuesCommitMismatch,
            )
            .into());
        }

        let claimed_app_exe_commit = compute_exe_commit(
            &hasher,
            &pvs.app_commit,
            &pvs.memory.initial_root,
            pvs.connector.initial_pc,
        );
        let claimed_app_commit =
            AppExecutionCommit::from_field_commit(claimed_app_exe_commit, claimed_app_vm_commit);
        let exe_commit_bn254 = claimed_app_commit.app_exe_commit.to_bn254();
        let vm_commit_bn254 = claimed_app_commit.app_vm_commit.to_bn254();

        if exe_commit_bn254 != expected_app_commit.app_exe_commit.to_bn254() {
            return Err(SdkError::InvalidAppExeCommit {
                expected: expected_app_commit.app_exe_commit,
                actual: claimed_app_commit.app_exe_commit,
            });
        } else if vm_commit_bn254 != expected_app_commit.app_vm_commit.to_bn254() {
            return Err(SdkError::InvalidAppVmCommit {
                expected: expected_app_commit.app_vm_commit,
                actual: claimed_app_commit.app_vm_commit,
            });
        }
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
            format, parse, FormatterConfig, IntTypes, MultilineFuncHeaderStyle, NumberUnderscore,
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
        };

        let parsed_interface =
            parse(EVM_HALO2_VERIFIER_INTERFACE).expect("Failed to parse interface");
        let parsed_halo2_verifier_code =
            parse(&halo2_verifier_code).expect("Failed to parse halo2 verifier code");
        let parsed_openvm_verifier_code =
            parse(&openvm_verifier_code).expect("Failed to parse openvm verifier code");

        let mut formatted_interface = String::new();
        let mut formatted_halo2_verifier_code = String::new();
        let mut formatted_openvm_verifier_code = String::new();

        format(
            &mut formatted_interface,
            parsed_interface,
            formatter_config.clone(),
        )
        .expect("Failed to format interface");
        format(
            &mut formatted_halo2_verifier_code,
            parsed_halo2_verifier_code,
            formatter_config.clone(),
        )
        .expect("Failed to format halo2 verifier code");
        format(
            &mut formatted_openvm_verifier_code,
            parsed_openvm_verifier_code,
            formatter_config,
        )
        .expect("Failed to format openvm verifier code");

        // Create temp dir
        let temp_dir = tempdir()
            .wrap_err("Failed to create temp dir")
            .map_err(SdkError::Other)?;
        let temp_path = temp_dir.path();
        let root_path = Path::new("src").join(format!("v{}", OPENVM_VERSION));

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
            .get(format!("src/v{}/OpenVmHalo2Verifier.sol", OPENVM_VERSION))
            .unwrap_or_else(|| {
                panic!(
                    "No 'src/v{}/OpenVmHalo2Verifier.sol' field found",
                    OPENVM_VERSION
                )
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
