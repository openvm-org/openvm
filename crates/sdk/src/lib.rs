use std::{
    borrow::Borrow,
    fs::read,
    marker::PhantomData,
    path::Path,
    sync::{Arc, OnceLock},
};

#[cfg(feature = "evm-verify")]
use alloy_sol_types::sol;
use commit::{commit_app_exe, AppExecutionCommit};
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
        verify_segments, ContinuationVmProof, Executor, InitFileGenerator, MeteredExecutor,
        PreflightExecutor, SystemConfig, VerifiedExecutionPayload, VirtualMachineError, VmBuilder,
        VmCircuitConfig, VmExecutionConfig, VmExecutor, VmVerificationError, CONNECTOR_AIR_ID,
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
pub use openvm_continuations::static_verifier::{
    DefaultStaticVerifierPvHandler, StaticVerifierPvHandler,
};
use openvm_continuations::verifier::{
    common::types::VmVerifierPvs,
    internal::types::{InternalVmVerifierPvs, VmStarkProof},
    root::RootVmVerifierConfig,
};
// Re-exports:
pub use openvm_continuations::{RootSC, C, F, SC};
use openvm_native_circuit::{NativeConfig, NativeCpuBuilder};
#[cfg(feature = "evm-prove")]
use openvm_native_recursion::halo2::utils::Halo2ParamsReader;
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    engine::{StarkEngine, StarkFriEngine},
    p3_bn254_fr::Bn254Fr,
};
use openvm_transpiler::{
    elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler, FromElf,
};
#[cfg(feature = "evm-verify")]
use snark_verifier_sdk::{evm::gen_evm_verifier_sol_code, halo2::aggregation::AggregationCircuit};

use crate::{
    commit::CommitBytes,
    config::{AggStarkConfig, SdkVmCpuBuilder},
    keygen::{asm::program_to_asm, AggStarkProvingKey},
    prover::{AppProver, StarkProver, VerifiedAppArtifacts},
};
#[cfg(feature = "evm-prove")]
use crate::{config::AggConfig, keygen::AggProvingKey, prover::EvmHalo2Prover, types::EvmProof};

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
#[derive(Getters, MutGetters, WithSetters)]
pub struct GenericSdk<E, VB, NativeBuilder>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E>,
    VB::VmConfig: VmExecutionConfig<F>,
{
    #[getset(get = "pub", set_with = "pub")]
    agg_tree_config: AggregationTreeConfig,
    #[getset(get = "pub", get_mut = "pub")]
    app_config: AppConfig<VB::VmConfig>,

    /// The `executor` may be used to construct different types of interpreters, given the program,
    /// for more specific execution purposes. By default, it is recommended to use the
    /// [`execute`](GenericSdk::execute) method.
    #[getset(get = "pub")]
    executor: VmExecutor<F, VB::VmConfig>,

    app_pk: OnceLock<AppProvingKey<VB::VmConfig>>,

    pub(crate) app_vm_builder: VB,
    pub(crate) native_builder: NativeBuilder,
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
        let executor = VmExecutor::new(app_config.app_vm_config.clone())
            .map_err(|e| SdkError::Vm(e.into()))?;
        Ok(Self {
            app_config,
            app_vm_builder: Default::default(),
            native_builder: Default::default(),
            agg_tree_config: Default::default(),
            transpiler: None,
            executor,
            app_pk: OnceLock::new(),
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
    pub fn execute(&self, exe: VmExe<F>, inputs: StdIn) -> Result<Vec<F>, SdkError> {
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

    // TODO: switch committed_exe to just exe
    pub fn app_prover(
        &self,
        app_committed_exe: Arc<VmCommittedExe<SC>>,
    ) -> Result<AppProver<E, VB>, SdkError> {
        let vm_pk = self.app_pk().app_vm_pk.clone();
        let prover =
            AppProver::<E, VB>::new(self.app_vm_builder.clone(), vm_pk, app_committed_exe)?;
        Ok(prover)
    }

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
        self.app_pk.get_or_init(|| {
            AppProvingKey::keygen(self.app_config.clone()).expect("app_keygen failed")
        })
    }

    #[cfg(feature = "evm-prove")]
    pub fn agg_keygen(
        &self,
        config: AggConfig,
        reader: &impl Halo2ParamsReader,
        pv_handler: &impl StaticVerifierPvHandler,
    ) -> Result<AggProvingKey, SdkError> {
        let agg_pk = AggProvingKey::keygen(config, reader, pv_handler)?;
        Ok(agg_pk)
    }

    pub fn agg_stark_keygen(&self, config: AggStarkConfig) -> Result<AggStarkProvingKey, SdkError> {
        let agg_pk = AggStarkProvingKey::keygen(config)?;
        Ok(agg_pk)
    }

    pub fn generate_root_verifier_asm(&self, agg_stark_pk: &AggStarkProvingKey) -> String {
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

    pub fn generate_e2e_stark_proof(
        &self,
        app_exe: Arc<VmCommittedExe<SC>>,
        agg_stark_pk: AggStarkProvingKey,
        inputs: StdIn,
    ) -> Result<VmStarkProof<SC>, SdkError> {
        let (app_pk, _) = self.app_keygen();
        let mut stark_prover = StarkProver::<E, _, _>::new(
            self.app_vm_builder.clone(),
            self.native_builder.clone(),
            app_pk,
            app_exe,
            agg_stark_pk,
            self.agg_tree_config,
        )?;
        let proof = stark_prover.prove(inputs)?;
        Ok(proof)
    }

    pub fn verify_stark_proof(
        &self,
        agg_stark_pk: &AggStarkProvingKey,
        expected_app_commit: AppExecutionCommit,
        proof: &VmStarkProof<SC>,
    ) -> Result<(), SdkError> {
        if proof.proof.per_air.len() < 3 {
            return Err(VmVerificationError::NotEnoughAirs(proof.proof.per_air.len()).into());
        } else if proof.proof.per_air[0].air_id != PROGRAM_AIR_ID {
            return Err(VmVerificationError::SystemAirMissing {
                air_id: PROGRAM_AIR_ID,
            }
            .into());
        } else if proof.proof.per_air[1].air_id != CONNECTOR_AIR_ID {
            return Err(VmVerificationError::SystemAirMissing {
                air_id: CONNECTOR_AIR_ID,
            }
            .into());
        } else if proof.proof.per_air[2].air_id != PUBLIC_VALUES_AIR_ID {
            return Err(VmVerificationError::SystemAirMissing {
                air_id: PUBLIC_VALUES_AIR_ID,
            }
            .into());
        }
        let public_values_air_proof_data = &proof.proof.per_air[2];

        let program_commit =
            proof.proof.commitments.main_trace[PROGRAM_CACHED_TRACE_INDEX].as_ref();
        let internal_commit: &[_; CHUNK] = &agg_stark_pk
            .internal_committed_exe
            .get_program_commit()
            .into();

        let (vm_pk, vm_commit) = if program_commit == internal_commit {
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
                &agg_stark_pk.internal_vm_pk,
                internal_pvs.extra_pvs.leaf_verifier_commit,
            )
        } else {
            (&agg_stark_pk.leaf_vm_pk, *program_commit)
        };
        let e = E::new(vm_pk.fri_params);
        e.verify(&vm_pk.vm_pk.get_vk(), &proof.proof)
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

        let exe_commit = compute_exe_commit(
            &hasher,
            &pvs.app_commit,
            &pvs.memory.initial_root,
            pvs.connector.initial_pc,
        );
        let app_commit = AppExecutionCommit::from_field_commit(exe_commit, vm_commit);
        let exe_commit_bn254 = app_commit.app_exe_commit.to_bn254();
        let vm_commit_bn254 = app_commit.app_vm_commit.to_bn254();

        if exe_commit_bn254 != expected_app_commit.app_exe_commit.to_bn254() {
            return Err(SdkError::InvalidAppExeCommit {
                expected: expected_app_commit.app_exe_commit,
                actual: app_commit.app_exe_commit,
            });
        } else if vm_commit_bn254 != expected_app_commit.app_vm_commit.to_bn254() {
            return Err(SdkError::InvalidAppVmCommit {
                expected: expected_app_commit.app_vm_commit,
                actual: app_commit.app_vm_commit,
            });
        }
        Ok(())
    }

    #[cfg(feature = "evm-prove")]
    pub fn generate_evm_proof(
        &self,
        reader: &impl Halo2ParamsReader,
        app_vm_builder: VB,
        app_pk: AppProvingKey<VB::VmConfig>,
        app_exe: Arc<VmCommittedExe<SC>>,
        agg_pk: AggProvingKey,
        inputs: StdIn,
    ) -> Result<EvmProof, SdkError>
    where
        VB: VmBuilder<E>,
        <VB::VmConfig as VmExecutionConfig<F>>::Executor: Executor<F>
            + MeteredExecutor<F>
            + PreflightExecutor<F, <VB as VmBuilder<E>>::RecordArena>,
    {
        let mut e2e_prover = EvmHalo2Prover::<E, _, _>::new(
            reader,
            app_vm_builder,
            self.native_builder.clone(),
            app_pk,
            app_exe,
            agg_pk,
            self.agg_tree_config,
        )?;
        let proof = e2e_prover.generate_proof_for_evm(inputs)?;
        Ok(proof)
    }

    #[cfg(feature = "evm-verify")]
    pub fn generate_halo2_verifier_solidity(
        &self,
        reader: &impl Halo2ParamsReader,
        agg_pk: &AggProvingKey,
    ) -> Result<types::EvmHalo2Verifier, SdkError> {
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

        let params = reader.read_params(agg_pk.halo2_pk.wrapper.pinning.metadata.config_params.k);
        let pinning = &agg_pk.halo2_pk.wrapper.pinning;

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

        let wrapper_pvs = agg_pk.halo2_pk.wrapper.pinning.metadata.num_pvs.clone();
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
        &self,
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
