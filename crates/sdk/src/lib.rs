use std::{borrow::Borrow, fs::read, marker::PhantomData, path::Path, sync::Arc};

#[cfg(feature = "evm-verify")]
use alloy_sol_types::sol;
use commit::{commit_app_exe, AppExecutionCommit};
use config::{AggregationTreeConfig, AppConfig};
use eyre::Result;
use keygen::{AppProvingKey, AppVerifyingKey};
use openvm_build::{
    build_guest_package, find_unique_executable, get_package, GuestOptions, TargetFilter,
};
use openvm_circuit::{
    arch::{
        hasher::{poseidon2::vm_poseidon2_hasher, Hasher},
        instructions::exe::VmExe,
        verify_segments, ContinuationVmProof, ExecutionError, InitFileGenerator,
        VerifiedExecutionPayload, VmConfig, VmExecutor, CONNECTOR_AIR_ID, PROGRAM_AIR_ID,
        PROGRAM_CACHED_TRACE_INDEX, PUBLIC_VALUES_AIR_ID,
    },
    system::{
        memory::{tree::public_values::extract_public_values, CHUNK},
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
    root::{types::RootVmVerifierInput, RootVmVerifierConfig},
};
// Re-exports:
pub use openvm_continuations::{RootSC, C, F, SC};
#[cfg(feature = "evm-prove")]
use openvm_native_recursion::halo2::utils::Halo2ParamsReader;
use openvm_stark_backend::proof::Proof;
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
    openvm_stark_backend::Chip,
    p3_bn254_fr::Bn254Fr,
};
use openvm_transpiler::{
    elf::Elf,
    openvm_platform::memory::MEM_SIZE,
    transpiler::{Transpiler, TranspilerError},
    FromElf,
};
#[cfg(feature = "evm-verify")]
use snark_verifier_sdk::{evm::gen_evm_verifier_sol_code, halo2::aggregation::AggregationCircuit};

#[cfg(feature = "evm-prove")]
use crate::{config::AggConfig, keygen::AggProvingKey, prover::EvmHalo2Prover, types::EvmProof};
use crate::{
    config::{AggStarkConfig, SdkVmConfig},
    keygen::{asm::program_to_asm, AggStarkProvingKey},
    prover::{AppProver, StarkProver},
};

pub mod codec;
pub mod commit;
pub mod config;
pub mod keygen;
pub mod prover;

mod stdin;
pub use stdin::*;

pub mod fs;
pub mod types;

pub type NonRootCommittedExe = VmCommittedExe<SC>;

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

/// The payload of a verified guest VM execution with user public values extracted and
/// verified.
pub struct VerifiedContinuationVmPayload {
    /// The Merklelized hash of:
    /// - Program code commitment (commitment of the cached trace)
    /// - Merkle root of the initial memory
    /// - Starting program counter (`pc_start`)
    ///
    /// The Merklelization uses Poseidon2 as a cryptographic hash function (for the leaves)
    /// and a cryptographic compression function (for internal nodes).
    pub exe_commit: [F; CHUNK],
    pub user_public_values: Vec<F>,
}

pub struct GenericSdk<E: StarkFriEngine<SC>> {
    agg_tree_config: AggregationTreeConfig,
    _phantom: PhantomData<E>,
}

impl<E: StarkFriEngine<SC>> Default for GenericSdk<E> {
    fn default() -> Self {
        Self {
            agg_tree_config: AggregationTreeConfig::default(),
            _phantom: PhantomData,
        }
    }
}

pub type Sdk = GenericSdk<BabyBearPoseidon2Engine>;

impl<E: StarkFriEngine<SC>> GenericSdk<E> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_agg_tree_config(mut self, agg_tree_config: AggregationTreeConfig) -> Self {
        self.agg_tree_config = agg_tree_config;
        self
    }

    pub fn agg_tree_config(&self) -> &AggregationTreeConfig {
        &self.agg_tree_config
    }

    pub fn build<P: AsRef<Path>>(
        &self,
        guest_opts: GuestOptions,
        vm_config: &SdkVmConfig,
        pkg_dir: P,
        target_filter: &Option<TargetFilter>,
        init_file_name: Option<&str>, // If None, we use "openvm-init.rs"
    ) -> Result<Elf> {
        vm_config.write_to_init_file(pkg_dir.as_ref(), init_file_name)?;
        let pkg = get_package(pkg_dir.as_ref());
        let target_dir = match build_guest_package(&pkg, &guest_opts, None, target_filter) {
            Ok(target_dir) => target_dir,
            Err(Some(code)) => {
                return Err(eyre::eyre!("Failed to build guest: code = {}", code));
            }
            Err(None) => {
                return Err(eyre::eyre!(
                    "Failed to build guest (OPENVM_SKIP_BUILD is set)"
                ));
            }
        };

        let elf_path = find_unique_executable(pkg_dir, target_dir, target_filter)?;
        let data = read(&elf_path)?;
        Elf::decode(&data, MEM_SIZE as u32)
    }

    pub fn transpile(
        &self,
        elf: Elf,
        transpiler: Transpiler<F>,
    ) -> Result<VmExe<F>, TranspilerError> {
        VmExe::from_elf(elf, transpiler)
    }

    pub fn execute<VC: VmConfig<F>>(
        &self,
        exe: VmExe<F>,
        vm_config: VC,
        inputs: StdIn,
    ) -> Result<Vec<F>, ExecutionError>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let vm = VmExecutor::new(vm_config);
        let final_memory = vm.execute(exe, inputs)?;
        let public_values = extract_public_values(
            &vm.config.system().memory_config.memory_dimensions(),
            vm.config.system().num_public_values,
            final_memory.as_ref().unwrap(),
        );
        Ok(public_values)
    }

    pub fn commit_app_exe(
        &self,
        app_fri_params: FriParameters,
        exe: VmExe<F>,
    ) -> Result<Arc<NonRootCommittedExe>> {
        let committed_exe = commit_app_exe(app_fri_params, exe);
        Ok(committed_exe)
    }

    pub fn app_keygen<VC: VmConfig<F>>(&self, config: AppConfig<VC>) -> Result<AppProvingKey<VC>>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let app_pk = AppProvingKey::keygen(config);
        Ok(app_pk)
    }

    pub fn generate_app_proof<VC: VmConfig<F>>(
        &self,
        app_pk: Arc<AppProvingKey<VC>>,
        app_committed_exe: Arc<NonRootCommittedExe>,
        inputs: StdIn,
    ) -> Result<ContinuationVmProof<SC>>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let app_prover = AppProver::<VC, E>::new(app_pk.app_vm_pk.clone(), app_committed_exe);
        let proof = app_prover.generate_app_proof(inputs);
        Ok(proof)
    }

    /// Verifies the [ContinuationVmProof], which is a collection of STARK proofs as well as
    /// additional Merkle proof for user public values.
    ///
    /// This function verifies the STARK proofs and additional conditions to ensure that the
    /// `proof` is a valid proof of guest VM execution that terminates successfully (exit code 0)
    /// _with respect to_ a commitment to some VM executable.
    /// It is the responsibility of the caller to check that the commitment matches the expected
    /// VM executable.
    pub fn verify_app_proof(
        &self,
        app_vk: &AppVerifyingKey,
        proof: &ContinuationVmProof<SC>,
    ) -> Result<VerifiedContinuationVmPayload> {
        let engine = E::new(app_vk.fri_params);
        let VerifiedExecutionPayload {
            exe_commit,
            final_memory_root,
        } = verify_segments(&engine, &app_vk.app_vm_vk, &proof.per_segment)?;

        let hasher = vm_poseidon2_hasher();
        proof
            .user_public_values
            .verify(&hasher, app_vk.memory_dimensions, final_memory_root)?;

        Ok(VerifiedContinuationVmPayload {
            exe_commit,
            user_public_values: proof.user_public_values.public_values.clone(),
        })
    }

    pub fn verify_app_proof_without_continuations(
        &self,
        app_vk: &AppVerifyingKey,
        proof: &Proof<SC>,
    ) -> Result<()> {
        let e = E::new(app_vk.fri_params);
        e.verify(&app_vk.app_vm_vk, proof)?;
        Ok(())
    }

    #[cfg(feature = "evm-prove")]
    pub fn agg_keygen(
        &self,
        config: AggConfig,
        reader: &impl Halo2ParamsReader,
        pv_handler: &impl StaticVerifierPvHandler,
    ) -> Result<AggProvingKey> {
        let agg_pk = AggProvingKey::keygen(config, reader, pv_handler);
        Ok(agg_pk)
    }

    pub fn agg_stark_keygen(&self, config: AggStarkConfig) -> Result<AggStarkProvingKey> {
        let agg_pk = AggStarkProvingKey::keygen(config);
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

    pub fn generate_root_verifier_input<VC: VmConfig<F>>(
        &self,
        app_pk: Arc<AppProvingKey<VC>>,
        app_exe: Arc<NonRootCommittedExe>,
        agg_stark_pk: AggStarkProvingKey,
        inputs: StdIn,
    ) -> Result<RootVmVerifierInput<SC>>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let stark_prover =
            StarkProver::<VC, E>::new(app_pk, app_exe, agg_stark_pk, self.agg_tree_config);
        let proof = stark_prover.generate_root_verifier_input(inputs);
        Ok(proof)
    }

    pub fn generate_e2e_stark_proof<VC: VmConfig<F>>(
        &self,
        app_pk: Arc<AppProvingKey<VC>>,
        app_exe: Arc<NonRootCommittedExe>,
        agg_stark_pk: AggStarkProvingKey,
        inputs: StdIn,
    ) -> Result<VmStarkProof<SC>>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let stark_prover =
            StarkProver::<VC, E>::new(app_pk, app_exe, agg_stark_pk, self.agg_tree_config);
        let proof = stark_prover.generate_e2e_stark_proof(inputs);
        Ok(proof)
    }

    pub fn verify_e2e_stark_proof(
        &self,
        agg_stark_pk: &AggStarkProvingKey,
        proof: &VmStarkProof<SC>,
        expected_exe_commit: &Bn254Fr,
        expected_vm_commit: &Bn254Fr,
    ) -> Result<AppExecutionCommit> {
        if proof.proof.per_air.len() < 3 {
            return Err(eyre::eyre!(
                "Invalid number of AIRs: expected at least 3, got {}",
                proof.proof.per_air.len()
            ));
        } else if proof.proof.per_air[0].air_id != PROGRAM_AIR_ID {
            return Err(eyre::eyre!("Missing program AIR"));
        } else if proof.proof.per_air[1].air_id != CONNECTOR_AIR_ID {
            return Err(eyre::eyre!("Missing connector AIR"));
        } else if proof.proof.per_air[2].air_id != PUBLIC_VALUES_AIR_ID {
            return Err(eyre::eyre!("Missing public values AIR"));
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
                return Err(eyre::eyre!(
                    "Invalid internal program commit: expected {:?}, got {:?}",
                    internal_commit,
                    internal_pvs.extra_pvs.internal_program_commit
                ));
            }
            (
                &agg_stark_pk.internal_vm_pk,
                internal_pvs.extra_pvs.leaf_verifier_commit,
            )
        } else {
            (&agg_stark_pk.leaf_vm_pk, *program_commit)
        };
        let e = E::new(vm_pk.fri_params);
        e.verify(&vm_pk.vm_pk.get_vk(), &proof.proof)?;

        let pvs: &VmVerifierPvs<_> =
            public_values_air_proof_data.public_values[..VmVerifierPvs::<u8>::width()].borrow();

        if let Some(exit_code) = pvs.connector.exit_code() {
            if exit_code != 0 {
                return Err(eyre::eyre!(
                    "Invalid exit code: expected 0, got {}",
                    exit_code
                ));
            }
        } else {
            return Err(eyre::eyre!("Program did not terminate"));
        }

        let hasher = vm_poseidon2_hasher();
        let public_values_root = hasher.merkle_root(&proof.user_public_values);
        if public_values_root != pvs.public_values_commit {
            return Err(eyre::eyre!(
                "Invalid public values root: expected {:?}, got {:?}",
                pvs.public_values_commit,
                public_values_root
            ));
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

        if exe_commit_bn254 != *expected_exe_commit {
            return Err(eyre::eyre!(
                "Invalid app exe commit: expected {:?}, got {:?}",
                expected_exe_commit,
                exe_commit_bn254
            ));
        } else if vm_commit_bn254 != *expected_vm_commit {
            return Err(eyre::eyre!(
                "Invalid app vm commit: expected {:?}, got {:?}",
                expected_vm_commit,
                vm_commit_bn254
            ));
        }
        Ok(app_commit)
    }

    #[cfg(feature = "evm-prove")]
    pub fn generate_evm_proof<VC: VmConfig<F>>(
        &self,
        reader: &impl Halo2ParamsReader,
        app_pk: Arc<AppProvingKey<VC>>,
        app_exe: Arc<NonRootCommittedExe>,
        agg_pk: AggProvingKey,
        inputs: StdIn,
    ) -> Result<EvmProof>
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let e2e_prover =
            EvmHalo2Prover::<VC, E>::new(reader, app_pk, app_exe, agg_pk, self.agg_tree_config);
        let proof = e2e_prover.generate_proof_for_evm(inputs);
        Ok(proof)
    }

    #[cfg(feature = "evm-verify")]
    pub fn generate_halo2_verifier_solidity(
        &self,
        reader: &impl Halo2ParamsReader,
        agg_pk: &AggProvingKey,
    ) -> Result<types::EvmHalo2Verifier> {
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
        let temp_dir = tempdir().wrap_err("Failed to create temp dir")?;
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
            eyre::bail!(
                "solc exited with status {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let parsed: Value = serde_json::from_slice(&output.stdout)?;

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
    ) -> Result<u64> {
        let calldata = evm_proof.verifier_calldata();
        let deployment_code = openvm_verifier.artifact.bytecode.clone();

        let gas_cost = snark_verifier::loader::evm::deploy_and_call(deployment_code, calldata)
            .map_err(|reason| eyre::eyre!("Sdk::verify_openvm_evm_proof: {reason:?}"))?;

        Ok(gas_cost)
    }
}
