use std::{fs::read, marker::PhantomData, path::Path, sync::Arc};

#[cfg(feature = "evm-verify")]
use alloy_primitives::{Bytes, FixedBytes};
#[cfg(feature = "evm-verify")]
use alloy_sol_types::{sol, SolCall, SolValue};
use eyre::Result;
use openvm_build::{
    build_guest_package, find_unique_executable, get_package, GuestOptions, TargetFilter,
};
use openvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, instructions::exe::VmExe, verify_segments,
        ContinuationVmProof, ExecutionError, VerifiedExecutionPayload, VmConfig, VmExecutor,
        VmVerificationError,
    },
    system::{
        memory::{tree::public_values::extract_public_values, CHUNK},
        program::trace::VmCommittedExe,
    },
};
use openvm_continuations::verifier::root::types::RootVmVerifierInput;
pub use openvm_continuations::{
    static_verifier::{DefaultStaticVerifierPvHandler, StaticVerifierPvHandler},
    RootSC, C, F, SC,
};
use openvm_native_recursion::halo2::utils::Halo2ParamsReader;
use openvm_stark_backend::proof::Proof;
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
    openvm_stark_backend::{verifier::VerificationError, Chip},
};
use openvm_transpiler::{
    elf::Elf,
    openvm_platform::memory::MEM_SIZE,
    transpiler::{Transpiler, TranspilerError},
    FromElf,
};
#[cfg(feature = "evm-verify")]
use snark_verifier_sdk::{evm::gen_evm_verifier_sol_code, halo2::aggregation::AggregationCircuit};

use crate::{
    commit::commit_app_exe,
    config::{AggConfig, AggregationTreeConfig, AppConfig},
    keygen::{AggProvingKey, AggStarkProvingKey, AppProvingKey, AppVerifyingKey},
    prover::{AppProver, StarkProver},
};
#[cfg(feature = "evm-prove")]
use crate::{prover::EvmHalo2Prover, types::EvmProof};

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

    pub fn agg_tree_config(&self) -> &AggregationTreeConfig {
        &self.agg_tree_config
    }

    pub fn set_agg_tree_config(&mut self, agg_tree_config: AggregationTreeConfig) {
        self.agg_tree_config = agg_tree_config;
    }

    pub fn build<P: AsRef<Path>>(
        &self,
        guest_opts: GuestOptions,
        pkg_dir: P,
        target_filter: &Option<TargetFilter>,
    ) -> Result<Elf> {
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
    ) -> Result<VerifiedContinuationVmPayload, VmVerificationError> {
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
    ) -> Result<(), VerificationError> {
        let e = E::new(app_vk.fri_params);
        e.verify(&app_vk.app_vm_vk, proof)
    }

    pub fn agg_keygen(
        &self,
        config: AggConfig,
        reader: &impl Halo2ParamsReader,
        pv_handler: &impl StaticVerifierPvHandler,
    ) -> Result<AggProvingKey> {
        let agg_pk = AggProvingKey::keygen(config, reader, pv_handler);
        Ok(agg_pk)
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
            process::Command,
        };

        use eyre::Context;
        use openvm_native_recursion::halo2::wrapper::EvmVerifierByteCode;
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
            // We subtract 14 to exclude the KZG accumulators and the app exe
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
            .replace("{OPENVM_VERSION}", env!("CARGO_PKG_VERSION"));

        // Create temp dir
        let temp_dir = tempdir().wrap_err("Failed to create temp dir")?;
        let temp_path = temp_dir.path();

        // Make interfaces dir
        let interfaces_path = temp_path.join("interfaces");
        create_dir_all(&interfaces_path)?;

        // Write the files to the temp dir. This is only for compilation
        // purposes.
        write(
            interfaces_path.join(EVM_HALO2_VERIFIER_INTERFACE_NAME),
            EVM_HALO2_VERIFIER_INTERFACE,
        )?;
        write(
            temp_path.join(EVM_HALO2_VERIFIER_PARENT_NAME),
            &halo2_verifier_code,
        )?;
        write(
            temp_path.join(EVM_HALO2_VERIFIER_BASE_NAME),
            &openvm_verifier_code,
        )?;

        // Run solc from the temp dir
        let output = Command::new("solc")
            .current_dir(temp_path)
            .arg("OpenVmHalo2Verifier.sol")
            .arg("--no-optimize-yul")
            .arg("--bin")
            .arg("--optimize")
            .arg("--optimize-runs")
            .arg("100000")
            .output()?;

        if !output.status.success() {
            eyre::bail!(
                "solc exited with status {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let bytecode = extract_binary(
            &output.stdout,
            "OpenVmHalo2Verifier.sol:OpenVmHalo2Verifier",
        );

        let evm_verifier = EvmHalo2Verifier {
            halo2_verifier_code,
            openvm_verifier_code,
            openvm_verifier_interface: EVM_HALO2_VERIFIER_INTERFACE.to_string(),
            artifact: EvmVerifierByteCode {
                sol_compiler_version: "0.8.19".to_string(),
                sol_compiler_options: "--no-optimize-yul --bin --optimize --optimize-runs 100000"
                    .to_string(),
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
        evm_proof: &EvmProof,
    ) -> Result<u64> {
        use crate::types::NUM_BN254_ACCUMULATORS;

        let EvmProof {
            accumulators,
            proof,
            user_public_values,
            exe_commit,
            leaf_commit,
        } = evm_proof;
        let mut exe_commit = *exe_commit;
        let mut leaf_commit = *leaf_commit;
        exe_commit.reverse();
        leaf_commit.reverse();

        assert_eq!(accumulators.len(), NUM_BN254_ACCUMULATORS * 32);
        let mut evm_accumulators: Vec<u8> = Vec::with_capacity(accumulators.len());
        accumulators
            .chunks(32)
            .for_each(|chunk| evm_accumulators.extend(chunk.iter().rev().cloned()));

        let mut proof_data = evm_accumulators;
        proof_data.extend(proof);

        assert!(
            user_public_values.len() % 32 == 0,
            "User public values length must be a multiple of 32"
        );

        // Take the first byte of each 32 byte chunk, and pack them together
        // into one payload.
        let user_public_values: Bytes =
            user_public_values
                .chunks(32)
                .fold(Vec::<u8>::new().into(), |acc: Bytes, chunk| {
                    // We only care about the first byte, everything else should be 0-bytes
                    (acc, FixedBytes::<1>::from(*chunk.first().unwrap()))
                        .abi_encode_packed()
                        .into()
                });

        let calldata = IOpenVmHalo2Verifier::verifyCall {
            publicValues: user_public_values.clone(),
            proofData: proof_data.into(),
            appExeCommit: exe_commit.into(),
            appVmCommit: leaf_commit.into(),
        }
        .abi_encode();
        let deployment_code = openvm_verifier.artifact.bytecode.clone();

        let gas_cost = snark_verifier::loader::evm::deploy_and_call(deployment_code, calldata)
            .map_err(|reason| eyre::eyre!("Sdk::verify_openvm_evm_proof: {reason:?}"))?;

        Ok(gas_cost)
    }
}

/// We will split the output by whitespace and look for the following
/// sequence:
/// [
///     ...
///     "=======",
///     "OpenVmHalo2Verifier.sol:OpenVmHalo2Verifier",
///     "=======",
///     "Binary:"
///     "[compiled bytecode]"
///     ...
/// ]
///
/// Once we find "OpenVmHalo2Verifier.sol:OpenVmHalo2Verifier," we can skip
/// to the appropriate offset to get the compiled bytecode.
#[cfg(feature = "evm-verify")]
fn extract_binary(output: &[u8], contract_name: &str) -> Vec<u8> {
    let split = split_by_ascii_whitespace(output);
    let contract_name_bytes = contract_name.as_bytes();

    for i in 0..split.len().saturating_sub(3) {
        if split[i] == contract_name_bytes {
            return hex::decode(split[i + 3]).expect("Invalid hex in Binary");
        }
    }

    panic!("Contract '{}' not found", contract_name);
}

#[cfg(feature = "evm-verify")]
fn split_by_ascii_whitespace(bytes: &[u8]) -> Vec<&[u8]> {
    let mut split = Vec::new();
    let mut start = None;
    for (idx, byte) in bytes.iter().enumerate() {
        if byte.is_ascii_whitespace() {
            if let Some(start) = start.take() {
                split.push(&bytes[start..idx]);
            }
        } else if start.is_none() {
            start = Some(idx);
        }
    }
    if let Some(last) = start {
        split.push(&bytes[last..]);
    }
    split
}
