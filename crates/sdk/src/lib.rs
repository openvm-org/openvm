use std::{fs::read, marker::PhantomData, path::Path, sync::Arc};

use commit::commit_app_exe;
use config::AppConfig;
use eyre::Result;
use keygen::{AppProvingKey, AppVerifyingKey};
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
use openvm_native_recursion::halo2::{
    utils::Halo2ParamsReader,
    wrapper::{EvmVerifier, Halo2WrapperProvingKey},
    RawEvmProof,
};
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

use crate::{
    config::AggConfig,
    keygen::{AggProvingKey, AggStarkProvingKey},
    prover::{AppProver, ContinuationProver, StarkProver},
};

pub mod codec;
pub mod commit;
pub mod config;
pub mod keygen;
pub mod prover;

mod stdin;
pub use stdin::*;

use crate::types::EvmProof;

pub mod fs;
pub mod types;

pub type NonRootCommittedExe = VmCommittedExe<SC>;

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
    _phantom: PhantomData<E>,
}

impl<E: StarkFriEngine<SC>> Default for GenericSdk<E> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

pub type Sdk = GenericSdk<BabyBearPoseidon2Engine>;

impl<E: StarkFriEngine<SC>> GenericSdk<E> {
    pub fn new() -> Self {
        Self::default()
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
        let stark_prover = StarkProver::<VC, E>::new(app_pk, app_exe, agg_stark_pk);
        let proof = stark_prover.generate_root_verifier_input(inputs);
        Ok(proof)
    }

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
        let e2e_prover = ContinuationProver::<VC, E>::new(reader, app_pk, app_exe, agg_pk);
        let proof = e2e_prover.generate_proof_for_evm(inputs);
        Ok(proof)
    }

    pub fn generate_snark_verifier_contract(
        &self,
        reader: &impl Halo2ParamsReader,
        agg_pk: &AggProvingKey,
    ) -> Result<EvmVerifier> {
        let params = reader.read_params(agg_pk.halo2_pk.wrapper.pinning.metadata.config_params.k);
        let evm_verifier = agg_pk.halo2_pk.wrapper.generate_evm_verifier(&params);
        Ok(evm_verifier)
    }

    pub fn verify_evm_proof(
        &self,
        evm_verifier: &EvmVerifier,
        evm_proof: &EvmProof,
    ) -> Result<u64> {
        let evm_proof: RawEvmProof = evm_proof.clone().try_into()?;
        let gas_cost = Halo2WrapperProvingKey::evm_verify(evm_verifier, &evm_proof)
            .map_err(|reason| eyre::eyre!("Sdk::verify_evm_proof: {reason:?}"))?;
        Ok(gas_cost)
    }
}
