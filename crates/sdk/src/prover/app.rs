use std::sync::{Arc, OnceLock};

use getset::Getters;
use itertools::Itertools;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::{vm_poseidon2_hasher, Poseidon2Hasher},
        instructions::exe::VmExe,
        verify_segments, ContinuationVmProof, ContinuationVmProver, Executor, MeteredExecutor,
        PreflightExecutor, VerifiedExecutionPayload, VirtualMachine, VirtualMachineError,
        VmBuilder, VmExecutionConfig, VmInstance, VmVerificationError,
    },
    system::memory::CHUNK,
};
use openvm_stark_backend::{
    config::{Com, Val},
    keygen::types::MultiStarkVerifyingKey,
    p3_field::PrimeField32,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    engine::{StarkEngine, StarkFriEngine},
};
use tracing::info_span;

use crate::{
    commit::{AppExecutionCommit, CommitBytes},
    keygen::AppVerifyingKey,
    prover::vm::{new_local_prover, types::VmProvingKey},
    util::check_max_constraint_degrees,
    StdIn, F, SC,
};

#[derive(Getters)]
pub struct AppProver<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub program_name: Option<String>,
    #[getset(get = "pub")]
    instance: VmInstance<E, VB>,
    #[getset(get = "pub")]
    app_vm_vk: MultiStarkVerifyingKey<E::SC>,
    #[getset(get = "pub")]
    leaf_verifier_program_commit: Com<E::SC>,

    app_execution_commit: OnceLock<AppExecutionCommit>,
}

impl<E, VB> AppProver<E, VB>
where
    E: StarkFriEngine,
    VB: VmBuilder<E>,
    Val<E::SC>: PrimeField32,
    Com<E::SC>: AsRef<[Val<E::SC>; CHUNK]> + From<[Val<E::SC>; CHUNK]> + Into<[Val<E::SC>; CHUNK]>,
{
    /// Creates a new [AppProver] instance. This method will re-commit the `exe` program on device.
    /// If a cached version of the program already exists on device, then directly use the
    /// [`Self::new_from_instance`] constructor.
    ///
    /// The `leaf_verifier_program_commit` is the commitment to the program of the leaf verifier
    /// that verifies the App VM circuit. It can be found in the `AppProvingKey`.
    pub fn new(
        vm_builder: VB,
        app_vm_pk: &VmProvingKey<E::SC, VB::VmConfig>,
        app_exe: Arc<VmExe<Val<E::SC>>>,
        leaf_verifier_program_commit: Com<E::SC>,
    ) -> Result<Self, VirtualMachineError> {
        let instance = new_local_prover(vm_builder, app_vm_pk, app_exe)?;
        let app_vm_vk = app_vm_pk.vm_pk.get_vk();

        Ok(Self::new_from_instance(
            instance,
            app_vm_vk,
            leaf_verifier_program_commit,
        ))
    }

    pub fn new_from_instance(
        instance: VmInstance<E, VB>,
        app_vm_vk: MultiStarkVerifyingKey<E::SC>,
        leaf_verifier_program_commit: Com<E::SC>,
    ) -> Self {
        Self {
            program_name: None,
            instance,
            app_vm_vk,
            leaf_verifier_program_commit,
            app_execution_commit: OnceLock::new(),
        }
    }

    pub fn set_program_name(&mut self, program_name: impl AsRef<str>) -> &mut Self {
        self.program_name = Some(program_name.as_ref().to_string());
        self
    }
    pub fn with_program_name(mut self, program_name: impl AsRef<str>) -> Self {
        self.set_program_name(program_name);
        self
    }

    /// Returns [AppExecutionCommit], which is a commitment to **both** the App VM and the App
    /// VmExe.
    pub fn app_commit(&self) -> AppExecutionCommit {
        *self.app_execution_commit.get_or_init(|| {
            AppExecutionCommit::compute::<E::SC>(
                &self.instance().vm.config().as_ref().memory_config,
                self.instance().exe(),
                self.instance().program_commitment().clone(),
                self.leaf_verifier_program_commit.clone(),
            )
        })
    }

    pub fn app_program_commit(&self) -> Com<E::SC> {
        self.instance().program_commitment().clone()
    }

    /// Generates proof for every continuation segment
    ///
    /// This function internally calls [verify_app_proof] to verify the result before returning the
    /// proof.
    pub fn prove(
        &mut self,
        input: StdIn<Val<E::SC>>,
    ) -> Result<ContinuationVmProof<E::SC>, VirtualMachineError>
    where
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
            + MeteredExecutor<Val<E::SC>>
            + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
    {
        assert!(
            self.vm_config().as_ref().continuation_enabled,
            "Use generate_app_proof_without_continuations instead."
        );
        check_max_constraint_degrees(
            self.vm_config().as_ref(),
            &self.instance.vm.engine.fri_params(),
        );
        let proofs = info_span!(
            "app proof",
            group = self
                .program_name
                .as_ref()
                .unwrap_or(&"app_proof".to_string())
        )
        .in_scope(|| {
            #[cfg(feature = "metrics")]
            metrics::counter!("fri.log_blowup")
                .absolute(self.instance.vm.engine.fri_params().log_blowup as u64);
            ContinuationVmProver::prove(&mut self.instance, input)
        })?;
        // We skip verification of the user public values proof here because it is directly computed
        // from the merkle tree above
        let res = verify_segments(
            &self.instance.vm.engine,
            &self.app_vm_vk,
            &proofs.per_segment,
        )?;
        let app_exe_commit_u32s = self.app_commit().app_exe_commit.to_u32_digest();
        let exe_commit_u32s = res.exe_commit.map(|x| x.as_canonical_u32());
        if exe_commit_u32s != app_exe_commit_u32s {
            return Err(VmVerificationError::ExeCommitMismatch {
                expected: app_exe_commit_u32s,
                actual: exe_commit_u32s,
            }
            .into());
        }
        Ok(proofs)
    }

    /// App Exe
    pub fn exe(&self) -> Arc<VmExe<Val<E::SC>>> {
        self.instance.exe().clone()
    }

    /// App VM
    pub fn vm(&self) -> &VirtualMachine<E, VB> {
        &self.instance.vm
    }

    /// App VM config
    pub fn vm_config(&self) -> &VB::VmConfig {
        self.instance.vm.config()
    }
}

/// The payload of a verified guest VM execution with user public values extracted and
/// verified.
pub struct VerifiedAppArtifacts {
    /// The Merklelized hash of:
    /// - Program code commitment (commitment of the cached trace)
    /// - Merkle root of the initial memory
    /// - Starting program counter (`pc_start`)
    ///
    /// The Merklelization uses Poseidon2 as a cryptographic hash function (for the leaves)
    /// and a cryptographic compression function (for internal nodes).
    pub app_exe_commit: CommitBytes,
    pub user_public_values: Vec<u8>,
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
    app_vk: &AppVerifyingKey,
    proof: &ContinuationVmProof<SC>,
) -> Result<VerifiedAppArtifacts, VmVerificationError> {
    static POSEIDON2_HASHER: OnceLock<Poseidon2Hasher<F>> = OnceLock::new();
    let engine = BabyBearPoseidon2Engine::new(app_vk.fri_params);
    let VerifiedExecutionPayload {
        exe_commit,
        final_memory_root,
    } = verify_segments(&engine, &app_vk.vk, &proof.per_segment)?;

    proof.user_public_values.verify(
        POSEIDON2_HASHER.get_or_init(vm_poseidon2_hasher),
        app_vk.memory_dimensions,
        final_memory_root,
    )?;

    let app_exe_commit = CommitBytes::from_u32_digest(&exe_commit.map(|x| x.as_canonical_u32()));
    // The user public values address space has cells have type u8
    let user_public_values = proof
        .user_public_values
        .public_values
        .iter()
        .map(|x| x.as_canonical_u32().try_into().unwrap())
        .collect_vec();
    Ok(VerifiedAppArtifacts {
        app_exe_commit,
        user_public_values,
    })
}
