use std::sync::Arc;

use openvm_circuit::{
    arch::{
        hasher::{poseidon2::vm_poseidon2_hasher, Hasher},
        instructions::exe::VmExe,
        VmConfig,
    },
    system::{
        memory::{memory_image_to_equipartition, tree::MemoryNode},
        program::trace::VmCommittedExe,
    },
};
use openvm_native_compiler::{conversion::CompilerOptions, ir::DIGEST_SIZE};
use openvm_stark_backend::{config::StarkGenericConfig, p3_field::PrimeField32};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
    openvm_stark_backend::p3_field::AbstractField,
    p3_baby_bear::BabyBear,
    p3_bn254_fr::Bn254Fr,
};

use crate::{
    keygen::AppProvingKey, verifier::leaf::LeafVmVerifierConfig, NonRootCommittedExe, F, SC,
};

/// `AppExecutionCommit` has all the commitments users should check against the final proof.
pub struct AppExecutionCommit<T> {
    /// Commitment of the leaf VM verifier program which commits the VmConfig of App VM.
    /// Internal verifier will verify `leaf_vm_verifier_commit`.
    pub leaf_vm_verifier_commit: [T; DIGEST_SIZE],
    /// Commitment of the executable. It's computed as
    /// compress(
    ///     compress(
    ///         hash(app_program_commit),
    ///         hash(init_memory_commit)
    ///     ),
    ///     hash(right_pad(pc_start, 0))
    /// )
    /// `right_pad` example, if pc_start = 123, right_pad(pc_start, 0) = [123,0,0,0,0,0,0,0]
    pub exe_commit: [T; DIGEST_SIZE],
}

impl AppExecutionCommit<F> {
    /// Users should use this function to compute `AppExecutionCommit` and check it against the final
    /// proof.
    pub fn compute<VC: VmConfig<F>>(
        app_vm_config: &VC,
        app_exe: &NonRootCommittedExe,
        leaf_vm_verifier_exe: &NonRootCommittedExe,
    ) -> Self {
        assert!(
            app_exe.exe.program.max_num_public_values <= app_vm_config.system().num_public_values
        );
        let hasher = vm_poseidon2_hasher();
        let memory_dimensions = app_vm_config.system().memory_config.memory_dimensions();
        let app_program_commit: [F; DIGEST_SIZE] =
            app_exe.committed_program.prover_data.commit.into();
        let leaf_verifier_program_commit: [F; DIGEST_SIZE] = leaf_vm_verifier_exe
            .committed_program
            .prover_data
            .commit
            .into();

        let init_memory_commit = MemoryNode::tree_from_memory(
            memory_dimensions,
            &memory_image_to_equipartition(app_exe.exe.init_memory.clone()),
            &hasher,
        )
        .hash();
        let mut padded_pc_start = [F::ZERO; DIGEST_SIZE];
        padded_pc_start[0] = F::from_canonical_u32(app_exe.exe.pc_start);
        let app_hash = hasher.hash(&app_program_commit);
        let init_memory_hash = hasher.hash(&init_memory_commit);
        let pc_start_hash = hasher.hash(&padded_pc_start);
        let compress_1 = hasher.compress(&app_hash, &init_memory_hash);
        let user_commit = hasher.compress(&compress_1, &pc_start_hash);

        Self {
            leaf_vm_verifier_commit: leaf_verifier_program_commit,
            exe_commit: user_commit,
        }
    }

    pub fn app_config_commit_to_bn254(&self) -> Bn254Fr {
        babybear_digest_to_bn254(&self.leaf_vm_verifier_commit)
    }

    pub fn exe_commit_to_bn254(&self) -> Bn254Fr {
        babybear_digest_to_bn254(&self.exe_commit)
    }
}

pub(crate) fn babybear_digest_to_bn254(digest: &[F; DIGEST_SIZE]) -> Bn254Fr {
    let mut ret = Bn254Fr::ZERO;
    let order = Bn254Fr::from_canonical_u32(BabyBear::ORDER_U32);
    let mut base = Bn254Fr::ONE;
    digest.iter().for_each(|&x| {
        ret += base * Bn254Fr::from_canonical_u32(x.as_canonical_u32());
        base *= order;
    });
    ret
}

pub fn generate_leaf_committed_exe<VC: VmConfig<F>>(
    leaf_fri_params: FriParameters,
    compiler_options: CompilerOptions,
    app_pk: &AppProvingKey<VC>,
) -> Arc<NonRootCommittedExe> {
    let app_vm_vk = app_pk.app_vm_pk.vm_pk.get_vk();
    let leaf_engine = BabyBearPoseidon2Engine::new(leaf_fri_params);
    let leaf_program = LeafVmVerifierConfig {
        app_fri_params: app_pk.app_vm_pk.fri_params,
        app_system_config: app_pk.app_vm_pk.vm_config.system().clone(),
        compiler_options,
    }
    .build_program(&app_vm_vk);
    Arc::new(VmCommittedExe::commit(
        leaf_program.into(),
        leaf_engine.config.pcs(),
    ))
}

pub fn commit_app_exe(
    app_fri_params: FriParameters,
    app_exe: impl Into<VmExe<F>>,
) -> Arc<NonRootCommittedExe> {
    let exe: VmExe<_> = app_exe.into();
    let app_engine = BabyBearPoseidon2Engine::new(app_fri_params);
    Arc::new(VmCommittedExe::<SC>::commit(exe, app_engine.config.pcs()))
}
