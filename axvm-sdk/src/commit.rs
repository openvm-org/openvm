use ax_stark_sdk::{
    ax_stark_backend::{config::Val, p3_field::AbstractField},
    config::baby_bear_poseidon2::BabyBearPoseidon2Config,
};
use axvm_circuit::{
    arch::{
        hasher::{poseidon2::vm_poseidon2_hasher, Hasher},
        VmGenericConfig,
    },
    system::{
        memory::{memory_image_to_equipartition, tree::MemoryNode},
        program::trace::AxVmCommittedExe,
    },
};
use axvm_native_compiler::ir::DIGEST_SIZE;

type SC = BabyBearPoseidon2Config;

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

impl AppExecutionCommit<Val<SC>> {
    /// Users should use this function to compute `AppExecutionCommit` and check it against the final
    /// proof.
    pub fn compute<VC: VmGenericConfig<Val<SC>>>(
        app_vm_config: &VC,
        app_exe: &AxVmCommittedExe<SC>,
        leaf_vm_verifier_exe: &AxVmCommittedExe<SC>,
    ) -> Self {
        assert!(
            app_exe.exe.program.max_num_public_values <= app_vm_config.system().num_public_values
        );
        let hasher = vm_poseidon2_hasher();
        let memory_dimensions = app_vm_config.system().memory_config.memory_dimensions();
        let app_program_commit: [Val<SC>; DIGEST_SIZE] =
            app_exe.committed_program.prover_data.commit.into();
        let leaf_verifier_program_commit: [Val<SC>; DIGEST_SIZE] = leaf_vm_verifier_exe
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
        let mut padded_pc_start = [Val::<SC>::ZERO; DIGEST_SIZE];
        padded_pc_start[0] = Val::<SC>::from_canonical_u32(app_exe.exe.pc_start);
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
}
