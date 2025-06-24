use openvm_circuit::{
    arch::{instructions::program::Program, SystemConfig, ADDR_SPACE_OFFSET},
    system::memory::merkle::public_values::PUBLIC_VALUES_ADDRESS_SPACE_OFFSET,
};
use openvm_native_compiler::{conversion::CompilerOptions, prelude::*};
use openvm_native_recursion::{
    challenger::duplex::DuplexChallengerVariable, fri::TwoAdicFriPcsVariable, hints::Hintable,
    stark::StarkVerifier, types::new_from_inner_multi_vk, utils::const_fri_config,
    vars::StarkProofVariable,
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Config, FriParameters},
    openvm_stark_backend::{
        keygen::types::MultiStarkVerifyingKey, p3_field::FieldAlgebra, p3_util::log2_strict_usize,
        proof::Proof,
    },
};

use crate::{
    verifier::{
        common::{
            assert_or_assign_connector_pvs, assert_or_assign_memory_pvs,
            assert_required_air_for_app_vm_present, get_connector_pvs, get_memory_pvs,
            get_program_commit, types::VmVerifierPvs,
        },
        leaf::types::UserPublicValuesRootProof,
        utils::VariableP2Compressor,
    },
    C, F,
};

pub mod types;
mod vars;

/// Config to generate leaf VM verifier program.
pub struct LeafVmVerifierConfig {
    pub app_fri_params: FriParameters,
    pub app_system_config: SystemConfig,
    pub compiler_options: CompilerOptions,
}

impl LeafVmVerifierConfig {
    pub fn build_program(
        &self,
        app_vm_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> Program<F> {
        let m_advice = new_from_inner_multi_vk(app_vm_vk);
        let mut builder = Builder::<C>::default();

        {
            builder.cycle_tracker_start("InitializePcsConst");
            let pcs = TwoAdicFriPcsVariable {
                config: const_fri_config(&mut builder, &self.app_fri_params),
            };
            builder.cycle_tracker_end("InitializePcsConst");
            builder.cycle_tracker_start("ReadProofsFromInput");
            let proofs: Array<C, StarkProofVariable<_>> =
                <Vec<Proof<BabyBearPoseidon2Config>> as Hintable<C>>::read(&mut builder);
            // At least 1 proof should be provided.
            builder.assert_nonzero(&proofs.len());
            builder.cycle_tracker_end("ReadProofsFromInput");

            builder.cycle_tracker_start("VerifyProofs");
            let pvs = VmVerifierPvs::<Felt<F>>::uninit(&mut builder);
            builder.range(0, proofs.len()).for_each(|i_vec, builder| {
                let i = i_vec[0];
                let proof = builder.get(&proofs, i);
                assert_required_air_for_app_vm_present(builder, &proof);
                StarkVerifier::verify::<DuplexChallengerVariable<C>>(
                    builder, &pcs, &m_advice, &proof,
                );
                {
                    let commit = get_program_commit(builder, &proof);
                    builder.if_eq(i, RVar::zero()).then_or_else(
                        |builder| {
                            builder.assign(&pvs.app_commit, commit);
                        },
                        |builder| builder.assert_eq::<[_; DIGEST_SIZE]>(pvs.app_commit, commit),
                    );
                }

                let proof_connector_pvs = get_connector_pvs(builder, &proof);
                assert_or_assign_connector_pvs(builder, &pvs.connector, i, &proof_connector_pvs);

                let proof_memory_pvs = get_memory_pvs(builder, &proof);
                assert_or_assign_memory_pvs(builder, &pvs.memory, i, &proof_memory_pvs);
            });
            builder.cycle_tracker_end("VerifyProofs");
            builder.cycle_tracker_start("ExtractPublicValuesCommit");
            let is_terminate = builder.cast_felt_to_var(pvs.connector.is_terminate);
            builder.if_eq(is_terminate, F::ONE).then(|builder| {
                let (pv_commit, expected_memory_root) =
                    self.verify_user_public_values_root(builder);
                builder.assert_eq::<[_; DIGEST_SIZE]>(pvs.memory.final_root, expected_memory_root);
                builder.assign(&pvs.public_values_commit, pv_commit);
            });
            for pv in pvs.flatten() {
                builder.commit_public_value(pv);
            }
            builder.cycle_tracker_end("ExtractPublicValuesCommit");

            builder.halt();
        }

        builder.compile_isa_with_options(self.compiler_options)
    }

    /// Read the public values root proof from the input stream and verify it.
    /// This verification must be consistent `openvm_circuit::system::memory::tree::public_values`.
    /// Returns the public values commit and the corresponding memory state root.
    fn verify_user_public_values_root(
        &self,
        builder: &mut Builder<C>,
    ) -> ([Felt<F>; DIGEST_SIZE], [Felt<F>; DIGEST_SIZE]) {
        let memory_dimensions = self.app_system_config.memory_config.memory_dimensions();
        let pv_as = PUBLIC_VALUES_ADDRESS_SPACE_OFFSET + ADDR_SPACE_OFFSET;
        let pv_start_idx = memory_dimensions.label_to_index((pv_as, 0));
        let pv_height = log2_strict_usize(self.app_system_config.num_public_values / DIGEST_SIZE);
        let proof_len = memory_dimensions.overall_height() - pv_height;
        let idx_prefix = pv_start_idx >> pv_height;

        // Read the public values root proof from the input stream.
        let root_proof = UserPublicValuesRootProof::<F>::read(builder);
        builder.assert_eq::<Usize<_>>(root_proof.sibling_hashes.len(), Usize::from(proof_len));
        let mut curr_commit = root_proof.public_values_commit;
        // Share the same state array to avoid unnecessary allocations.
        let compressor = VariableP2Compressor::new(builder);
        for i in 0..proof_len {
            let sibling_hash = builder.get(&root_proof.sibling_hashes, i);
            let (l_hash, r_hash) = if idx_prefix & (1 << i) != 0 {
                (sibling_hash, curr_commit)
            } else {
                (curr_commit, sibling_hash)
            };
            curr_commit = compressor.compress(builder, &l_hash, &r_hash);
        }
        (root_proof.public_values_commit, curr_commit)
    }
}
