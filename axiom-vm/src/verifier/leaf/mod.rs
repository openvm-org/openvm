use std::array;

use ax_stark_sdk::{
    ax_stark_backend::{
        keygen::types::MultiStarkVerifyingKey, p3_field::AbstractField, p3_util::log2_strict_usize,
        prover::types::Proof,
    },
    config::{baby_bear_poseidon2::BabyBearPoseidon2Config, FriParameters},
};
use axvm_circuit::{
    arch::{
        instructions::program::Program, VmConfig, CONNECTOR_AIR_ID, MERKLE_AIR_ID,
        PROGRAM_CACHED_TRACE_INDEX,
    },
    system::{
        connector::VmConnectorPvs, memory::tree::public_values::PUBLIC_VALUES_ADDRESS_SPACE_OFFSET,
    },
};
use axvm_native_compiler::{conversion::CompilerOptions, prelude::*};
use axvm_recursion::{
    challenger::duplex::DuplexChallengerVariable,
    digest::DigestVariable,
    fri::TwoAdicFriPcsVariable,
    hints::{Hintable, InnerVal},
    stark::StarkVerifier,
    types::{new_from_inner_multi_vk, InnerConfig},
    utils::const_fri_config,
    vars::StarkProofVariable,
};
use types::LeafVmVerifierPvs;

use crate::{config::AxiomVmConfig, verifier::leaf::types::UserPublicValuesRootProof};

pub mod types;
mod vars;

type C = InnerConfig;
type F = InnerVal;

/// Config to generate
pub struct LeafVmVerifierConfig {
    pub fri_params: FriParameters,
    pub app_vm_config: VmConfig,
    pub compiler_options: CompilerOptions,
}

impl LeafVmVerifierConfig {
    pub fn build_program(
        &self,
        app_vm_vk: MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> Program<F> {
        self.app_vm_config.memory_config.memory_dimensions();
        let m_advice = new_from_inner_multi_vk(&app_vm_vk);
        let mut builder = Builder::<C>::default();

        {
            let pcs = TwoAdicFriPcsVariable {
                config: const_fri_config(&mut builder, &self.fri_params),
            };
            let proofs: Array<C, StarkProofVariable<_>> =
                <Vec<Proof<BabyBearPoseidon2Config>> as Hintable<C>>::read(&mut builder);
            // At least 1 proof should be provided.
            builder.assert_ne::<Usize<_>>(proofs.len(), RVar::zero());

            let pvs = LeafVmVerifierPvs::<Felt<F>>::uninit(&mut builder);
            builder.range(0, proofs.len()).for_each(|i, builder| {
                let proof = builder.get(&proofs, i);
                StarkVerifier::verify::<DuplexChallengerVariable<C>>(
                    builder, &pcs, &m_advice, &proof,
                );
                {
                    let t_id = RVar::from(PROGRAM_CACHED_TRACE_INDEX);
                    let commit = builder.get(&proof.commitments.main_trace, t_id);
                    let commit = if let DigestVariable::Felt(commit) = commit {
                        commit
                    } else {
                        unreachable!()
                    };
                    builder.if_eq(i, RVar::zero()).then_or_else(
                        |builder| assign_slice(builder, &pvs.app_commit, &commit, 0),
                        |builder| assert_slice(builder, &pvs.app_commit, &commit, 0),
                    );
                }
                {
                    let a_id = RVar::from(CONNECTOR_AIR_ID);
                    let a_input = builder.get(&proof.per_air, a_id);
                    let connector_pvs = &pvs.connector;
                    let input_pvs = &a_input.public_values;
                    builder.if_eq(i, RVar::zero()).then_or_else(
                        |builder| assign_connector(builder, connector_pvs, input_pvs),
                        |builder| {
                            // assert prev.final_pc == curr.initial_pc
                            let initial_pc = builder.get(input_pvs, 0);
                            builder.assert_felt_eq(connector_pvs.final_pc, initial_pc);
                            // Update final_pc
                            let final_pc = builder.get(input_pvs, 1);
                            builder.assign(&connector_pvs.final_pc, final_pc);
                            // assert prev.is_terminate == 0
                            builder.assert_felt_eq(connector_pvs.is_terminate, F::zero());
                            // Update is_terminate
                            let is_terminate = builder.get(input_pvs, 3);
                            builder.assign(&connector_pvs.is_terminate, is_terminate);
                            // Update exit_code
                            let exit_code = builder.get(input_pvs, 2);
                            builder.assign(&connector_pvs.exit_code, exit_code);
                        },
                    );
                }
                {
                    let a_id = RVar::from(MERKLE_AIR_ID);
                    let a_input = builder.get(&proof.per_air, a_id);
                    builder.if_eq(i, RVar::zero()).then_or_else(
                        |builder| {
                            assign_slice(
                                builder,
                                &pvs.memory.initial_root,
                                &a_input.public_values,
                                0,
                            );
                            assign_slice(
                                builder,
                                &pvs.memory.final_root,
                                &a_input.public_values,
                                DIGEST_SIZE,
                            );
                        },
                        |builder| {
                            // assert prev.final_root == curr.initial_root
                            assert_slice(
                                builder,
                                &pvs.memory.final_root,
                                &a_input.public_values,
                                0,
                            );
                            // Update final_root
                            assign_slice(
                                builder,
                                &pvs.memory.final_root,
                                &a_input.public_values,
                                DIGEST_SIZE,
                            );
                        },
                    );
                }
            });

            let is_terminate = builder.cast_felt_to_var(pvs.connector.is_terminate);
            builder.if_eq(is_terminate, F::one()).then(|builder| {
                let (pv_commit, expected_memory_root) =
                    self.verify_user_public_values_root(builder);
                builder.assert_eq::<[_; DIGEST_SIZE]>(pvs.memory.final_root, expected_memory_root);
                builder.assign(&pvs.public_values_commit, pv_commit);
            });
            for pv in pvs.flatten() {
                builder.commit_public_value(pv);
            }

            builder.halt();
        }

        builder.compile_isa_with_options(self.compiler_options.clone())
    }

    /// Read the public values root proof from the input stream and verify it.
    // This verification must be consistent `axvm-circuit::system::memory::tree::public_values`.
    /// Returns the public values commit and the corresponding memory state root.
    fn verify_user_public_values_root(
        &self,
        builder: &mut Builder<C>,
    ) -> ([Felt<F>; DIGEST_SIZE], [Felt<F>; DIGEST_SIZE]) {
        let memory_dimensions = self.app_vm_config.memory_config.memory_dimensions();
        let pv_as = F::from_canonical_usize(
            PUBLIC_VALUES_ADDRESS_SPACE_OFFSET + memory_dimensions.as_offset,
        );
        let pv_start_idx = memory_dimensions.label_to_index((pv_as, 0));
        let pv_height = log2_strict_usize(self.app_vm_config.num_public_values / DIGEST_SIZE);
        let proof_len = memory_dimensions.overall_height() - pv_height;
        let idx_prefix = pv_start_idx >> pv_height;

        // Read the public values root proof from the input stream.
        let root_proof = UserPublicValuesRootProof::<F>::read(builder);
        builder.assert_eq::<Usize<_>>(root_proof.sibling_hashes.len(), Usize::from(proof_len));
        let mut curr_commit = root_proof.public_values_commit;
        // Share the same state array to avoid unnecessary allocations.
        let state: Array<C, Felt<_>> = builder.array(PERMUTATION_WIDTH);
        for i in 0..proof_len {
            let sibling_hash = builder.get(&root_proof.sibling_hashes, i);
            let (l_hash, r_hash) = if idx_prefix & (1 << i) != 0 {
                (sibling_hash, curr_commit)
            } else {
                (curr_commit, sibling_hash)
            };
            for j in 0..DIGEST_SIZE {
                builder.set(&state, j, l_hash[j]);
                builder.set(&state, DIGEST_SIZE + j, r_hash[j]);
            }
            builder.poseidon2_permute_mut(&state);
            curr_commit = array::from_fn(|j| builder.get(&state, j));
        }
        (root_proof.public_values_commit, curr_commit)
    }
}

impl AxiomVmConfig {
    pub fn leaf_verifier_vm_config(&self) -> VmConfig {
        VmConfig::aggregation(
            LeafVmVerifierPvs::<u8>::width(),
            self.poseidon2_max_constraint_degree,
        )
    }
}

fn assign_slice<const CHUNK: usize>(
    builder: &mut Builder<C>,
    dst_slice: &[Felt<F>; CHUNK],
    src: &Array<C, Felt<F>>,
    src_offset: usize,
) {
    for (i, dst) in dst_slice.iter().enumerate() {
        let pv = builder.get(src, i + src_offset);
        builder.assign(dst, pv);
    }
}

fn assert_slice<const CHUNK: usize>(
    builder: &mut Builder<C>,
    dst_slice: &[Felt<F>; CHUNK],
    src: &Array<C, Felt<F>>,
    src_offset: usize,
) {
    for (i, &dst) in dst_slice.iter().enumerate() {
        let pv = builder.get(src, i + src_offset);
        builder.assert_felt_eq(dst, pv);
    }
}

fn assign_connector(
    builder: &mut Builder<C>,
    dst: &VmConnectorPvs<Felt<F>>,
    src: &Array<C, Felt<F>>,
) {
    let VmConnectorPvs {
        initial_pc,
        final_pc,
        exit_code,
        is_terminate,
    } = dst;
    let v = builder.get(src, RVar::from(0));
    builder.assign(initial_pc, v);
    let v = builder.get(src, RVar::from(1));
    builder.assign(final_pc, v);
    let v = builder.get(src, RVar::from(2));
    builder.assign(exit_code, v);
    let v = builder.get(src, RVar::from(3));
    builder.assign(is_terminate, v);
}
