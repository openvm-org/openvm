use std::borrow::Borrow;

use ax_stark_sdk::{
    ax_stark_backend::keygen::types::MultiStarkVerifyingKey,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Config, FriParameters},
};
use axvm_circuit::arch::{instructions::program::Program, PUBLIC_VALUES_AIR_ID};
use axvm_native_compiler::{conversion::CompilerOptions, prelude::*};
use axvm_recursion::{
    challenger::duplex::DuplexChallengerVariable,
    fri::TwoAdicFriPcsVariable,
    hints::{Hintable, InnerVal},
    stark::StarkVerifier,
    types::{new_from_inner_multi_vk, InnerConfig},
    utils::const_fri_config,
};

use crate::verifier::{
    common::{
        assert_or_assign_connector_pvs, assert_or_assign_memory_pvs,
        assert_single_segment_vm_exit_successfully, get_program_commit,
    },
    internal::{
        types::{InternalVmVerifierInput, InternalVmVerifierPvs},
        vars::InternalVmVerifierInputVariable,
    },
    utils::{assign_array_to_slice, eq_felt_slice},
};

pub mod types;
mod vars;

type C = InnerConfig;
type F = InnerVal;

/// Config to generate internal VM verifier program.
pub struct InternalVmVerifierConfig {
    pub fri_params: FriParameters,
    pub compiler_options: CompilerOptions,
}

impl InternalVmVerifierConfig {
    pub fn build_program(
        &self,
        agg_vm_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> Program<F> {
        let m_advice = new_from_inner_multi_vk(agg_vm_vk);
        let mut builder = Builder::<C>::default();

        {
            let InternalVmVerifierInputVariable {
                self_program_commit,
                proofs,
            } = InternalVmVerifierInput::<BabyBearPoseidon2Config>::read(&mut builder);
            let pcs = TwoAdicFriPcsVariable {
                config: const_fri_config(&mut builder, &self.fri_params),
            };
            // At least 1 proof should be provided.
            builder.assert_ne::<Usize<_>>(proofs.len(), RVar::zero());

            let pvs = InternalVmVerifierPvs::<Felt<F>>::uninit(&mut builder);
            builder.assign(&pvs.self_program_commit, self_program_commit);

            builder.range(0, proofs.len()).for_each(|i, builder| {
                let proof = builder.get(&proofs, i);
                StarkVerifier::verify::<DuplexChallengerVariable<C>>(
                    builder, &pcs, &m_advice, &proof,
                );
                assert_single_segment_vm_exit_successfully(builder, &proof);

                let program_commit = get_program_commit(builder, &proof);
                let is_self_program = eq_felt_slice(builder, &self_program_commit, &program_commit);

                let flatten_proof_vm_pvs =
                    InternalVmVerifierPvs::<Felt<F>>::uninit(builder).flatten();
                let proof_vm_pvs: &InternalVmVerifierPvs<_> = {
                    let proof_vm_pvs_arr = builder
                        .get(&proof.per_air, PUBLIC_VALUES_AIR_ID)
                        .public_values;
                    // Leaf verifier has less logical public values but the number of PublicValuesAir is the same.
                    assign_array_to_slice(builder, &flatten_proof_vm_pvs, &proof_vm_pvs_arr, 0);
                    flatten_proof_vm_pvs.as_slice().borrow()
                };

                let proof_leaf_commit: [Felt<_>; DIGEST_SIZE] = builder.uninit();
                builder.if_eq(is_self_program, RVar::one()).then_or_else(
                    |builder| {
                        // Handle recursive verification
                        // For proofs, its program commitment should be committed.
                        builder.assert_eq::<[_; DIGEST_SIZE]>(
                            proof_vm_pvs.self_program_commit,
                            program_commit,
                        );
                        builder.assign(&proof_leaf_commit, proof_vm_pvs.leaf_verifier_commit);
                    },
                    |builder| {
                        // Treat the proof as a leaf verifier proof when it is not a self program.
                        builder.assign(&proof_leaf_commit, program_commit);
                    },
                );
                builder.if_eq(i, RVar::zero()).then_or_else(
                    |builder| {
                        builder.assign(
                            &pvs.vm_verifier_pvs.app_commit,
                            proof_vm_pvs.vm_verifier_pvs.app_commit,
                        );
                        builder.assign(&pvs.leaf_verifier_commit, proof_leaf_commit);
                    },
                    |builder| {
                        builder.assert_eq::<[_; DIGEST_SIZE]>(
                            pvs.vm_verifier_pvs.app_commit,
                            proof_vm_pvs.vm_verifier_pvs.app_commit,
                        );
                        builder.assert_eq::<[_; DIGEST_SIZE]>(
                            pvs.leaf_verifier_commit,
                            proof_leaf_commit,
                        );
                    },
                );

                assert_or_assign_connector_pvs(
                    builder,
                    &pvs.vm_verifier_pvs.connector,
                    i,
                    &proof_vm_pvs.vm_verifier_pvs.connector,
                );
                assert_or_assign_memory_pvs(
                    builder,
                    &pvs.vm_verifier_pvs.memory,
                    i,
                    &proof_vm_pvs.vm_verifier_pvs.memory,
                );
                // This is only needed when `is_terminate` but branching here won't save much, so we
                // always assign it.
                builder.assign(
                    &pvs.vm_verifier_pvs.public_values_commit,
                    proof_vm_pvs.vm_verifier_pvs.public_values_commit,
                );
            });
            for pv in pvs.flatten() {
                builder.commit_public_value(pv);
            }

            builder.halt();
        }

        builder.compile_isa_with_options(self.compiler_options.clone())
    }
}
