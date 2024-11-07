use std::{array, borrow::Borrow};

use ax_stark_sdk::{
    ax_stark_backend::{keygen::types::MultiStarkVerifyingKey, p3_field::AbstractField},
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
        assert_single_segment_vm_exit_successfully, get_program_commit, types::VmVerifierPvs,
    },
    internal::types::InternalVmVerifierPvs,
    root::{
        types::{RootVmVerifierInput, RootVmVerifierPvs},
        vars::RootVmVerifierInputVariable,
    },
    utils::{assign_array_to_slice, eq_felt_slice, VariableP2Hasher},
};

pub mod types;
mod vars;
type C = InnerConfig;
type F = InnerVal;

/// Config to generate Root VM verifier program.
pub struct RootVmVerifierConfig {
    pub fri_params: FriParameters,
    pub num_public_values: usize,
    pub internal_vm_verifier_commit: [F; DIGEST_SIZE],
    pub compiler_options: CompilerOptions,
}
impl RootVmVerifierConfig {
    pub fn build_program(
        &self,
        agg_vm_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> Program<F> {
        let m_advice = new_from_inner_multi_vk(agg_vm_vk);
        let mut builder = Builder::<C>::default();

        {
            let RootVmVerifierInputVariable {
                proofs,
                public_values,
            } = RootVmVerifierInput::<BabyBearPoseidon2Config>::read(&mut builder);
            let pcs = TwoAdicFriPcsVariable {
                config: const_fri_config(&mut builder, &self.fri_params),
            };
            let internal_vm_verifier_commit =
                array::from_fn(|i| builder.eval(self.internal_vm_verifier_commit[i]));
            // At least 1 proof should be provided.
            builder.assert_ne::<Usize<_>>(proofs.len(), RVar::zero());

            let merged_pvs = VmVerifierPvs::<Felt<F>>::uninit(&mut builder);
            let expected_leaf_commit: [Felt<F>; DIGEST_SIZE] = array::from_fn(|_| builder.uninit());
            builder.range(0, proofs.len()).for_each(|i, builder| {
                let proof = builder.get(&proofs, i);
                StarkVerifier::verify::<DuplexChallengerVariable<C>>(
                    builder, &pcs, &m_advice, &proof,
                );
                assert_single_segment_vm_exit_successfully(builder, &proof);

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

                let program_commit = get_program_commit(builder, &proof);
                let is_internal =
                    eq_felt_slice(builder, &program_commit, &internal_vm_verifier_commit);
                let proof_leaf_commit: [Felt<_>; DIGEST_SIZE] = builder.uninit();
                builder.if_eq(is_internal, F::ONE).then_or_else(
                    |builder| {
                        // assert self_program_commit == program_commit
                        builder.assert_eq::<[_; DIGEST_SIZE]>(
                            proof_vm_pvs.self_program_commit,
                            program_commit,
                        );
                        builder.assign(&proof_leaf_commit, proof_vm_pvs.leaf_verifier_commit);
                    },
                    |builder| {
                        // Treat this as a leaf verifier proof.
                        builder.assign(&proof_leaf_commit, program_commit);
                    },
                );
                builder.if_eq(i, RVar::zero()).then_or_else(
                    |builder| {
                        builder.assign(
                            &merged_pvs.app_commit,
                            proof_vm_pvs.vm_verifier_pvs.app_commit,
                        );
                        builder.assign(&expected_leaf_commit, proof_leaf_commit);
                    },
                    |builder| {
                        builder.assert_eq::<[_; DIGEST_SIZE]>(
                            merged_pvs.app_commit,
                            // If this is a leaf verifier proof which logical public values don't
                            // have `leaf_verifier_commit`/`self_program_commit`, `app_commit` is
                            // still in the same position.
                            proof_vm_pvs.vm_verifier_pvs.app_commit,
                        );
                        builder
                            .assert_eq::<[_; DIGEST_SIZE]>(expected_leaf_commit, proof_leaf_commit);
                    },
                );

                assert_or_assign_connector_pvs(
                    builder,
                    &merged_pvs.connector,
                    i,
                    &proof_vm_pvs.vm_verifier_pvs.connector,
                );
                assert_or_assign_memory_pvs(
                    builder,
                    &merged_pvs.memory,
                    i,
                    &proof_vm_pvs.vm_verifier_pvs.memory,
                );
                // This is only needed when `is_terminate` but branching here won't save much, so we
                // always assign it.
                builder.assign(
                    &merged_pvs.public_values_commit,
                    proof_vm_pvs.vm_verifier_pvs.public_values_commit,
                );
            });
            // App Program should terminate
            builder.assert_felt_eq(merged_pvs.connector.is_terminate, F::ONE);
            // App Program should exit successfully
            builder.assert_felt_eq(merged_pvs.connector.exit_code, F::ZERO);

            builder.assert_eq::<Usize<_>>(public_values.len(), RVar::from(self.num_public_values));
            let public_values_vec: Vec<Felt<F>> = (0..self.num_public_values)
                .map(|i| builder.get(&public_values, i))
                .collect();
            let hasher = VariableP2Hasher::new(&mut builder);
            let pv_commit = hasher.merkle_root(&mut builder, &public_values_vec);
            builder.assert_eq::<[_; DIGEST_SIZE]>(merged_pvs.public_values_commit, pv_commit);

            let pvs = RootVmVerifierPvs {
                exe_commit: compute_exe_commit(
                    &mut builder,
                    &hasher,
                    merged_pvs.app_commit,
                    merged_pvs.memory.initial_root,
                    merged_pvs.connector.initial_pc,
                ),
                leaf_verifier_commit: expected_leaf_commit,
                public_values: public_values_vec,
            };
            pvs.flatten()
                .into_iter()
                .for_each(|v| builder.commit_public_value(v));

            builder.halt();
        }

        builder.compile_isa_with_options(self.compiler_options.clone())
    }
}

fn compute_exe_commit<C: Config>(
    builder: &mut Builder<C>,
    hasher: &VariableP2Hasher<C>,
    app_commit: [Felt<C::F>; DIGEST_SIZE],
    init_memory: [Felt<C::F>; DIGEST_SIZE],
    pc_start: Felt<C::F>,
) -> [Felt<C::F>; DIGEST_SIZE] {
    let app_commit_hash = hasher.hash(builder, &app_commit);
    let init_memory_hash = hasher.hash(builder, &init_memory);
    let const_zero = hasher.const_zero;
    let padded_pc_start = array::from_fn(|i| if i == 0 { pc_start } else { const_zero });
    let pc_start_hash = hasher.hash(builder, &padded_pc_start);
    let compress_1 = hasher
        .compressor
        .compress(builder, &app_commit_hash, &init_memory_hash);
    hasher
        .compressor
        .compress(builder, &compress_1, &pc_start_hash)
}
