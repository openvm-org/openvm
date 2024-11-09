use std::array;

use ax_stark_sdk::{
    ax_stark_backend::{keygen::types::MultiStarkVerifyingKey, p3_field::AbstractField},
    config::FriParameters,
};
use axvm_circuit::arch::instructions::program::Program;
use axvm_native_compiler::{conversion::CompilerOptions, prelude::*};
use axvm_recursion::{
    fri::TwoAdicFriPcsVariable, hints::Hintable, types::new_from_inner_multi_vk,
    utils::const_fri_config,
};

use crate::{
    verifier::{
        common::non_leaf::NonLeafVerifierVariables,
        root::{
            types::{RootVmVerifierInput, RootVmVerifierPvs},
            vars::RootVmVerifierInputVariable,
        },
        utils::VariableP2Hasher,
    },
    C, F, SC,
};

pub mod types;
mod vars;

/// Config to generate Root VM verifier program.
pub struct RootVmVerifierConfig {
    pub leaf_fri_params: FriParameters,
    pub internal_fri_params: FriParameters,
    pub num_public_values: usize,
    pub internal_vm_verifier_commit: [F; DIGEST_SIZE],
    pub compiler_options: CompilerOptions,
}
impl RootVmVerifierConfig {
    pub fn build_program(
        &self,
        leaf_vm_vk: &MultiStarkVerifyingKey<SC>,
        internal_vm_vk: &MultiStarkVerifyingKey<SC>,
    ) -> Program<F> {
        let leaf_advice = new_from_inner_multi_vk(leaf_vm_vk);
        let internal_advice = new_from_inner_multi_vk(internal_vm_vk);
        let mut builder = Builder::<C>::default();

        {
            let RootVmVerifierInputVariable {
                proofs,
                public_values,
            } = RootVmVerifierInput::<SC>::read(&mut builder);
            let leaf_pcs = TwoAdicFriPcsVariable {
                config: const_fri_config(&mut builder, &self.leaf_fri_params),
            };
            let internal_pcs = TwoAdicFriPcsVariable {
                config: const_fri_config(&mut builder, &self.leaf_fri_params),
            };
            let internal_program_commit =
                array::from_fn(|i| builder.eval(self.internal_vm_verifier_commit[i]));
            let non_leaf_verifier = NonLeafVerifierVariables {
                internal_program_commit,
                leaf_pcs,
                leaf_advice,
                internal_pcs,
                internal_advice,
            };
            let (merged_pvs, expected_leaf_commit) =
                non_leaf_verifier.verify_internal_or_leaf_verifier_proofs(&mut builder, &proofs);

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
