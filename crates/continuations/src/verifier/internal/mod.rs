use openvm_circuit::arch::instructions::program::Program;
use openvm_native_compiler::{conversion::CompilerOptions, prelude::*};
use openvm_native_recursion::{
    fri::TwoAdicFriPcsVariable, hints::Hintable, types::new_from_inner_multi_vk,
    utils::const_fri_config,
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Config, FriParameters},
    openvm_stark_backend::keygen::types::MultiStarkVerifyingKey,
};

use crate::{
    verifier::{
        common::non_leaf::NonLeafVerifierVariables,
        internal::{
            types::{InternalVmVerifierExtraPvs, InternalVmVerifierInput, InternalVmVerifierPvs},
            vars::InternalVmVerifierInputVariable,
        },
    },
    C, F,
};

pub mod types;
pub mod vars;

/// Config to generate internal VM verifier program.
pub struct InternalVmVerifierConfig {
    pub leaf_fri_params: FriParameters,
    pub internal_fri_params: FriParameters,
    pub compiler_options: CompilerOptions,
}

impl InternalVmVerifierConfig {
    pub fn build_program(
        &self,
        leaf_vm_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        internal_vm_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> Program<F> {
        let leaf_advice = new_from_inner_multi_vk(leaf_vm_vk);
        let internal_advice = new_from_inner_multi_vk(internal_vm_vk);
        let mut builder = Builder::<C>::default();
        {
            builder.cycle_tracker_start("ReadProofsFromInput");
            let InternalVmVerifierInputVariable {
                self_program_commit,
                proofs,
            } = InternalVmVerifierInput::<BabyBearPoseidon2Config>::read(&mut builder);
            builder.cycle_tracker_end("ReadProofsFromInput");
            builder.cycle_tracker_start("InitializePcsConst");
            let leaf_pcs = TwoAdicFriPcsVariable {
                config: const_fri_config(&mut builder, &self.leaf_fri_params),
            };
            let internal_pcs = TwoAdicFriPcsVariable {
                config: const_fri_config(&mut builder, &self.internal_fri_params),
            };
            builder.cycle_tracker_end("InitializePcsConst");
            let non_leaf_verifier = NonLeafVerifierVariables {
                internal_program_commit: self_program_commit,
                leaf_pcs,
                leaf_advice,
                internal_pcs,
                internal_advice,
            };
            builder.cycle_tracker_start("VerifyProofs");
            let (vm_verifier_pvs, leaf_verifier_commit) =
                non_leaf_verifier.verify_internal_or_leaf_verifier_proofs(&mut builder, &proofs);
            builder.cycle_tracker_end("VerifyProofs");
            let pvs = InternalVmVerifierPvs {
                vm_verifier_pvs,
                extra_pvs: InternalVmVerifierExtraPvs {
                    internal_program_commit: self_program_commit,
                    leaf_verifier_commit,
                },
            };
            for pv in pvs.flatten() {
                builder.commit_public_value(pv);
            }

            builder.halt();
        }

        builder.compile_isa_with_options(self.compiler_options)
    }
}
