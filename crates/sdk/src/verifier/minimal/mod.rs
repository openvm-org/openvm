use openvm_circuit::arch::{instructions::program::Program, SystemConfig};
use openvm_native_compiler::{
    conversion::CompilerOptions,
    ir::{Array, Builder, Felt, RVar, Usize, DIGEST_SIZE},
};
use openvm_native_recursion::{
    challenger::duplex::DuplexChallengerVariable, fri::TwoAdicFriPcsVariable, hints::Hintable,
    stark::StarkVerifier, types::new_from_inner_multi_vk, utils::const_fri_config,
    vars::StarkProofVariable,
};
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey, p3_field::FieldAlgebra, prover::types::Proof,
};
use openvm_stark_sdk::config::{baby_bear_poseidon2::BabyBearPoseidon2Config, FriParameters};
use types::{MinimalVmVerifierInput, MinimalVmVerifierPvs};
use vars::MinimalVmVerifierInputVariable;

use super::{
    common::{
        assert_or_assign_connector_pvs, assert_or_assign_memory_pvs,
        assert_required_air_for_app_vm_present, get_connector_pvs, get_memory_pvs,
        get_program_commit, types::VmVerifierPvs,
    },
    root::{
        compute_exe_commit,
        types::{RootVmVerifierInput, RootVmVerifierPvs},
        vars::RootVmVerifierInputVariable,
    },
    utils::VariableP2Hasher,
};
use crate::{verifier::utils::verify_user_public_values_root, C, F, SC};

pub mod types;
mod vars;

pub struct MinimalVmVerifierConfig {
    pub num_public_values: usize,
    pub app_fri_params: FriParameters,
    pub app_system_config: SystemConfig,
    pub compiler_options: CompilerOptions,
}

impl MinimalVmVerifierConfig {
    pub fn build_program(&self, app_vm_vk: &MultiStarkVerifyingKey<SC>) -> Program<F> {
        println!("MinimalVmVerifierConfig::build_program");
        let m_advice = new_from_inner_multi_vk(app_vm_vk);
        let mut builder = Builder::<C>::default();

        {
            builder.cycle_tracker_start("InitializePcsConst");
            let pcs = TwoAdicFriPcsVariable {
                config: const_fri_config(&mut builder, &self.app_fri_params),
            };
            builder.cycle_tracker_end("InitializePcsConst");

            builder.cycle_tracker_start("ReadProofsFromInput");
            // let MinimalVmVerifierInputVariable {
            //     proof,
            //     public_values,
            // } = MinimalVmVerifierInput::<SC>::read(&mut builder);
            let proof: StarkProofVariable<_> =
                <Proof<BabyBearPoseidon2Config> as Hintable<C>>::read(&mut builder);
            let public_values = VmVerifierPvs::<Felt<F>>::uninit(&mut builder);
            // Only one proof should be provided.
            // builder.assert_eq::<Usize<_>>(proofs.len(), RVar::one());
            // let proof = builder.get(&proof, RVar::zero());
            builder.cycle_tracker_end("ReadProofsFromInput");

            builder.cycle_tracker_start("VerifyProofs");
            assert_required_air_for_app_vm_present(&mut builder, &proof);
            StarkVerifier::verify::<DuplexChallengerVariable<C>>(
                &mut builder,
                &pcs,
                &m_advice,
                &proof,
            );
            {
                let commit = get_program_commit(&mut builder, &proof);
                builder.assign(&public_values.app_commit, commit);
            }

            let proof_connector_pvs = get_connector_pvs(&mut builder, &proof);
            assert_or_assign_connector_pvs(
                &mut builder,
                &public_values.connector,
                RVar::zero(),
                &proof_connector_pvs,
            );

            let proof_memory_pvs = get_memory_pvs(&mut builder, &proof);
            assert_or_assign_memory_pvs(
                &mut builder,
                &public_values.memory,
                RVar::zero(),
                &proof_memory_pvs,
            );
            builder.cycle_tracker_end("VerifyProofs");

            builder.cycle_tracker_start("ExtractPublicValues");
            let is_terminate = builder.cast_felt_to_var(public_values.connector.is_terminate);
            builder.if_eq(is_terminate, F::ONE).then(|builder| {
                let (pv_commit, expected_memory_root) = verify_user_public_values_root(
                    builder,
                    self.app_system_config.num_public_values,
                    self.app_system_config.memory_config,
                );
                builder.assert_eq::<[_; DIGEST_SIZE]>(
                    public_values.memory.final_root,
                    expected_memory_root,
                );
                builder.assign(&public_values.public_values_commit, pv_commit);
            });
            let hasher = VariableP2Hasher::new(&mut builder);
            let public_values_vec: Vec<Felt<F>> = public_values.flatten();
            builder.cycle_tracker_end("ExtractPublicValues");

            let root_pvs = MinimalVmVerifierPvs {
                exe_commit: compute_exe_commit(
                    &mut builder,
                    &hasher,
                    public_values.app_commit,
                    public_values.memory.initial_root,
                    public_values.connector.initial_pc,
                ),
                leaf_verifier_commit: public_values.app_commit,
                public_values: public_values_vec,
            };
            root_pvs
                .flatten()
                .into_iter()
                .for_each(|v| builder.commit_public_value(v));

            builder.halt();
        }

        builder.compile_isa_with_options(self.compiler_options)
    }
}
