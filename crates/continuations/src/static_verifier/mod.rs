use openvm_circuit::arch::PROGRAM_CACHED_TRACE_INDEX;
use openvm_native_compiler::prelude::*;
use openvm_native_recursion::{
    challenger::multi_field32::MultiField32ChallengerVariable,
    config::outer::{new_from_outer_multi_vk, OuterConfig},
    digest::DigestVariable,
    fri::TwoAdicFriPcsVariable,
    halo2::DslOperations,
    stark::StarkVerifier,
    utils::const_fri_config,
    vars::StarkProofVariable,
    witness::Witnessable,
};
use openvm_stark_backend::keygen::types::MultiStarkVerifyingKey;
use openvm_stark_sdk::{
    config::FriParameters, openvm_stark_backend::proof::Proof, p3_bn254_fr::Bn254Fr,
};

use crate::{
    verifier::{
        common::{
            assert_single_segment_vm_exit_successfully_with_connector_air_id, types::SpecialAirIds,
        },
        root::types::RootVmVerifierPvs,
        utils::compress_babybear_var_to_bn254,
    },
    RootSC,
};
/// Custom public values handler for static verifier.
/// This trait implementation defines what the public values of the
/// final EVM proof will be.
pub trait StaticVerifierPvHandler {
    /// Returns the number of public values, as [Bn254Fr] field elements.
    fn handle_public_values(
        &self,
        builder: &mut Builder<OuterConfig>,
        input: &StarkProofVariable<OuterConfig>,
        special_air_ids: &SpecialAirIds,
    ) -> usize;
}

pub struct DefaultStaticVerifierPvHandler;

impl StaticVerifierPvHandler for DefaultStaticVerifierPvHandler {
    fn handle_public_values(
        &self,
        builder: &mut Builder<OuterConfig>,
        input: &StarkProofVariable<OuterConfig>,
        special_air_ids: &SpecialAirIds,
    ) -> usize {
        let pv_air = builder.get(&input.per_air, special_air_ids.public_values_air_id);
        let public_values: Vec<_> = pv_air
            .public_values
            .vec()
            .into_iter()
            .map(|x| builder.cast_felt_to_var(x))
            .collect();
        let pvs = RootVmVerifierPvs::from_flatten(public_values);
        let exe_commit = compress_babybear_var_to_bn254(builder, pvs.exe_commit);
        let leaf_commit = compress_babybear_var_to_bn254(builder, pvs.leaf_verifier_commit);
        let num_public_values = 2 + pvs.public_values.len();
        builder.static_commit_public_value(0, exe_commit);
        builder.static_commit_public_value(1, leaf_commit);
        for (i, x) in pvs.public_values.into_iter().enumerate() {
            builder.static_commit_public_value(i + 2, x);
        }
        num_public_values
    }
}

/// Config to generate static verifier DSL operations.
pub struct StaticVerifierConfig {
    pub root_verifier_fri_params: FriParameters,
    pub special_air_ids: SpecialAirIds,
    pub root_verifier_program_commit: [Bn254Fr; 1],
}

impl StaticVerifierConfig {
    pub fn build_static_verifier_operations(
        &self,
        root_verifier_vk: &MultiStarkVerifyingKey<RootSC>,
        proof: &Proof<RootSC>,
        pv_handler: &impl StaticVerifierPvHandler,
    ) -> DslOperations<OuterConfig> {
        let mut builder = Builder::<OuterConfig>::default();
        builder.flags.static_only = true;
        let num_public_values = {
            builder.cycle_tracker_start("VerifierProgram");
            let input = proof.read(&mut builder);
            self.verify_root_proof(&mut builder, root_verifier_vk, &input);

            let num_public_values =
                pv_handler.handle_public_values(&mut builder, &input, &self.special_air_ids);
            builder.cycle_tracker_end("VerifierProgram");
            num_public_values
        };
        DslOperations {
            operations: builder.operations,
            num_public_values,
        }
    }

    /// `root_verifier_vk` is the verifying key of the root verifier STARK circuit.
    /// `root_verifier_fri_params` are the FRI parameters used to prove the root
    /// verifier STARK circuit.
    fn verify_root_proof(
        &self,
        builder: &mut Builder<OuterConfig>,
        root_verifier_vk: &MultiStarkVerifyingKey<RootSC>,
        input: &StarkProofVariable<OuterConfig>,
    ) {
        let advice = new_from_outer_multi_vk(root_verifier_vk);
        let pcs = TwoAdicFriPcsVariable {
            config: const_fri_config(builder, &self.root_verifier_fri_params),
        };
        StarkVerifier::verify::<MultiField32ChallengerVariable<_>>(builder, &pcs, &advice, input);
        {
            // Program AIR is the only AIR with a cached trace. The cached trace index doesn't
            // change after reordering.
            let t_id = RVar::from(PROGRAM_CACHED_TRACE_INDEX);
            let commit = builder.get(&input.commitments.main_trace, t_id);
            let commit = if let DigestVariable::Var(commit_arr) = commit {
                builder.get(&commit_arr, 0)
            } else {
                unreachable!()
            };
            builder.assert_var_eq(commit, self.root_verifier_program_commit[0]);
        }
        assert_single_segment_vm_exit_successfully_with_connector_air_id(
            builder,
            input,
            self.special_air_ids.connector_air_id,
        );
    }
}
