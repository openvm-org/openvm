use openvm_circuit::arch::PROGRAM_CACHED_TRACE_INDEX;
use openvm_native_compiler::prelude::*;
use openvm_native_recursion::{
    challenger::multi_field32::MultiField32ChallengerVariable,
    config::outer::{new_from_outer_multi_vk, OuterConfig},
    digest::DigestVariable,
    fri::TwoAdicFriPcsVariable,
    halo2::{verifier::Halo2VerifierProvingKey, DslOperations, Halo2Params, Halo2Prover},
    hints::Hintable,
    stark::StarkVerifier,
    utils::const_fri_config,
    vars::StarkProofVariable,
    witness::Witnessable,
};
use openvm_stark_sdk::{
    openvm_stark_backend::{p3_field::FieldAlgebra, proof::Proof},
    p3_bn254_fr::Bn254Fr,
};

use crate::{
    keygen::RootVerifierProvingKey,
    prover::{vm::SingleSegmentVmProver, RootVerifierLocalProver},
    verifier::{
        common::{
            assert_single_segment_vm_exit_successfully_with_connector_air_id, types::SpecialAirIds,
        },
        root::types::{RootVmVerifierInput, RootVmVerifierPvs},
        utils::compress_babybear_var_to_bn254,
    },
    RootSC, F, SC,
};

impl RootVerifierProvingKey {
    /// Keygen the static verifier for this root verifier.
    pub fn keygen_static_verifier(
        &self,
        params: &Halo2Params,
        root_proof: Proof<RootSC>,
        pv_handler: Option<&impl StaticVerifierPvHandler>,
    ) -> Halo2VerifierProvingKey {
        let mut witness = Witness::default();
        root_proof.write(&mut witness);
        let dsl_operations = build_static_verifier_operations(self, &root_proof, pv_handler);
        Halo2VerifierProvingKey {
            pinning: Halo2Prover::keygen(params, dsl_operations.clone(), witness),
            dsl_ops: dsl_operations,
        }
    }

    pub fn generate_dummy_root_proof(&self, dummy_internal_proof: Proof<SC>) -> Proof<RootSC> {
        let prover = RootVerifierLocalProver::new(self.clone());
        // 2 * DIGEST_SIZE for exe_commit and leaf_commit
        let num_public_values = prover
            .root_verifier_pk
            .vm_pk
            .vm_config
            .system
            .num_public_values
            - 2 * DIGEST_SIZE;
        SingleSegmentVmProver::prove(
            &prover,
            RootVmVerifierInput {
                proofs: vec![dummy_internal_proof],
                public_values: vec![F::ZERO; num_public_values],
            }
            .write(),
        )
    }
}

/// Custom public values handler for static verifier. Implement this trait on a struct and pass it in to `RootVerifierProvingKey::keygen_static_verifier`.
/// If this trait is not implemented, `None` should be passed in for pv_handler to use the default handler.
pub trait StaticVerifierPvHandler {
    fn handle_public_values(
        &self,
        builder: &mut Builder<OuterConfig>,
        input: &StarkProofVariable<OuterConfig>,
        root_verifier_pk: &RootVerifierProvingKey,
        special_air_ids: &SpecialAirIds,
    ) -> usize;
}

impl StaticVerifierPvHandler for RootVerifierProvingKey {
    fn handle_public_values(
        &self,
        builder: &mut Builder<OuterConfig>,
        input: &StarkProofVariable<OuterConfig>,
        _root_verifier_pk: &RootVerifierProvingKey,
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

fn build_static_verifier_operations(
    root_verifier_pk: &RootVerifierProvingKey,
    proof: &Proof<RootSC>,
    pv_handler: Option<&impl StaticVerifierPvHandler>,
) -> DslOperations<OuterConfig> {
    let special_air_ids = root_verifier_pk.air_id_permutation().get_special_air_ids();
    let mut builder = Builder::<OuterConfig>::default();
    builder.flags.static_only = true;
    let num_public_values = {
        builder.cycle_tracker_start("VerifierProgram");
        let input = proof.read(&mut builder);
        verify_root_proof(&mut builder, &input, root_verifier_pk, &special_air_ids);

        let num_public_values = match &pv_handler {
            Some(handler) => handler.handle_public_values(
                &mut builder,
                &input,
                root_verifier_pk,
                &special_air_ids,
            ),
            None => root_verifier_pk.handle_public_values(
                &mut builder,
                &input,
                root_verifier_pk,
                &special_air_ids,
            ),
        };
        builder.cycle_tracker_end("VerifierProgram");
        num_public_values
    };
    DslOperations {
        operations: builder.operations,
        num_public_values,
    }
}

fn verify_root_proof(
    builder: &mut Builder<OuterConfig>,
    input: &StarkProofVariable<OuterConfig>,
    root_verifier_pk: &RootVerifierProvingKey,
    special_air_ids: &SpecialAirIds,
) {
    let advice = new_from_outer_multi_vk(&root_verifier_pk.vm_pk.vm_pk.get_vk());
    let pcs = TwoAdicFriPcsVariable {
        config: const_fri_config(builder, &root_verifier_pk.vm_pk.fri_params),
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
        let expected_program_commit: [Bn254Fr; 1] = root_verifier_pk
            .root_committed_exe
            .get_program_commit()
            .into();
        builder.assert_var_eq(commit, expected_program_commit[0]);
    }
    assert_single_segment_vm_exit_successfully_with_connector_air_id(
        builder,
        input,
        special_air_ids.connector_air_id,
    );
}
