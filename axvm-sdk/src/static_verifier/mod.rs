mod utils;

use ax_stark_sdk::{
    ax_stark_backend::{
        p3_field::{AbstractField, PrimeField32},
        prover::types::Proof,
    },
    p3_bn254_fr::Bn254Fr,
};
use axvm_circuit::{arch::PROGRAM_CACHED_TRACE_INDEX, prover::SingleSegmentVmProver};
use axvm_native_compiler::prelude::*;
use axvm_recursion::{
    challenger::multi_field32::MultiField32ChallengerVariable,
    config::outer::{new_from_outer_multi_vk, OuterConfig},
    digest::DigestVariable,
    fri::TwoAdicFriPcsVariable,
    halo2::{verifier::Halo2VerifierCircuit, DslOperations, Halo2Prover},
    hints::Hintable,
    stark::StarkVerifier,
    utils::const_fri_config,
    witness::Witnessable,
};
use p3_baby_bear::BabyBear;

use crate::{
    keygen::RootVerifierProvingKey,
    prover::RootVerifierLocalProver,
    verifier::{
        common::assert_single_segment_vm_exit_successfully_with_connector_air_id,
        root::types::{RootVmVerifierInput, RootVmVerifierPvs},
    },
    OuterSC, F, SC,
};

impl RootVerifierProvingKey {
    /// Keygen the static verifier for this root verifier.
    pub fn keygen_static_verifier(
        &self,
        halo2_k: usize,
        dummy_internal_proof: Proof<SC>,
    ) -> Halo2VerifierCircuit {
        let dummy_root_proof = self.generate_dummy_root_proof(dummy_internal_proof);
        let mut witness = Witness::default();
        dummy_root_proof.write(&mut witness);
        let dsl_operations = build_static_verifier_operations(self, &dummy_root_proof);
        Halo2VerifierCircuit {
            pinning: Halo2Prover::keygen(halo2_k, dsl_operations.clone(), witness),
            dsl_ops: dsl_operations,
        }
    }

    fn generate_dummy_root_proof(&self, dummy_internal_proof: Proof<SC>) -> Proof<OuterSC> {
        let prover = RootVerifierLocalProver::new(self.clone());
        // 2 * DIGEST_SIZE for exe_commit and leaf_commit
        let num_public_values =
            prover.root_verifier_pk.vm_pk.vm_config.num_public_values - 2 * DIGEST_SIZE;
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

fn build_static_verifier_operations(
    root_verifier_pk: &RootVerifierProvingKey,
    proof: &Proof<OuterSC>,
) -> DslOperations<OuterConfig> {
    let advice = new_from_outer_multi_vk(&root_verifier_pk.vm_pk.vm_pk.get_vk());
    let special_air_ids = root_verifier_pk.air_id_permutation().get_special_air_ids();
    let mut builder = Builder::<OuterConfig>::default();
    builder.flags.static_only = true;
    let num_public_values = {
        builder.cycle_tracker_start("VerifierProgram");
        let input = proof.read(&mut builder);

        let pcs = TwoAdicFriPcsVariable {
            config: const_fri_config(&mut builder, &root_verifier_pk.vm_pk.fri_params),
        };
        StarkVerifier::verify::<MultiField32ChallengerVariable<_>>(
            &mut builder,
            &pcs,
            &advice,
            &input,
        );
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
            &mut builder,
            &input,
            special_air_ids.connector_air_id,
        );

        let pv_air = builder.get(&input.per_air, special_air_ids.public_values_air_id);
        let public_values: Vec<_> = pv_air
            .public_values
            .vec()
            .into_iter()
            .map(|x| builder.cast_felt_to_var(x))
            .collect();
        let pvs = RootVmVerifierPvs::from_flatten(public_values);
        let exe_commit = compress_babybear_var_to_bn254(&mut builder, pvs.exe_commit);
        let leaf_commit = compress_babybear_var_to_bn254(&mut builder, pvs.leaf_verifier_commit);
        let num_public_values = 2 + pvs.public_values.len();
        builder.static_commit_public_value(0, exe_commit);
        builder.static_commit_public_value(1, leaf_commit);
        for (i, x) in pvs.public_values.into_iter().enumerate() {
            builder.static_commit_public_value(i + 2, x);
        }
        builder.cycle_tracker_end("VerifierProgram");
        num_public_values
    };
    DslOperations {
        operations: builder.operations,
        num_public_values,
    }
}

fn compress_babybear_var_to_bn254(
    builder: &mut Builder<OuterConfig>,
    var: [Var<Bn254Fr>; DIGEST_SIZE],
) -> Var<Bn254Fr> {
    let mut ret = SymbolicVar::ZERO;
    let order = Bn254Fr::from_canonical_u32(BabyBear::ORDER_U32);
    let mut base = Bn254Fr::ONE;
    var.iter().for_each(|&x| {
        ret += x * base;
        base *= order;
    });
    builder.eval(ret)
}
