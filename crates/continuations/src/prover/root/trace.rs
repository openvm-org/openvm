use itertools::Itertools;
use openvm_circuit::system::memory::merkle::public_values::UserPublicValuesProof;
use openvm_recursion_circuit::system::{
    AggregationSubCircuit, VerifierExternalData, VerifierTraceGen,
};
use openvm_stark_backend::{
    proof::Proof,
    prover::{ProverBackend, ProvingContext},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, DIGEST_SIZE, EF, F,
};
use tracing::instrument;

use super::RootProver;
use crate::{
    circuit::{deferral::DeferralMerkleProofs, root::RootTraceGen},
    RootSC, SC,
};

impl<
        PB: ProverBackend<Val = F, Challenge = EF>,
        S: AggregationSubCircuit + VerifierTraceGen<PB, RootSC>,
        T: RootTraceGen<PB>,
    > RootProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    fn generate_proving_ctx_internal(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<PB::Val>>,
    ) -> Option<ProvingContext<PB>> {
        assert_eq!(
            user_pvs_proof.public_values.len(),
            self.circuit.num_user_pvs
        );

        // These AIRs should have the same height regardless of proof or user_pvs_proof.
        let mut pre_data = self.agg_node_tracegen.generate_pre_verifier_subcircuit_ctx(
            &proof,
            user_pvs_proof,
            self.circuit.memory_dimensions,
        );
        let (post_verifier_subcircuit_ctxs, other_compress_inputs) =
            self.agg_node_tracegen.generate_other_proving_ctxs(
                &proof,
                self.circuit.memory_dimensions,
                deferral_merkle_proofs,
            );
        pre_data
            .poseidon2_compress_inputs
            .extend(other_compress_inputs);

        let verifier_trace_heights = self.trace_heights.as_ref().map(|v| {
            let num_airs = v.len();
            &v[3..num_airs]
        });

        let range_check_inputs = vec![];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &pre_data.poseidon2_compress_inputs,
            poseidon2_permute_inputs: &pre_data.poseidon2_permute_inputs,
            range_check_inputs: &range_check_inputs,
            required_heights: verifier_trace_heights,
            final_transcript_state: None,
        };

        let subcircuit_ctxs = self.circuit.verifier_circuit.generate_proving_ctxs(
            &self.child_vk,
            self.child_vk_pcs_data.clone(),
            &[proof],
            &mut external_data,
            default_duplex_sponge_recorder(),
        );

        subcircuit_ctxs.map(|subcircuit_ctxs| ProvingContext {
            per_trace: pre_data
                .air_proving_ctxs
                .into_iter()
                .chain(subcircuit_ctxs)
                .chain(post_verifier_subcircuit_ctxs)
                .enumerate()
                .collect_vec(),
        })
    }

    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
    ) -> Option<ProvingContext<PB>> {
        assert!(
            self.circuit.def_hook_commit.is_none(),
            "deferral-enabled root prover requires generate_proving_ctx_with_deferrals"
        );
        self.generate_proving_ctx_internal(proof, user_pvs_proof, None)
    }

    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx_with_deferrals(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
        deferral_merkle_proofs: &DeferralMerkleProofs<PB::Val>,
    ) -> Option<ProvingContext<PB>> {
        self.generate_proving_ctx_internal(proof, user_pvs_proof, Some(deferral_merkle_proofs))
    }
}
