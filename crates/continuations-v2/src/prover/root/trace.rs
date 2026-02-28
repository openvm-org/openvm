use std::iter::once;

use itertools::Itertools;
use openvm_circuit::system::memory::merkle::public_values::UserPublicValuesProof;
use openvm_stark_backend::{
    proof::Proof,
    prover::{ProverBackend, ProvingContext},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, Digest, DIGEST_SIZE, EF, F,
};
use recursion_circuit::system::{
    AggregationSubCircuit, CachedTraceCtx, VerifierExternalData, VerifierTraceGen,
};
use tracing::instrument;

use super::RootProver;
use crate::{circuit::root::RootTraceGen, SC};

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: RootTraceGen<PB>,
    > RootProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
    ) -> Option<ProvingContext<PB>> {
        assert_eq!(
            user_pvs_proof.public_values.len(),
            self.circuit.num_user_pvs
        );

        // These AIRs should have the same height regardless of proof or user_pvs_proof
        let (verifier_pvs_ctx, mut poseidon2_inputs) =
            self.agg_node_tracegen.generate_verifier_pvs_ctx(&proof);
        let (agg_other_ctxs, other_inputs) = self
            .agg_node_tracegen
            .generate_other_proving_ctxs(user_pvs_proof, self.circuit.memory_dimensions);
        poseidon2_inputs.extend(other_inputs);

        let verifier_trace_heights = self.trace_heights.as_ref().map(|v| {
            let num_airs = v.len();
            &v[1..(num_airs - agg_other_ctxs.len())]
        });

        let range_check_inputs = vec![];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &poseidon2_inputs,
            range_check_inputs: &range_check_inputs,
            required_heights: verifier_trace_heights,
            final_transcript_state: None,
        };

        let cached_trace_ctx = CachedTraceCtx::PcsData(self.child_vk_pcs_data.clone());
        let subcircuit_ctxs = self.circuit.verifier_circuit.generate_proving_ctxs(
            &self.child_vk,
            cached_trace_ctx,
            &[proof],
            &mut external_data,
            default_duplex_sponge_recorder(),
        );

        subcircuit_ctxs.map(|subcircuit_ctxs| ProvingContext {
            per_trace: once(verifier_pvs_ctx)
                .chain(subcircuit_ctxs)
                .chain(agg_other_ctxs)
                .enumerate()
                .collect_vec(),
        })
    }
}
