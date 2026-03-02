use itertools::Itertools;
use openvm_stark_backend::{
    proof::Proof,
    prover::{ProverBackend, ProvingContext},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, Digest, EF, F,
};
use recursion_circuit::system::{
    AggregationSubCircuit, CachedTraceCtx, VerifierExternalData, VerifierTraceGen,
};
use tracing::instrument;

use super::CompressionProver;
use crate::{
    circuit::inner::{InnerTraceGen, ProofsType},
    SC,
};

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: InnerTraceGen<PB>,
    > CompressionProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proof: Proof<SC>,
        proofs_type: ProofsType,
    ) -> ProvingContext<PB> {
        let proof_slice = &[proof];
        let (pre_ctxs, poseidon2_inputs) = self
            .agg_node_tracegen
            .generate_pre_verifier_subcircuit_ctxs(
                proof_slice,
                proofs_type,
                None,
                false,
                self.child_vk_pcs_data.commitment,
            );

        let range_check_inputs = vec![];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &poseidon2_inputs,
            range_check_inputs: &range_check_inputs,
            required_heights: None,
            final_transcript_state: None,
        };

        let subcircuit_ctxs = self
            .circuit
            .verifier_circuit
            .generate_proving_ctxs(
                &self.child_vk,
                CachedTraceCtx::Records(self.cached_trace_record.clone()),
                proof_slice,
                &mut external_data,
                default_duplex_sponge_recorder(),
            )
            .unwrap();

        let post_ctxs = self
            .agg_node_tracegen
            .generate_post_verifier_subcircuit_ctxs(proof_slice, proofs_type, false);

        ProvingContext {
            per_trace: pre_ctxs
                .into_iter()
                .chain(subcircuit_ctxs)
                .chain(post_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }
}
