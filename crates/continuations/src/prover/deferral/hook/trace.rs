use std::iter::once;

use itertools::Itertools;
use openvm_recursion_circuit::system::{
    AggregationSubCircuit, VerifierExternalData, VerifierTraceGen,
};
use openvm_stark_backend::{
    proof::Proof,
    prover::{ProverBackend, ProvingContext},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, Digest, EF, F,
};
use tracing::instrument;

use super::DeferralHookProver;
use crate::{
    circuit::deferral::aggregation::hook::{
        DeferralHookPreCtx, DeferralHookTraceGen, DeferralIoCommit,
    },
    SC,
};

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC>,
        T: DeferralHookTraceGen<PB>,
    > DeferralHookProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proof: Proof<SC>,
        leaf_children: Vec<DeferralIoCommit<F>>,
    ) -> ProvingContext<PB> {
        let DeferralHookPreCtx {
            verifier_pvs_ctx,
            decommit_ctx,
            onion_ctx,
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
        } = self
            .agg_node_tracegen
            .pre_verifier_subcircuit_tracegen(&proof, leaf_children);

        let range_check_inputs = vec![];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &poseidon2_compress_inputs,
            poseidon2_permute_inputs: &poseidon2_permute_inputs,
            range_check_inputs: &range_check_inputs,
            required_heights: None,
            final_transcript_state: None,
        };

        let proof_slice = &[proof];
        let subcircuit_ctxs = self
            .circuit
            .verifier_circuit
            .generate_proving_ctxs(
                &self.child_vk,
                self.child_vk_pcs_data.clone(),
                proof_slice,
                &mut external_data,
                default_duplex_sponge_recorder(),
            )
            .unwrap();

        ProvingContext {
            per_trace: once(verifier_pvs_ctx)
                .chain(once(decommit_ctx))
                .chain(once(onion_ctx))
                .chain(subcircuit_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }
}
