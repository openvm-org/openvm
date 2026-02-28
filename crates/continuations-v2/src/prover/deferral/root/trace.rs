use std::iter::once;

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

use super::DeferralRootProver;
use crate::{
    circuit::deferral::aggregation::root::{
        DeferralIoCommit, DeferralRootPreCtx, DeferralRootTraceGen,
    },
    SC,
};

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: DeferralRootTraceGen<PB>,
    > DeferralRootProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proof: Proof<SC>,
        leaf_children: Vec<DeferralIoCommit<F>>,
    ) -> ProvingContext<PB> {
        let DeferralRootPreCtx {
            verifier_pvs_ctx,
            decommit_ctx,
            onion_ctx,
            poseidon2_inputs,
        } = self
            .agg_node_tracegen
            .pre_verifier_subcircuit_tracegen(&proof, leaf_children);

        let range_check_inputs = vec![];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &poseidon2_inputs,
            range_check_inputs: &range_check_inputs,
            required_heights: None,
            final_transcript_state: None,
        };

        let cached_trace_ctx = CachedTraceCtx::PcsData(self.child_vk_pcs_data.clone());
        let proof_slice = &[proof];
        let subcircuit_ctxs = self
            .circuit
            .verifier_circuit
            .generate_proving_ctxs(
                &self.child_vk,
                cached_trace_ctx,
                proof_slice,
                &mut external_data,
                default_duplex_sponge_recorder(),
            )
            .unwrap();

        ProvingContext {
            per_trace: once(verifier_pvs_ctx)
                .chain(subcircuit_ctxs)
                .chain(once(decommit_ctx))
                .chain(once(onion_ctx))
                .enumerate()
                .collect_vec(),
        }
    }
}
