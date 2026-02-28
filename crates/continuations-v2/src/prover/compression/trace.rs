use std::iter::once;

use itertools::Itertools;
use openvm_stark_backend::{
    proof::Proof,
    prover::{ProverBackend, ProvingContext},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, Digest, EF, F,
};
use recursion_circuit::system::{AggregationSubCircuit, CachedTraceCtx, VerifierTraceGen};
use tracing::instrument;

use super::CompressionProver;
use crate::{circuit::nonroot::NonRootTraceGen, SC};

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: NonRootTraceGen<PB>,
    > CompressionProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(&self, proof: Proof<SC>) -> ProvingContext<PB> {
        let proof_slice = &[proof];
        let verifier_pvs_ctx = self.agg_node_tracegen.generate_verifier_pvs_ctx(
            proof_slice,
            false,
            self.child_vk_pcs_data.commitment,
        );
        let subcircuit_ctxs = self.circuit.verifier_circuit.generate_proving_ctxs_base(
            &self.child_vk,
            CachedTraceCtx::Records(self.cached_trace_record.clone()),
            proof_slice,
            default_duplex_sponge_recorder(),
        );
        let agg_other_ctxs = self
            .agg_node_tracegen
            .generate_other_proving_ctxs(proof_slice, false);

        ProvingContext {
            per_trace: once(verifier_pvs_ctx)
                .chain(subcircuit_ctxs)
                .chain(agg_other_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }
}
