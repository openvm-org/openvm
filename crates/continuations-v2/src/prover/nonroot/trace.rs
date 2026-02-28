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

use super::{ChildVkKind, NonRootAggregationProver};
use crate::{circuit::nonroot::NonRootTraceGen, SC};

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: NonRootTraceGen<PB>,
    > NonRootAggregationProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
    ) -> ProvingContext<PB> {
        assert!(proofs.len() <= self.circuit.verifier_circuit.max_num_proofs());

        let (child_vk, child_dag_commit) = match child_vk_kind {
            ChildVkKind::RecursiveSelf => (&self.vk, self.self_vk_pcs_data.clone().unwrap()),
            _ => (&self.child_vk, self.child_vk_pcs_data.clone()),
        };
        let child_is_app = matches!(child_vk_kind, ChildVkKind::App);

        let verifier_pvs_ctx = self.agg_node_tracegen.generate_verifier_pvs_ctx(
            proofs,
            child_is_app,
            child_dag_commit.commitment,
        );
        let cached_trace_ctx = CachedTraceCtx::PcsData(child_dag_commit);
        let subcircuit_ctxs = self.circuit.verifier_circuit.generate_proving_ctxs_base(
            child_vk,
            cached_trace_ctx,
            proofs,
            default_duplex_sponge_recorder(),
        );
        let agg_other_ctxs = self
            .agg_node_tracegen
            .generate_other_proving_ctxs(proofs, child_is_app);

        ProvingContext {
            per_trace: once(verifier_pvs_ctx)
                .chain(agg_other_ctxs)
                .chain(subcircuit_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }
}
