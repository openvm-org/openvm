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

use super::{DeferralChildVkKind, DeferralInnerProver};
use crate::{
    circuit::deferral::aggregation::inner::{DeferralInnerPreCtx, DeferralInnerTraceGen},
    SC,
};

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC>,
        T: DeferralInnerTraceGen<PB>,
    > DeferralInnerProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: DeferralChildVkKind,
        child_merkle_depth: Option<usize>,
    ) -> ProvingContext<PB> {
        assert!(proofs.len() <= self.circuit.verifier_circuit.max_num_proofs());
        assert!((1..=2).contains(&proofs.len()));
        assert!(
            child_merkle_depth.is_some() || proofs.len() == 1,
            "child_merkle_depth=None is only valid for single-proof wrappers"
        );

        let (child_vk, child_dag_commit, child_is_def) = match child_vk_kind {
            DeferralChildVkKind::DeferralCircuit => {
                (&self.child_vk, self.child_vk_pcs_data.clone(), false)
            }
            DeferralChildVkKind::DeferralAggregation => {
                (&self.child_vk, self.child_vk_pcs_data.clone(), true)
            }
            DeferralChildVkKind::RecursiveSelf => {
                (&self.vk, self.self_vk_pcs_data.clone().unwrap(), true)
            }
        };

        let DeferralInnerPreCtx {
            verifier_pvs_ctx,
            def_pvs_ctx,
            input_ctx,
            poseidon2_inputs,
        } = self.agg_node_tracegen.pre_verifier_subcircuit_tracegen(
            proofs,
            child_is_def,
            child_dag_commit.commitment,
            child_merkle_depth,
        );
        let cached_trace_ctx = CachedTraceCtx::PcsData(child_dag_commit);

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
                child_vk,
                cached_trace_ctx,
                proofs,
                &mut external_data,
                default_duplex_sponge_recorder(),
            )
            .unwrap();

        ProvingContext {
            per_trace: once(verifier_pvs_ctx)
                .chain(once(def_pvs_ctx))
                .chain(once(input_ctx))
                .chain(subcircuit_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }
}
