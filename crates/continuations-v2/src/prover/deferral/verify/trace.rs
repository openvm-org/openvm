use std::iter::once;

use itertools::Itertools;
use openvm_circuit::{
    arch::POSEIDON2_WIDTH, system::memory::merkle::public_values::UserPublicValuesProof,
};
use openvm_stark_backend::{
    proof::Proof,
    prover::{ProverBackend, ProvingContext},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, Digest, DIGEST_SIZE, EF, F,
};
use p3_field::PrimeCharacteristicRing;
use recursion_circuit::system::{
    AggregationSubCircuit, CachedTraceCtx, VerifierExternalData, VerifierTraceGen,
};
use tracing::instrument;

use super::DeferredVerifyProver;
use crate::{
    circuit::deferral::verify::{DeferredVerifyTraceGen, PreVerifierData},
    SC,
};

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: DeferredVerifyTraceGen<PB>,
    > DeferredVerifyProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
    ) -> ProvingContext<PB> {
        assert_eq!(
            user_pvs_proof.public_values.len(),
            self.circuit.num_user_pvs
        );

        let PreVerifierData {
            pre_verifier_ctxs,
            post_verifier_ctx,
            poseidon2_inputs,
            range_inputs,
            verifier_pvs_record,
            output_commit,
        } = self.agg_node_tracegen.pre_verifier_subcircuit_tracegen(
            &proof,
            user_pvs_proof,
            self.circuit.memory_dimensions,
        );

        let mut final_transcript_state = [F::ZERO; POSEIDON2_WIDTH];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &poseidon2_inputs,
            range_check_inputs: &range_inputs,
            required_heights: None,
            final_transcript_state: Some(&mut final_transcript_state),
        };

        let proof_slice = &[proof];
        let cached_trace_ctx = CachedTraceCtx::PcsData(self.child_vk_pcs_data.clone());
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

        let verifier_pvs_ctx = self.agg_node_tracegen.generate_verifier_pvs_ctx(
            &proof_slice[0],
            verifier_pvs_record,
            final_transcript_state,
            output_commit,
        );

        ProvingContext {
            per_trace: once(verifier_pvs_ctx)
                .chain(pre_verifier_ctxs)
                .chain(subcircuit_ctxs)
                .chain(once(post_verifier_ctx))
                .enumerate()
                .collect_vec(),
        }
    }
}
