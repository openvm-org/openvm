use std::iter::once;

use itertools::Itertools;
use openvm_circuit::{
    arch::POSEIDON2_WIDTH, system::memory::merkle::public_values::UserPublicValuesProof,
};
#[cfg(feature = "cuda")]
use openvm_circuit_primitives::hybrid_chip::cpu_proving_ctx_to_gpu;
use openvm_continuations::{circuit::deferral::DeferralMerkleProofs, SC};
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::GpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_common::stream::GpuDeviceCtx;
#[cfg(feature = "cuda")]
use openvm_recursion_circuit::system::GpuVerifierTraceGen;
use openvm_recursion_circuit::system::{
    AggregationSubCircuit, CachedTraceCtx, VerifierExternalData, VerifierTraceGen,
};
use openvm_stark_backend::{
    proof::Proof,
    prover::{ProverBackend, ProvingContext},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, DIGEST_SIZE, F,
};
use p3_field::PrimeCharacteristicRing;
use tracing::instrument;

use crate::{prover::DeferredVerifyProver, DeferredVerifyTraceGen, PreVerifierData};

impl<S, T> DeferredVerifyProver<CpuBackend<SC>, S, T>
where
    S: AggregationSubCircuit + VerifierTraceGen<CpuBackend<SC>, SC>,
    T: DeferredVerifyTraceGen<CpuBackend<SC>>,
    <CpuBackend<SC> as ProverBackend>::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
    ) -> ProvingContext<CpuBackend<SC>> {
        assert_eq!(
            user_pvs_proof.public_values.len(),
            self.circuit.num_user_pvs
        );

        let PreVerifierData {
            pre_verifier_ctxs,
            post_verifier_ctxs,
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
            range_inputs,
            verifier_pvs_record,
            output_commit,
        } = self.agg_node_tracegen.pre_verifier_subcircuit_tracegen(
            &proof,
            user_pvs_proof,
            self.circuit.memory_dimensions,
            self.circuit.def_idx,
            deferral_merkle_proofs,
        );

        let mut final_transcript_state = [F::ZERO; POSEIDON2_WIDTH];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &poseidon2_compress_inputs,
            poseidon2_permute_inputs: &poseidon2_permute_inputs,
            range_check_inputs: &range_inputs,
            required_heights: None,
            final_transcript_state: Some(&mut final_transcript_state),
        };

        let proof_slice = &[proof];
        let subcircuit_ctxs = self
            .circuit
            .verifier_circuit
            .generate_proving_ctxs(
                &self.child_vk,
                CachedTraceCtx::PcsData(self.child_vk_pcs_data.clone()),
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
                .chain(post_verifier_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }
}

#[cfg(feature = "cuda")]
impl<S, T> DeferredVerifyProver<GpuBackend, S, T>
where
    S: AggregationSubCircuit
        + VerifierTraceGen<CpuBackend<SC>, SC>
        + GpuVerifierTraceGen<GpuBackend, SC>,
    T: DeferredVerifyTraceGen<CpuBackend<SC>>,
    <GpuBackend as ProverBackend>::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
        device_ctx: &GpuDeviceCtx,
    ) -> ProvingContext<GpuBackend> {
        assert_eq!(
            user_pvs_proof.public_values.len(),
            self.circuit.num_user_pvs
        );

        let PreVerifierData {
            pre_verifier_ctxs,
            post_verifier_ctxs,
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
            range_inputs,
            verifier_pvs_record,
            output_commit,
        } = self.agg_node_tracegen.pre_verifier_subcircuit_tracegen(
            &proof,
            user_pvs_proof,
            self.circuit.memory_dimensions,
            self.circuit.def_idx,
            deferral_merkle_proofs,
        );

        let mut final_transcript_state = [F::ZERO; POSEIDON2_WIDTH];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &poseidon2_compress_inputs,
            poseidon2_permute_inputs: &poseidon2_permute_inputs,
            range_check_inputs: &range_inputs,
            required_heights: None,
            final_transcript_state: Some(&mut final_transcript_state),
        };

        let proof_slice = &[proof];
        let subcircuit_ctxs = self
            .circuit
            .verifier_circuit
            .generate_proving_ctxs_gpu(
                &self.child_vk,
                CachedTraceCtx::PcsData(self.child_vk_pcs_data.clone()),
                proof_slice,
                &mut external_data,
                device_ctx,
                default_duplex_sponge_recorder(),
            )
            .unwrap();

        let verifier_pvs_ctx = cpu_proving_ctx_to_gpu(
            self.agg_node_tracegen.generate_verifier_pvs_ctx(
                &proof_slice[0],
                verifier_pvs_record,
                final_transcript_state,
                output_commit,
            ),
            device_ctx,
        );

        ProvingContext {
            per_trace: once(verifier_pvs_ctx)
                .chain(pre_verifier_ctxs.map(|air_ctx| cpu_proving_ctx_to_gpu(air_ctx, device_ctx)))
                .chain(subcircuit_ctxs)
                .chain(
                    post_verifier_ctxs
                        .into_iter()
                        .map(|air_ctx| cpu_proving_ctx_to_gpu(air_ctx, device_ctx)),
                )
                .enumerate()
                .collect_vec(),
        }
    }
}
