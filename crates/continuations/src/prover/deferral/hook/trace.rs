use std::iter::once;

use itertools::Itertools;
#[cfg(feature = "cuda")]
use openvm_circuit_primitives::hybrid_chip::cpu_proving_ctx_to_gpu;
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
use openvm_stark_sdk::config::baby_bear_poseidon2::{default_duplex_sponge_recorder, F};
use tracing::instrument;

use super::DeferralHookProver;
use crate::{
    circuit::deferral::hook::{DeferralHookPreCtx, DeferralHookTraceGen, DeferralIoCommit},
    SC,
};

impl<S, T> DeferralHookProver<CpuBackend<SC>, S, T>
where
    S: AggregationSubCircuit + VerifierTraceGen<CpuBackend<SC>, SC>,
    T: DeferralHookTraceGen<CpuBackend<SC>>,
    <CpuBackend<SC> as ProverBackend>::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proof: Proof<SC>,
        leaf_children: Vec<DeferralIoCommit<F>>,
    ) -> ProvingContext<CpuBackend<SC>> {
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
                CachedTraceCtx::PcsData(self.child_vk_pcs_data.clone()),
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

#[cfg(feature = "cuda")]
impl<S, T> DeferralHookProver<GpuBackend, S, T>
where
    S: AggregationSubCircuit
        + VerifierTraceGen<CpuBackend<SC>, SC>
        + GpuVerifierTraceGen<GpuBackend, SC>,
    T: DeferralHookTraceGen<CpuBackend<SC>>,
    <GpuBackend as ProverBackend>::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proof: Proof<SC>,
        leaf_children: Vec<DeferralIoCommit<F>>,
        device_ctx: &GpuDeviceCtx,
    ) -> ProvingContext<GpuBackend> {
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
            .generate_proving_ctxs_gpu(
                &self.child_vk,
                CachedTraceCtx::PcsData(self.child_vk_pcs_data.clone()),
                proof_slice,
                &mut external_data,
                device_ctx,
                default_duplex_sponge_recorder(),
            )
            .unwrap();

        ProvingContext {
            per_trace: once(cpu_proving_ctx_to_gpu(verifier_pvs_ctx, device_ctx))
                .chain(once(cpu_proving_ctx_to_gpu(decommit_ctx, device_ctx)))
                .chain(once(cpu_proving_ctx_to_gpu(onion_ctx, device_ctx)))
                .chain(subcircuit_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }
}
