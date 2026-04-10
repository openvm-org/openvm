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
use openvm_verify_stark_host::pvs::{DeferralPvs, VkCommit};
use tracing::instrument;

use super::{ChildVkKind, InnerAggregationProver};
use crate::{
    circuit::inner::{InnerTraceGen, ProofsType},
    SC,
};

impl<S, T> InnerAggregationProver<CpuBackend<SC>, S, T>
where
    S: AggregationSubCircuit + VerifierTraceGen<CpuBackend<SC>, SC>,
    T: InnerTraceGen<CpuBackend<SC>>,
    <CpuBackend<SC> as ProverBackend>::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
    ) -> ProvingContext<CpuBackend<SC>> {
        assert!(proofs.len() <= self.circuit.verifier_circuit.max_num_proofs());

        let (child_vk, child_vk_pcs_data) = match child_vk_kind {
            ChildVkKind::RecursiveSelf => (&self.vk, self.self_vk_pcs_data.clone().unwrap()),
            _ => (&self.child_vk, self.child_vk_pcs_data.clone()),
        };
        let child_is_app = matches!(child_vk_kind, ChildVkKind::App);
        let child_vk_commit = VkCommit {
            cached_commit: child_vk_pcs_data.commitment,
            vk_pre_hash: child_vk.pre_hash,
        };

        let pre_data = self
            .agg_node_tracegen
            .generate_pre_verifier_subcircuit_ctxs(
                proofs,
                proofs_type,
                absent_trace_pvs,
                child_is_app,
                child_vk_commit,
            );

        let range_check_inputs = vec![];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &pre_data.poseidon2_compress_inputs,
            poseidon2_permute_inputs: &pre_data.poseidon2_permute_inputs,
            range_check_inputs: &range_check_inputs,
            required_heights: None,
            final_transcript_state: None,
        };

        let subcircuit_ctxs = self
            .circuit
            .verifier_circuit
            .generate_proving_ctxs(
                child_vk,
                CachedTraceCtx::PcsData(child_vk_pcs_data),
                proofs,
                &mut external_data,
                default_duplex_sponge_recorder(),
            )
            .unwrap();
        let post_ctxs = self
            .agg_node_tracegen
            .generate_post_verifier_subcircuit_ctxs(proofs, proofs_type, child_is_app);

        ProvingContext {
            per_trace: pre_data
                .air_proving_ctxs
                .into_iter()
                .chain(subcircuit_ctxs)
                .chain(post_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }
}

#[cfg(feature = "cuda")]
impl<S, T> InnerAggregationProver<GpuBackend, S, T>
where
    S: AggregationSubCircuit
        + VerifierTraceGen<CpuBackend<SC>, SC>
        + GpuVerifierTraceGen<GpuBackend, SC>,
    T: InnerTraceGen<CpuBackend<SC>>,
    <GpuBackend as ProverBackend>::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        device_ctx: &GpuDeviceCtx,
    ) -> ProvingContext<GpuBackend> {
        assert!(proofs.len() <= self.circuit.verifier_circuit.max_num_proofs());

        let (child_vk, child_vk_pcs_data) = match child_vk_kind {
            ChildVkKind::RecursiveSelf => (&self.vk, self.self_vk_pcs_data.clone().unwrap()),
            _ => (&self.child_vk, self.child_vk_pcs_data.clone()),
        };
        let child_is_app = matches!(child_vk_kind, ChildVkKind::App);
        let child_vk_commit = VkCommit {
            cached_commit: child_vk_pcs_data.commitment,
            vk_pre_hash: child_vk.pre_hash,
        };

        let pre_data = self
            .agg_node_tracegen
            .generate_pre_verifier_subcircuit_ctxs(
                proofs,
                proofs_type,
                absent_trace_pvs,
                child_is_app,
                child_vk_commit,
            );

        let range_check_inputs = vec![];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &pre_data.poseidon2_compress_inputs,
            poseidon2_permute_inputs: &pre_data.poseidon2_permute_inputs,
            range_check_inputs: &range_check_inputs,
            required_heights: None,
            final_transcript_state: None,
        };

        let subcircuit_ctxs = self
            .circuit
            .verifier_circuit
            .generate_proving_ctxs_gpu(
                child_vk,
                CachedTraceCtx::PcsData(child_vk_pcs_data),
                proofs,
                &mut external_data,
                device_ctx,
                default_duplex_sponge_recorder(),
            )
            .unwrap();
        let post_ctxs = self
            .agg_node_tracegen
            .generate_post_verifier_subcircuit_ctxs(proofs, proofs_type, child_is_app)
            .into_iter()
            .map(|air_ctx| cpu_proving_ctx_to_gpu(air_ctx, device_ctx))
            .collect_vec();

        ProvingContext {
            per_trace: pre_data
                .air_proving_ctxs
                .into_iter()
                .map(|air_ctx| cpu_proving_ctx_to_gpu(air_ctx, device_ctx))
                .chain(subcircuit_ctxs)
                .chain(post_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }
}
