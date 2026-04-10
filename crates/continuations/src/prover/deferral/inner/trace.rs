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
use openvm_stark_sdk::config::baby_bear_poseidon2::default_duplex_sponge_recorder;
use openvm_verify_stark_host::pvs::VkCommit;
use tracing::instrument;

use super::{DeferralChildVkKind, DeferralInnerProver};
use crate::{
    circuit::deferral::inner::{DeferralInnerPreCtx, DeferralInnerTraceGen},
    SC,
};

impl<S, T> DeferralInnerProver<CpuBackend<SC>, S, T>
where
    S: AggregationSubCircuit + VerifierTraceGen<CpuBackend<SC>, SC>,
    T: DeferralInnerTraceGen<CpuBackend<SC>>,
    <CpuBackend<SC> as ProverBackend>::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: DeferralChildVkKind,
        child_merkle_depth: Option<usize>,
    ) -> ProvingContext<CpuBackend<SC>> {
        assert!(proofs.len() <= self.circuit.verifier_circuit.max_num_proofs());
        assert!((1..=2).contains(&proofs.len()));
        assert!(
            child_merkle_depth.is_some() || proofs.len() == 1,
            "child_merkle_depth=None is only valid for single-proof wrappers"
        );

        let (child_vk, child_vk_pcs_data, child_is_agg) = match child_vk_kind {
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
        let child_vk_commit = VkCommit {
            cached_commit: child_vk_pcs_data.commitment,
            vk_pre_hash: child_vk.pre_hash,
        };

        let DeferralInnerPreCtx {
            verifier_pvs_ctx,
            def_pvs_ctx,
            input_ctx,
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
        } = self.agg_node_tracegen.pre_verifier_subcircuit_tracegen(
            proofs,
            child_is_agg,
            child_vk_commit,
            child_merkle_depth,
        );

        let range_check_inputs = vec![];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &poseidon2_compress_inputs,
            poseidon2_permute_inputs: &poseidon2_permute_inputs,
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

#[cfg(feature = "cuda")]
impl<S, T> DeferralInnerProver<GpuBackend, S, T>
where
    S: AggregationSubCircuit
        + VerifierTraceGen<CpuBackend<SC>, SC>
        + GpuVerifierTraceGen<GpuBackend, SC>,
    T: DeferralInnerTraceGen<CpuBackend<SC>>,
    <GpuBackend as ProverBackend>::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: DeferralChildVkKind,
        child_merkle_depth: Option<usize>,
        device_ctx: &GpuDeviceCtx,
    ) -> ProvingContext<GpuBackend> {
        assert!(proofs.len() <= self.circuit.verifier_circuit.max_num_proofs());
        assert!((1..=2).contains(&proofs.len()));
        assert!(
            child_merkle_depth.is_some() || proofs.len() == 1,
            "child_merkle_depth=None is only valid for single-proof wrappers"
        );

        let (child_vk, child_vk_pcs_data, child_is_agg) = match child_vk_kind {
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
        let child_vk_commit = VkCommit {
            cached_commit: child_vk_pcs_data.commitment,
            vk_pre_hash: child_vk.pre_hash,
        };

        let DeferralInnerPreCtx {
            verifier_pvs_ctx,
            def_pvs_ctx,
            input_ctx,
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
        } = self.agg_node_tracegen.pre_verifier_subcircuit_tracegen(
            proofs,
            child_is_agg,
            child_vk_commit,
            child_merkle_depth,
        );

        let range_check_inputs = vec![];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &poseidon2_compress_inputs,
            poseidon2_permute_inputs: &poseidon2_permute_inputs,
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

        ProvingContext {
            per_trace: once(cpu_proving_ctx_to_gpu(verifier_pvs_ctx, device_ctx))
                .chain(once(cpu_proving_ctx_to_gpu(def_pvs_ctx, device_ctx)))
                .chain(once(cpu_proving_ctx_to_gpu(input_ctx, device_ctx)))
                .chain(subcircuit_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }
}
