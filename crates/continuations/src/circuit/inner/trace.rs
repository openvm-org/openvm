use itertools::Itertools;
#[cfg(feature = "cuda")]
use openvm_circuit_primitives::hybrid_chip::cpu_proving_ctx_to_gpu;
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::GpuBackend;
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, ProverBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};
use openvm_verify_stark_host::pvs::{DagCommit, DeferralPvs};

use crate::circuit::{SingleAirTraceData, SubCircuitTraceData};

#[derive(Copy, Clone)]
pub enum ProofsType {
    Vm,
    Deferral,
    Mix,
    Combined,
}

// Trait that inner provers use to remain generic in PB
pub trait InnerTraceGen<PB: ProverBackend> {
    fn new(deferral_enabled: bool) -> Self;
    fn generate_pre_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        child_is_app: bool,
        child_dag_commit: DagCommit<F>,
    ) -> SubCircuitTraceData<PB>;
    fn generate_post_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        child_is_app: bool,
    ) -> Vec<AirProvingContext<PB>>;
}

pub struct InnerTraceGenImpl {
    pub deferral_enabled: bool,
}

impl InnerTraceGen<CpuBackend<BabyBearPoseidon2Config>> for InnerTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn generate_pre_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        child_is_app: bool,
        child_dag_commit: DagCommit<F>,
    ) -> SubCircuitTraceData<CpuBackend<BabyBearPoseidon2Config>> {
        let SingleAirTraceData {
            air_proving_ctx: verifier_pvs_ctx,
            mut poseidon2_compress_inputs,
            poseidon2_permute_inputs,
        } = super::verifier::generate_proving_ctx(
            proofs,
            proofs_type,
            child_is_app,
            child_dag_commit,
            self.deferral_enabled,
        );
        let vm_pvs_ctx = super::vm_pvs::generate_proving_ctx(
            proofs,
            proofs_type,
            child_is_app,
            self.deferral_enabled,
        );

        let idx2_ctx = if self.deferral_enabled {
            let (def_pvs_ctx, def_poseidon2_inputs) = super::def_pvs::generate_proving_ctx(
                proofs,
                proofs_type,
                child_is_app,
                absent_trace_pvs,
            );
            poseidon2_compress_inputs.extend_from_slice(&def_poseidon2_inputs);
            def_pvs_ctx
        } else {
            super::unset::generate_proving_ctx(&[], child_is_app)
        };

        SubCircuitTraceData {
            air_proving_ctxs: vec![verifier_pvs_ctx, vm_pvs_ctx, idx2_ctx],
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
        }
    }

    fn generate_post_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        child_is_app: bool,
    ) -> Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>> {
        if !self.deferral_enabled {
            return vec![];
        }

        let (vm_unset, def_unset) = match proofs_type {
            ProofsType::Vm => (
                vec![],
                proofs.iter().enumerate().map(|(i, _)| i).collect_vec(),
            ),
            ProofsType::Deferral => (
                proofs.iter().enumerate().map(|(i, _)| i).collect_vec(),
                vec![],
            ),
            ProofsType::Mix => (vec![1], vec![0]),
            ProofsType::Combined => (vec![], vec![]),
        };
        vec![
            super::unset::generate_proving_ctx(&vm_unset, child_is_app),
            super::unset::generate_proving_ctx(&def_unset, child_is_app),
        ]
    }
}

#[cfg(feature = "cuda")]
impl InnerTraceGen<GpuBackend> for InnerTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn generate_pre_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        child_is_app: bool,
        child_dag_commit: DagCommit<F>,
    ) -> SubCircuitTraceData<GpuBackend> {
        let data = self.generate_pre_verifier_subcircuit_ctxs(
            proofs,
            proofs_type,
            absent_trace_pvs,
            child_is_app,
            child_dag_commit,
        );
        SubCircuitTraceData {
            air_proving_ctxs: data
                .air_proving_ctxs
                .into_iter()
                .map(cpu_proving_ctx_to_gpu)
                .collect_vec(),
            poseidon2_compress_inputs: data.poseidon2_compress_inputs,
            poseidon2_permute_inputs: data.poseidon2_permute_inputs,
        }
    }

    fn generate_post_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        child_is_app: bool,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let cpu_ctxs =
            self.generate_post_verifier_subcircuit_ctxs(proofs, proofs_type, child_is_app);
        cpu_ctxs
            .into_iter()
            .map(cpu_proving_ctx_to_gpu)
            .collect_vec()
    }
}
