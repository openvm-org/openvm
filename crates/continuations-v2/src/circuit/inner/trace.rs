use itertools::Itertools;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{data_transporter::transport_air_proving_ctx_to_device, GpuBackend};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, CpuBackend, ProverBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use verify_stark::pvs::DeferralPvs;

#[derive(Copy, Clone)]
pub enum ProofsType {
    Vm,
    Deferral,
    Mix,
    Combined,
}

// Trait that inner and compression provers use to remain generic in PB
pub trait InnerTraceGen<PB: ProverBackend> {
    fn new(deferral_enabled: bool) -> Self;
    fn generate_pre_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        child_is_app: bool,
        child_dag_commit: PB::Commitment,
    ) -> (Vec<AirProvingContext<PB>>, Vec<[F; POSEIDON2_WIDTH]>);
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
        child_dag_commit: [F; DIGEST_SIZE],
    ) -> (
        Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (verifier_pvs_ctx, mut poseidon2_inputs) = super::verifier::generate_proving_ctx(
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
            poseidon2_inputs.extend_from_slice(&def_poseidon2_inputs);
            def_pvs_ctx
        } else {
            super::unset::generate_proving_ctx(&[], child_is_app)
        };

        (
            vec![verifier_pvs_ctx, vm_pvs_ctx, idx2_ctx],
            poseidon2_inputs,
        )
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
        child_dag_commit: [F; DIGEST_SIZE],
    ) -> (
        Vec<AirProvingContext<GpuBackend>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (cpu_ctxs, poseidon2_inputs) = self.generate_pre_verifier_subcircuit_ctxs(
            proofs,
            proofs_type,
            absent_trace_pvs,
            child_is_app,
            child_dag_commit,
        );
        let gpu_ctxs = cpu_ctxs
            .into_iter()
            .map(transport_air_proving_ctx_to_device)
            .collect_vec();
        (gpu_ctxs, poseidon2_inputs)
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
            .map(transport_air_proving_ctx_to_device)
            .collect_vec()
    }
}
