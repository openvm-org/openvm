#[cfg(feature = "cuda")]
use openvm_cuda_backend::{data_transporter::transport_air_proving_ctx_to_device, GpuBackend};
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, CpuBackend, ProverBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};

pub mod app {
    pub use openvm_circuit::arch::{
        CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID, PROGRAM_CACHED_TRACE_INDEX,
        PUBLIC_VALUES_AIR_ID,
    };
}

pub mod receiver;
pub mod verifier;

// Trait that non-root and compression provers use to remain generic in PB
pub trait NonRootTraceGen<PB: ProverBackend> {
    fn new() -> Self;
    fn generate_verifier_pvs_ctx(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_app: bool,
        child_dag_commit: PB::Commitment,
    ) -> AirProvingContext<PB>;
    fn generate_other_proving_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_app: bool,
    ) -> Vec<AirProvingContext<PB>>;
}

pub struct NonRootTraceGenImpl;

impl NonRootTraceGen<CpuBackend<BabyBearPoseidon2Config>> for NonRootTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn generate_verifier_pvs_ctx(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_app: bool,
        child_dag_commit: [F; DIGEST_SIZE],
    ) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
        verifier::generate_proving_ctx(proofs, child_is_app, child_dag_commit)
    }

    fn generate_other_proving_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_app: bool,
    ) -> Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>> {
        vec![receiver::generate_proving_ctx(proofs, child_is_app)]
    }
}

#[cfg(feature = "cuda")]
impl NonRootTraceGen<GpuBackend> for NonRootTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn generate_verifier_pvs_ctx(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_app: bool,
        child_dag_commit: [F; DIGEST_SIZE],
    ) -> AirProvingContext<GpuBackend> {
        let cpu_ctx = verifier::generate_proving_ctx(proofs, child_is_app, child_dag_commit);
        transport_air_proving_ctx_to_device(cpu_ctx)
    }

    fn generate_other_proving_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_app: bool,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let cpu_ctx = receiver::generate_proving_ctx(proofs, child_is_app);
        vec![transport_air_proving_ctx_to_device(cpu_ctx)]
    }
}
