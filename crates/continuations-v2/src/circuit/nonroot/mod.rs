#[cfg(feature = "cuda")]
use cuda_backend_v2::{GpuBackendV2, transport_air_proving_ctx_to_device};
use stark_backend_v2::{
    DIGEST_SIZE, F,
    proof::Proof,
    prover::{AirProvingContextV2, CpuBackendV2, ProverBackendV2},
};

pub mod app {
    pub use openvm_circuit::arch::{
        CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID, PROGRAM_CACHED_TRACE_INDEX,
        PUBLIC_VALUES_AIR_ID,
    };
}

pub mod receiver;
pub mod verifier;

// Trait that non-root and compression provers use to remain generic in PB
pub trait NonRootTraceGen<PB: ProverBackendV2> {
    fn new() -> Self;
    fn generate_verifier_pvs_ctx(
        &self,
        proofs: &[Proof],
        child_is_app: bool,
        child_dag_commit: PB::Commitment,
    ) -> AirProvingContextV2<PB>;
    fn generate_other_proving_ctxs(
        &self,
        proofs: &[Proof],
        child_is_app: bool,
    ) -> Vec<AirProvingContextV2<PB>>;
}

pub struct NonRootTraceGenImpl;

impl NonRootTraceGen<CpuBackendV2> for NonRootTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn generate_verifier_pvs_ctx(
        &self,
        proofs: &[Proof],
        child_is_app: bool,
        child_dag_commit: [F; DIGEST_SIZE],
    ) -> AirProvingContextV2<CpuBackendV2> {
        verifier::generate_proving_ctx(proofs, child_is_app, child_dag_commit)
    }

    fn generate_other_proving_ctxs(
        &self,
        proofs: &[Proof],
        child_is_app: bool,
    ) -> Vec<AirProvingContextV2<CpuBackendV2>> {
        vec![receiver::generate_proving_ctx(proofs, child_is_app)]
    }
}

#[cfg(feature = "cuda")]
impl NonRootTraceGen<GpuBackendV2> for NonRootTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn generate_verifier_pvs_ctx(
        &self,
        proofs: &[Proof],
        child_is_app: bool,
        child_dag_commit: [F; DIGEST_SIZE],
    ) -> AirProvingContextV2<GpuBackendV2> {
        let cpu_ctx = verifier::generate_proving_ctx(proofs, child_is_app, child_dag_commit);
        transport_air_proving_ctx_to_device(cpu_ctx)
    }

    fn generate_other_proving_ctxs(
        &self,
        proofs: &[Proof],
        child_is_app: bool,
    ) -> Vec<AirProvingContextV2<GpuBackendV2>> {
        let cpu_ctx = receiver::generate_proving_ctx(proofs, child_is_app);
        vec![transport_air_proving_ctx_to_device(cpu_ctx)]
    }
}
