#[cfg(feature = "cuda")]
use cuda_backend_v2::{GpuBackendV2, transport_matrix_h2d_col_major};
use stark_backend_v2::{
    DIGEST_SIZE, F,
    proof::Proof,
    prover::{AirProvingContextV2, CpuBackendV2, ProverBackendV2},
};

pub mod dag_commit;
pub mod public_values;

pub trait AggNodeTraceGen<PB: ProverBackendV2> {
    fn new() -> Self;
    fn generate_verifier_pvs_ctx(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
        child_vk_commit: PB::Commitment,
    ) -> AirProvingContextV2<PB>;
    fn generate_other_proving_ctxs(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
    ) -> Vec<AirProvingContextV2<PB>>;
}

pub struct NonRootTraceGen;

impl AggNodeTraceGen<CpuBackendV2> for NonRootTraceGen {
    fn new() -> Self {
        Self
    }

    fn generate_verifier_pvs_ctx(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
        child_vk_commit: [F; DIGEST_SIZE],
    ) -> AirProvingContextV2<CpuBackendV2> {
        public_values::verifier::generate_proving_ctx(proofs, user_pv_commit, child_vk_commit)
    }

    fn generate_other_proving_ctxs(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
    ) -> Vec<AirProvingContextV2<CpuBackendV2>> {
        vec![public_values::receiver::generate_proving_ctx(
            proofs,
            user_pv_commit.is_some(),
        )]
    }
}

#[cfg(feature = "cuda")]
impl AggNodeTraceGen<GpuBackendV2> for NonRootTraceGen {
    fn new() -> Self {
        Self
    }

    fn generate_verifier_pvs_ctx(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
        child_vk_commit: [F; DIGEST_SIZE],
    ) -> AirProvingContextV2<GpuBackendV2> {
        let cpu_ctx =
            public_values::verifier::generate_proving_ctx(proofs, user_pv_commit, child_vk_commit);
        AirProvingContextV2 {
            cached_mains: vec![],
            common_main: transport_matrix_h2d_col_major(&cpu_ctx.common_main).unwrap(),
            public_values: cpu_ctx.public_values,
        }
    }

    fn generate_other_proving_ctxs(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
    ) -> Vec<AirProvingContextV2<GpuBackendV2>> {
        let cpu_ctx =
            public_values::receiver::generate_proving_ctx(proofs, user_pv_commit.is_some());
        vec![AirProvingContextV2::simple_no_pis(
            transport_matrix_h2d_col_major(&cpu_ctx.common_main).unwrap(),
        )]
    }
}
