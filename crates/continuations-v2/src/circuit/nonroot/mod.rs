#[cfg(feature = "cuda")]
use cuda_backend_v2::{GpuBackendV2, transport_matrix_h2d_col_major};
use stark_backend_v2::{
    DIGEST_SIZE, F,
    proof::Proof,
    prover::{AirProvingContextV2, CpuBackendV2},
};

use crate::circuit::AggNodeTraceGen;

pub mod app {
    pub use openvm_circuit::arch::{
        CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID, PROGRAM_CACHED_TRACE_INDEX,
        PUBLIC_VALUES_AIR_ID,
    };
}

pub mod receiver;
pub mod verifier;

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
        verifier::generate_proving_ctx(proofs, user_pv_commit, child_vk_commit)
    }

    fn generate_other_proving_ctxs(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
    ) -> Vec<AirProvingContextV2<CpuBackendV2>> {
        vec![receiver::generate_proving_ctx(
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
        let cpu_ctx = verifier::generate_proving_ctx(proofs, user_pv_commit, child_vk_commit);
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
        let cpu_ctx = receiver::generate_proving_ctx(proofs, user_pv_commit.is_some());
        vec![AirProvingContextV2::simple_no_pis(
            transport_matrix_h2d_col_major(&cpu_ctx.common_main).unwrap(),
        )]
    }
}
