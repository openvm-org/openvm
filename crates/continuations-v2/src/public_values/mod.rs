#[cfg(feature = "cuda")]
use cuda_backend_v2::{GpuBackendV2, transport_matrix_h2d_col_major};
use stark_backend_v2::{
    DIGEST_SIZE, F,
    proof::Proof,
    prover::{AirProvingContextV2, CpuBackendV2, ProverBackendV2},
};
use stark_recursion_circuit_derive::AlignedBorrow;

pub mod app {
    pub use openvm_circuit::arch::{
        CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID, PROGRAM_CACHED_TRACE_INDEX,
        PUBLIC_VALUES_AIR_ID,
    };
}

pub mod receiver;
pub mod verifier;

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy)]
pub struct NonRootVerifierPvs<F> {
    // app commit pvs
    pub user_pv_commit: [F; DIGEST_SIZE],
    pub program_commit: [F; DIGEST_SIZE],

    // connector pvs
    pub initial_pc: F,
    pub final_pc: F,
    pub exit_code: F,
    pub is_terminate: F,

    // memory merkle pvs
    pub initial_root: [F; DIGEST_SIZE],
    pub final_root: [F; DIGEST_SIZE],

    // verifier-specific pvs
    pub internal_flag: F,
    pub leaf_commit: [F; DIGEST_SIZE],
    pub internal_for_leaf_commit: [F; DIGEST_SIZE],
    pub internal_recursive_commit: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct RootVerifierPvs<F> {
    pub app_commit: [F; DIGEST_SIZE],
    pub leaf_commit: [F; DIGEST_SIZE],
    pub internal_for_leaf_commit: [F; DIGEST_SIZE],
    pub internal_recursive_commit: [F; DIGEST_SIZE],
}

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
