use stark_backend_v2::{
    DIGEST_SIZE, F,
    proof::Proof,
    prover::{AirProvingContextV2, ProverBackendV2},
};

pub mod dag_commit;
pub mod nonroot;
pub mod root;

pub const CONSTRAINT_EVAL_AIR_ID: usize = 1;
pub const CONSTRAINT_EVAL_CACHED_INDEX: usize = 0;

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
