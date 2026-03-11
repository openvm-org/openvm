use std::sync::Arc;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    prover::{AirProvingContext, ProverBackend},
    AirRef, StarkProtocolConfig,
};
use recursion_circuit::prelude::F;

pub mod deferral;
pub mod inner;
pub mod root;
pub mod subair;

pub const CONSTRAINT_EVAL_CACHED_INDEX: usize = 0;

pub struct SubCircuitTraceData<PB: ProverBackend> {
    pub air_proving_ctxs: Vec<AirProvingContext<PB>>,
    pub poseidon2_compress_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
    pub poseidon2_permute_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
}

pub struct SingleAirTraceData<PB: ProverBackend> {
    pub air_proving_ctx: AirProvingContext<PB>,
    pub poseidon2_compress_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
    pub poseidon2_permute_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
}

// TODO: move to stark-backend-v2
pub trait Circuit<SC: StarkProtocolConfig<F = F>> {
    fn airs(&self) -> Vec<AirRef<SC>>;
}

impl<SC: StarkProtocolConfig<F = F>, C: Circuit<SC>> Circuit<SC> for Arc<C> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        self.as_ref().airs()
    }
}

pub(crate) mod utils {
    use openvm_circuit_primitives::utils::assert_array_eq;
    use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
    use p3_air::AirBuilder;
    use recursion_circuit::utils::assert_zeros;
    use verify_stark::pvs::{DagCommit, VerifierBasePvs};

    pub fn assert_dag_commit_eq<AB: AirBuilder, I1: Into<AB::Expr>, I2: Into<AB::Expr>>(
        builder: &mut AB,
        x: DagCommit<I1>,
        y: DagCommit<I2>,
    ) {
        assert_array_eq(builder, x.cached_commit, y.cached_commit);
        assert_array_eq(builder, x.vk_pre_hash, y.vk_pre_hash);
    }

    pub fn assert_dag_commit_unset<AB: AirBuilder, I1: Into<AB::Expr>>(
        builder: &mut AB,
        x: DagCommit<I1>,
    ) {
        assert_zeros(builder, x.cached_commit);
        assert_zeros(builder, x.vk_pre_hash);
    }

    pub fn vk_commit_components<F: Copy>(pvs: &VerifierBasePvs<F>) -> Vec<[F; DIGEST_SIZE]> {
        vec![
            pvs.app_dag_commit.cached_commit,
            pvs.app_dag_commit.vk_pre_hash,
            pvs.leaf_dag_commit.cached_commit,
            pvs.leaf_dag_commit.vk_pre_hash,
            pvs.internal_for_leaf_dag_commit.cached_commit,
            pvs.internal_for_leaf_dag_commit.vk_pre_hash,
        ]
    }
}
