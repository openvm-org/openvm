use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    proof::Proof,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend, ProverBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};
use p3_matrix::dense::RowMajorMatrix;

use crate::system::Preflight;

/// Backend-generic trait to generate a proving context
pub(crate) trait ModuleChip<PB: ProverBackend> {
    /// Context needed for trace generation (e.g., VK, proofs, preflights).
    type Ctx<'a>;

    /// Generate an AirProvingContext. If required_height is Some(..), then the
    /// resulting trace matrices must have height required_height. This function
    /// should return None iff required_height is defined AND the matrix requires
    /// more than required_height rows.
    fn generate_proving_ctx(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<AirProvingContext<PB>>;
}

/// Trait to generate a CPU row-major common trace
pub(crate) trait RowMajorChip<F> {
    /// Context needed for trace generation (e.g., VK, proofs, preflights).
    type Ctx<'a>;

    /// Generate row major trace with the same semantics as TraceGenerator
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>>;
}

pub(crate) struct StandardTracegenCtx<'a> {
    pub vk: &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    pub proofs: &'a [&'a Proof<BabyBearPoseidon2Config>],
    pub preflights: &'a [&'a Preflight],
}

impl<T: RowMajorChip<F>> ModuleChip<CpuBackend<BabyBearPoseidon2Config>> for T {
    type Ctx<'a> = T::Ctx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_proving_ctx(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>> {
        let common_main_rm = self.generate_trace(ctx, required_height);
        common_main_rm.map(|m| AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&m)))
    }
}

#[cfg(feature = "cuda")]
pub(crate) mod cuda {
    use openvm_cuda_backend::{data_transporter::transport_matrix_h2d_row, GpuBackend};

    use super::*;
    use crate::cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu};

    pub(crate) struct StandardTracegenGpuCtx<'a> {
        pub vk: &'a VerifyingKeyGpu,
        pub proofs: &'a [ProofGpu],
        pub preflights: &'a [PreflightGpu],
    }

    pub(crate) fn generate_gpu_proving_ctx<T: RowMajorChip<F>>(
        t: &T,
        ctx: &T::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<AirProvingContext<GpuBackend>> {
        let common_main_rm = t.generate_trace(ctx, required_height);
        common_main_rm
            .map(|m| AirProvingContext::simple_no_pis(transport_matrix_h2d_row(&m).unwrap()))
    }
}
