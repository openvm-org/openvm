use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{
    F,
    keygen::types::MultiStarkVerifyingKeyV2,
    proof::Proof,
    prover::{AirProvingContextV2, ColMajorMatrix, CpuBackendV2, ProverBackendV2},
};

use crate::system::Preflight;

/// Backend-generic trait to generate a proving context
pub(crate) trait ModuleChip<PB: ProverBackendV2> {
    /// Context needed for trace generation (e.g., VK, proofs, preflights).
    type Ctx<'a>;

    /// Generate an AirProvingContextV2. If required_height is Some(..), then the
    /// resulting trace matrices must have height required_height. This function
    /// should return None iff required_height is defined AND the matrix requires
    /// more than required_height rows.
    fn generate_proving_ctx(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<AirProvingContextV2<PB>>;
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
    pub vk: &'a MultiStarkVerifyingKeyV2,
    pub proofs: &'a [&'a Proof],
    pub preflights: &'a [&'a Preflight],
}

impl<T: RowMajorChip<F>> ModuleChip<CpuBackendV2> for T {
    type Ctx<'a> = T::Ctx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_proving_ctx(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<AirProvingContextV2<CpuBackendV2>> {
        let common_main_rm = self.generate_trace(ctx, required_height);
        common_main_rm
            .map(|m| AirProvingContextV2::simple_no_pis(ColMajorMatrix::from_row_major(&m)))
    }
}

#[cfg(feature = "cuda")]
pub(crate) mod cuda {
    use std::sync::Arc;

    use cuda_backend_v2::GpuBackendV2;
    use openvm_cuda_backend::data_transporter::transport_matrix_to_device;

    use crate::cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu};

    use super::*;

    pub(crate) struct StandardTracegenGpuCtx<'a> {
        pub vk: &'a VerifyingKeyGpu,
        pub proofs: &'a [ProofGpu],
        pub preflights: &'a [PreflightGpu],
    }

    pub(crate) fn generate_gpu_proving_ctx<T: RowMajorChip<F>>(
        t: &T,
        ctx: &T::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<AirProvingContextV2<GpuBackendV2>> {
        let common_main_rm = t.generate_trace(ctx, required_height);
        common_main_rm
            .map(|m| AirProvingContextV2::simple_no_pis(transport_matrix_to_device(Arc::new(m))))
    }
}
