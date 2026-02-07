use cuda_backend_v2::{F, GpuBackendV2};
use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::memory_manager::MemTracker;
use stark_backend_v2::{SystemParams, prover::AirProvingContextV2};

use super::{FinalyPolyQueryEvalCols, compute_round_offsets};
use crate::{
    cuda::to_device_or_nullptr,
    tracegen::ModuleChip,
    whir::{
        cuda_abi::final_poly_query_eval_tracegen, cuda_tracegen::WhirBlobGpu, num_queries_per_round,
    },
};

pub(in crate::whir) struct FinalPolyQueryEvalGpuCtx<'a> {
    pub blob: &'a WhirBlobGpu,
    pub params: &'a SystemParams,
}

pub(in crate::whir) struct FinalPolyQueryEvalGpuTraceGenerator;

impl ModuleChip<GpuBackendV2> for FinalPolyQueryEvalGpuTraceGenerator {
    type Ctx<'a> = FinalPolyQueryEvalGpuCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_proving_ctx(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<AirProvingContextV2<GpuBackendV2>> {
        let blob = ctx.blob;
        let params = ctx.params;

        let mem = MemTracker::start("tracegen.whir_final_poly_query_eval");
        let num_valid_rows = blob.final_poly_query_eval_records.len();
        let height = if let Some(h) = required_height {
            if h < num_valid_rows {
                return None;
            }
            h
        } else {
            num_valid_rows.next_power_of_two()
        };
        let width = FinalyPolyQueryEvalCols::<F>::width();
        let trace_d = DeviceMatrix::with_capacity(height, width);

        let num_queries_per_round = num_queries_per_round(&params);
        let final_poly_len = 1usize << params.log_final_poly_len();
        let round_offsets = compute_round_offsets(
            params.num_whir_rounds(),
            params.k_whir(),
            final_poly_len,
            &num_queries_per_round,
        );
        let rows_per_proof = *round_offsets
            .last()
            .expect("round offsets vector must include sentinel");
        let round_offsets_d = to_device_or_nullptr(&round_offsets).unwrap();
        let num_queries_per_round_d = to_device_or_nullptr(&num_queries_per_round).unwrap();

        unsafe {
            final_poly_query_eval_tracegen(
                trace_d.buffer(),
                num_valid_rows,
                height,
                &blob.final_poly_query_eval_records,
                params.num_whir_rounds(),
                rows_per_proof,
                &round_offsets_d,
                params.log_final_poly_len(),
                &num_queries_per_round_d,
            )
            .unwrap();
        }
        mem.emit_metrics();
        Some(AirProvingContextV2::simple_no_pis(trace_d))
    }
}
