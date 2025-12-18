use cuda_backend_v2::F;
use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::memory_manager::MemTracker;
use stark_backend_v2::SystemParams;

use super::{FinalyPolyQueryEvalCols, compute_round_offsets};
use crate::{
    cuda::to_device_or_nullptr,
    whir::{
        cuda_abi::final_poly_query_eval_tracegen, cuda_tracegen::WhirBlobGpu, num_queries_per_round,
    },
};

#[tracing::instrument(level = "trace", skip_all)]
pub(in crate::whir) fn generate_trace(
    blob: &WhirBlobGpu,
    params: &SystemParams,
) -> DeviceMatrix<F> {
    let mem = MemTracker::start("tracegen.whir_final_poly_query_eval");
    let num_valid_rows = blob.final_poly_query_eval_records.len();
    let height = num_valid_rows.next_power_of_two();
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
    trace_d
}
