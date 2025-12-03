use cuda_backend_v2::F;
use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::memory_manager::MemTracker;
use p3_field::TwoAdicField;
use stark_backend_v2::SystemParams;

use crate::whir::{
    cuda_abi::initial_opened_values_tracegen, cuda_tracegen::WhirBlobGpu,
    initial_opened_values::InitialOpenedValuesCols,
};

#[tracing::instrument(skip_all)]
pub(in crate::whir) fn generate_trace(
    num_proofs: usize,
    blob: &WhirBlobGpu,
    params: SystemParams,
) -> DeviceMatrix<F> {
    let mem = MemTracker::start("tracegen.whir_initial_opened_values");
    let num_valid_rows = blob.initial_opened_values_records.len();
    let omega_k = F::two_adic_generator(params.k_whir);
    let height = num_valid_rows.next_power_of_two();
    let width = InitialOpenedValuesCols::<F>::width();
    let trace_d = DeviceMatrix::with_capacity(height, width);

    unsafe {
        initial_opened_values_tracegen(
            trace_d.buffer(),
            num_valid_rows,
            height,
            &blob.initial_opened_values_records,
            params.k_whir,
            params.num_whir_queries,
            params.num_whir_rounds(),
            omega_k,
            &blob.mus,
            &blob.zis,
            &blob.zi_roots,
            &blob.yis,
            &blob.raw_queries,
            &blob.iov_rows_per_proof_psums,
            &blob.commits_per_proof_psums,
            &blob.stacking_chunks_psums,
            &blob.stacking_widths_psums,
            &blob.mu_pows,
            num_proofs,
        )
        .unwrap();
    }
    mem.emit_metrics();
    trace_d
}
