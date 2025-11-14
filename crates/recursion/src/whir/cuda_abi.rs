use cuda_backend_v2::{EF, F};
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

use crate::whir::InitialOpenedValueRecord;

extern "C" {
    fn _initial_opened_values_tracegen(
        trace_d: *mut F,
        num_valid_rows: usize,
        height: usize,
        records_d: *const InitialOpenedValueRecord,
        k_whir: usize,
        num_whir_queries: usize,
        num_whir_rounds: usize,
        omega_k: F,
        mus_d: *const EF,
        zi_d: *const F,
        zi_root_d: *const F,
        yi_d: *const EF,
        merkle_idx_bit_src_d: *const F,
        rows_per_proof_psums: *const usize,
        commits_per_proof_psums: *const usize,
        stacking_chunks_psums_per_proof: *const usize,
        stacking_widths_psums_per_proof: *const usize,
        mu_pows: *const EF,
        num_proofs: usize,
    ) -> i32;
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn initial_opened_values_tracegen(
    trace_d: &DeviceBuffer<F>,
    num_valid_rows: usize,
    height: usize,
    records_d: &DeviceBuffer<InitialOpenedValueRecord>,
    k_whir: usize,
    num_whir_queries: usize,
    num_whir_rounds: usize,
    omega_k: F,
    mus_d: &DeviceBuffer<EF>,
    // Flattened across proofs; num_queries per proof
    zi_d: &DeviceBuffer<F>,
    // Flattened across proofs; num_queries per proof
    zi_root_d: &DeviceBuffer<F>,
    // Flattened across proofs; num_queries per proof
    yi_d: &DeviceBuffer<EF>,
    // Flattened across proofs; num_queries per proof
    raw_queries_d: &DeviceBuffer<F>,
    rows_per_proof_psums: &DeviceBuffer<usize>,
    commits_per_proof_psums: &DeviceBuffer<usize>,
    stacking_chunks_psums_per_proof: &DeviceBuffer<usize>,
    stacking_widths_psums_per_proof: &DeviceBuffer<usize>,
    mu_pows: &DeviceBuffer<EF>,
    num_proofs: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_initial_opened_values_tracegen(
        trace_d.as_mut_ptr(),
        num_valid_rows,
        height,
        records_d.as_ptr(),
        k_whir,
        num_whir_queries,
        num_whir_rounds,
        omega_k,
        mus_d.as_ptr(),
        zi_d.as_ptr(),
        zi_root_d.as_ptr(),
        yi_d.as_ptr(),
        raw_queries_d.as_ptr(),
        rows_per_proof_psums.as_ptr(),
        commits_per_proof_psums.as_ptr(),
        stacking_chunks_psums_per_proof.as_ptr(),
        stacking_widths_psums_per_proof.as_ptr(),
        mu_pows.as_ptr(),
        num_proofs,
    ))
}
