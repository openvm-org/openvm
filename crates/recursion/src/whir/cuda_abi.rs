use openvm_cuda_backend::prelude::{EF, F};
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

pub use crate::system::PoseidonStatePair;
use crate::whir::{final_poly_query_eval::FinalPolyQueryEvalRecord, folding::FoldRecord};

extern "C" {
    fn _initial_opened_values_tracegen(
        trace_d: *mut F,
        num_valid_rows: usize,
        height: usize,
        codeword_value_accs_d: *const EF,
        poseidon_states_d: *const PoseidonStatePair,
        k_whir: usize,
        num_initial_queries: usize,
        total_queries: usize,
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

    fn _final_poly_query_eval_tracegen(
        trace_d: *mut F,
        num_valid_rows: usize,
        height: usize,
        records_d: *const FinalPolyQueryEvalRecord,
        gammas_d: *const EF,
        num_whir_rounds: usize,
        rows_per_proof: usize,
        round_offsets_d: *const usize,
        log_final_poly_len: usize,
        num_queries_per_round_d: *const usize,
    ) -> i32;

    fn _non_initial_opened_values_tracegen(
        trace_d: *mut F,
        num_valid_rows: usize,
        height: usize,
        codeword_opened_values_d: *const EF,
        codeword_states_d: *const F,
        num_whir_rounds: usize,
        k_whir: usize,
        omega_k: F,
        zis_d: *const F,
        zi_roots_d: *const F,
        yis_d: *const EF,
        raw_queries_d: *const F,
        round_row_offsets_d: *const usize,
        rows_per_proof: usize,
        query_offsets_d: *const usize,
        total_queries: usize,
    ) -> i32;

    fn _whir_folding_tracegen(
        trace_d: *mut F,
        num_valid_rows: u32,
        height: u32,
        records_d: *const FoldRecord,
        num_rounds: u32,
        total_queries: u32,
        k_whir: u32,
    ) -> i32;
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn initial_opened_values_tracegen(
    trace_d: &DeviceBuffer<F>,
    num_valid_rows: usize,
    height: usize,
    codeword_value_accs_d: &DeviceBuffer<EF>,
    poseidon_states_d: &DeviceBuffer<PoseidonStatePair>,
    k_whir: usize,
    num_initial_queries: usize,
    total_queries: usize,
    omega_k: F,
    mus_d: &DeviceBuffer<EF>,
    // Flattened across proofs; total_queries per proof
    zi_d: &DeviceBuffer<F>,
    // Flattened across proofs; total_queries per proof
    zi_root_d: &DeviceBuffer<F>,
    // Flattened across proofs; total_queries per proof
    yi_d: &DeviceBuffer<EF>,
    // Flattened across proofs; total_queries per proof
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
        codeword_value_accs_d.as_ptr(),
        poseidon_states_d.as_ptr(),
        k_whir,
        num_initial_queries,
        total_queries,
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

#[allow(clippy::too_many_arguments)]
pub unsafe fn final_poly_query_eval_tracegen(
    trace_d: &DeviceBuffer<F>,
    num_valid_rows: usize,
    height: usize,
    records_d: &DeviceBuffer<FinalPolyQueryEvalRecord>,
    gammas_d: &DeviceBuffer<EF>,
    num_whir_rounds: usize,
    rows_per_proof: usize,
    round_offsets_d: &DeviceBuffer<usize>,
    log_final_poly_len: usize,
    num_queries_per_round_d: &DeviceBuffer<usize>,
) -> Result<(), CudaError> {
    CudaError::from_result(_final_poly_query_eval_tracegen(
        trace_d.as_mut_ptr(),
        num_valid_rows,
        height,
        records_d.as_ptr(),
        gammas_d.as_ptr(),
        num_whir_rounds,
        rows_per_proof,
        round_offsets_d.as_ptr(),
        log_final_poly_len,
        num_queries_per_round_d.as_ptr(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn non_initial_opened_values_tracegen(
    trace_d: &DeviceBuffer<F>,
    num_valid_rows: usize,
    height: usize,
    codeword_opened_values_d: &DeviceBuffer<EF>,
    codeword_states_d: &DeviceBuffer<F>,
    num_whir_rounds: usize,
    k_whir: usize,
    omega_k: F,
    zis_d: &DeviceBuffer<F>,
    zi_roots_d: &DeviceBuffer<F>,
    yis_d: &DeviceBuffer<EF>,
    raw_queries_d: &DeviceBuffer<F>,
    round_row_offsets_d: &DeviceBuffer<usize>,
    rows_per_proof: usize,
    query_offsets_d: &DeviceBuffer<usize>,
    total_queries: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_non_initial_opened_values_tracegen(
        trace_d.as_mut_ptr(),
        num_valid_rows,
        height,
        codeword_opened_values_d.as_ptr(),
        codeword_states_d.as_ptr(),
        num_whir_rounds,
        k_whir,
        omega_k,
        zis_d.as_ptr(),
        zi_roots_d.as_ptr(),
        yis_d.as_ptr(),
        raw_queries_d.as_ptr(),
        round_row_offsets_d.as_ptr(),
        rows_per_proof,
        query_offsets_d.as_ptr(),
        total_queries,
    ))
}

pub unsafe fn whir_folding_tracegen(
    trace_d: &DeviceBuffer<F>,
    num_valid_rows: u32,
    height: u32,
    records_d: &DeviceBuffer<FoldRecord>,
    num_rounds: u32,
    num_queries: u32,
    k_whir: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_whir_folding_tracegen(
        trace_d.as_mut_ptr(),
        num_valid_rows,
        height,
        records_d.as_ptr(),
        num_rounds,
        num_queries,
        k_whir,
    ))
}
