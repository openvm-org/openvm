#![allow(clippy::missing_safety_doc)]

use cuda_backend_v2::{EF, F};
use openvm_cuda_backend::chip::UInt2;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

use crate::{
    batch_constraint::{
        cuda_utils::*,
        eq_airs::{RecordIdx, StackedIdxRecord},
    },
    cuda::types::TraceHeight,
};

#[repr(C)]
pub(crate) struct AffineFpExt {
    pub(crate) a: EF,
    pub(crate) b: EF,
}

#[repr(C)]
pub(crate) struct ConstraintsFoldingPerProof {
    pub(crate) lambda_tidx: u32,
    pub(crate) lambda: EF,
}

extern "C" {
    fn _sym_expr_common_tracegen(
        d_trace: *mut F,
        height: usize,
        l_skip: usize,
        d_log_heights: *const usize,
        d_sort_idx_by_air_idx: *const usize,
        num_airs: usize,
        num_proofs: usize,
        max_num_proofs: usize,
        d_expr_evals: *const EF,
        d_ee_bounds_0: *const usize,
        d_ee_bounds_1: *const usize,
        d_constraint_nodes: *const FlatSymbolicConstraintNode,
        d_constraint_nodes_bounds: *const usize,
        d_interactions: *const FlatInteraction,
        d_interactions_bounds: *const usize,
        d_interaction_messages: *const usize,
        d_unused_variables: *const FlatSymbolicVariable,
        d_unused_variables_bounds: *const usize,
        d_record_bounds: *const u32,
        d_air_ids_per_record: *const u32,
        num_records_per_proof: usize,
        d_sumcheck_rnds: *const EF,
        d_sumcheck_bounds: *const usize,
        d_cached_records: *const CachedGpuRecord,
    ) -> i32;

    fn _eq_3b_tracegen(
        d_trace: *mut F,
        num_valid_rows: usize,
        height: usize,
        num_proofs: usize,
        l_skip: usize,
        records: *const StackedIdxRecord,
        record_bounds: *const usize,
        record_idxs: *const RecordIdx,
        record_idxs_bounds: *const usize,
        rows_per_proof_bounds: *const usize,
        n_logups: *const usize,
        xis: *const EF,
        xi_bounds: *const usize,
    ) -> i32;

    fn _constraints_folding_tracegen_temp_bytes(
        d_proof_and_sort_idxs: *const UInt2,
        d_cur_sum_evals: *mut AffineFpExt,
        num_valid_rows: u32,
        temp_bytes_out: *mut usize,
    ) -> i32;

    fn _constraints_folding_tracegen(
        d_trace: *mut F,
        height: usize,
        width: usize,
        d_proof_and_sort_idxs: *const UInt2,
        d_cur_sum_evals: *mut AffineFpExt,
        d_values: *const EF,
        h_row_bounds: *const u32,
        d_constraint_bounds: *const *const u32,
        d_sorted_trace_heights: *const *const TraceHeight,
        d_eq_ns: *const *const EF,
        d_per_proof: *const ConstraintsFoldingPerProof,
        num_proofs: u32,
        num_airs: u32,
        num_valid_rows: u32,
        l_skip: u32,
        d_temp_buffer: *mut core::ffi::c_void,
        temp_bytes: usize,
    ) -> i32;
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn sym_expr_common_tracegen(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    l_skip: usize,
    d_log_heights: &DeviceBuffer<usize>,
    d_sort_idx_by_air_idx: &DeviceBuffer<usize>,
    num_airs: usize,
    num_proofs: usize,
    max_num_proofs: usize,
    d_expr_evals: &DeviceBuffer<EF>,
    d_ee_bounds_0: &DeviceBuffer<usize>,
    d_ee_bounds_1: &DeviceBuffer<usize>,
    d_constraint_nodes: &DeviceBuffer<FlatSymbolicConstraintNode>,
    d_constraint_nodes_bounds: &DeviceBuffer<usize>,
    d_interactions: &DeviceBuffer<FlatInteraction>,
    d_interactions_bounds: &DeviceBuffer<usize>,
    d_interaction_messages: &DeviceBuffer<usize>,
    d_unused_variables: &DeviceBuffer<FlatSymbolicVariable>,
    d_unused_variables_bounds: &DeviceBuffer<usize>,
    d_record_bounds: &DeviceBuffer<u32>,
    d_air_ids_per_record: &DeviceBuffer<u32>,
    num_records_per_proof: usize,
    d_sumcheck_rnds: &DeviceBuffer<EF>,
    d_sumcheck_bounds: &DeviceBuffer<usize>,
    d_cached_records: Option<&DeviceBuffer<CachedGpuRecord>>,
) -> Result<(), CudaError> {
    let cached_records_ptr = d_cached_records.map_or(::core::ptr::null(), |b| b.as_ptr());
    CudaError::from_result(_sym_expr_common_tracegen(
        d_trace.as_mut_ptr(),
        height,
        l_skip,
        d_log_heights.as_ptr(),
        d_sort_idx_by_air_idx.as_ptr(),
        num_airs,
        num_proofs,
        max_num_proofs,
        d_expr_evals.as_ptr(),
        d_ee_bounds_0.as_ptr(),
        d_ee_bounds_1.as_ptr(),
        d_constraint_nodes.as_ptr(),
        d_constraint_nodes_bounds.as_ptr(),
        d_interactions.as_ptr(),
        d_interactions_bounds.as_ptr(),
        d_interaction_messages.as_ptr(),
        d_unused_variables.as_ptr(),
        d_unused_variables_bounds.as_ptr(),
        d_record_bounds.as_ptr(),
        d_air_ids_per_record.as_ptr(),
        num_records_per_proof,
        d_sumcheck_rnds.as_ptr(),
        d_sumcheck_bounds.as_ptr(),
        cached_records_ptr,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn eq_3b_tracegen(
    d_trace: &DeviceBuffer<F>,
    num_valid_rows: usize,
    height: usize,
    num_proofs: usize,
    l_skip: usize,
    d_records: &DeviceBuffer<StackedIdxRecord>,
    d_record_bounds: &DeviceBuffer<usize>,
    d_record_idxs: &DeviceBuffer<RecordIdx>,
    d_record_idxs_bounds: &DeviceBuffer<usize>,
    d_rows_per_proof_bounds: &DeviceBuffer<usize>,
    d_n_logups: &DeviceBuffer<usize>,
    d_xis: &DeviceBuffer<EF>,
    d_xi_bounds: &DeviceBuffer<usize>,
) -> Result<(), CudaError> {
    CudaError::from_result(_eq_3b_tracegen(
        d_trace.as_mut_ptr(),
        num_valid_rows,
        height,
        num_proofs,
        l_skip,
        d_records.as_ptr(),
        d_record_bounds.as_ptr(),
        d_record_idxs.as_ptr(),
        d_record_idxs_bounds.as_ptr(),
        d_rows_per_proof_bounds.as_ptr(),
        d_n_logups.as_ptr(),
        d_xis.as_ptr(),
        d_xi_bounds.as_ptr(),
    ))
}

pub unsafe fn constraints_folding_tracegen_temp_bytes(
    d_proof_and_sort_idxs: &DeviceBuffer<UInt2>,
    d_cur_sum_evals: &DeviceBuffer<AffineFpExt>,
    num_valid_rows: u32,
) -> Result<usize, CudaError> {
    let mut temp_bytes = 0usize;
    CudaError::from_result(_constraints_folding_tracegen_temp_bytes(
        d_proof_and_sort_idxs.as_ptr(),
        d_cur_sum_evals.as_mut_ptr(),
        num_valid_rows,
        &mut temp_bytes as *mut usize,
    ))?;
    Ok(temp_bytes)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn constraints_folding_tracegen(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    width: usize,
    d_proof_and_sort_idxs: &DeviceBuffer<UInt2>,
    d_cur_sum_evals: &DeviceBuffer<AffineFpExt>,
    d_values: &DeviceBuffer<EF>,
    h_row_bounds: &[u32],
    d_constraint_bounds: Vec<*const u32>,
    d_sorted_trace_heights: Vec<*const TraceHeight>,
    d_eq_ns: Vec<*const EF>,
    d_per_proof: &DeviceBuffer<ConstraintsFoldingPerProof>,
    num_proofs: u32,
    num_airs: u32,
    num_valid_rows: u32,
    l_skip: u32,
    d_temp_buffer: &DeviceBuffer<u8>,
    temp_bytes: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_constraints_folding_tracegen(
        d_trace.as_mut_ptr(),
        height,
        width,
        d_proof_and_sort_idxs.as_ptr(),
        d_cur_sum_evals.as_mut_ptr(),
        d_values.as_ptr(),
        h_row_bounds.as_ptr(),
        d_constraint_bounds.as_ptr(),
        d_sorted_trace_heights.as_ptr(),
        d_eq_ns.as_ptr(),
        d_per_proof.as_ptr(),
        num_proofs,
        num_airs,
        num_valid_rows,
        l_skip,
        d_temp_buffer.as_mut_ptr() as *mut core::ffi::c_void,
        temp_bytes,
    ))
}
