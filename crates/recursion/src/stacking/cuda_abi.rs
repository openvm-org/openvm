#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::{EF, F};
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StackedTraceData {
    pub commit_idx: u32,
    pub start_col_idx: u32,
    pub start_row_idx: u32,
    pub log_height: u32,
    pub width: u32,
    pub need_rot: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StackedSliceData {
    pub commit_idx: u32,
    pub col_idx: u32,
    pub row_idx: u32,
    pub n: i32,
    pub is_last_for_claim: bool,
    pub need_rot: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PolyPrecomputation {
    pub eq_in: EF,
    pub k_rot_in: EF,
    pub eq_bits: EF,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ColumnOpeningClaims {
    pub sort_idx: u32,
    pub part_idx: u32,
    pub col_idx: u32,
    pub col_claim: EF,
    pub rot_claim: EF,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OpeningRecordsPerProof {
    pub tidx_before_column_openings: u32,
    pub last_main_idx: u32,
    pub lambda: EF,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StackingClaim {
    pub commit_idx: u32,
    pub stacked_col_idx: u32,
    pub claim: EF,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ClaimsRecordsPerProof {
    pub initial_tidx: u32,
    pub mu: EF,
    pub mu_pow_witness: F,
    pub mu_pow_sample: F,
}

extern "C" {
    // stacking/utils.cu
    fn _stacked_slice_data(
        d_out: *mut StackedSliceData,
        d_slice_offsets: *const u32,
        d_stacked_trace_data: *const StackedTraceData,
        num_airs: u32,
        num_commits: u32,
        num_slices: u32,
        n_stack: u32,
        l_skip: u32,
    ) -> i32;

    fn _compute_coefficients_temp_bytes(
        d_coeff_terms: *mut EF,
        d_coeff_term_keys: *mut u64,
        d_coeffs: *mut EF,
        d_coeff_keys: *mut u64,
        num_slices: u32,
        d_num_coeffs: *mut usize,
        h_temp_bytes_out: *mut usize,
    ) -> i32;

    fn _compute_coefficients(
        d_coeff_terms: *mut EF,
        d_coeff_term_keys: *mut u64,
        d_coeffs: *mut EF,
        d_coeff_keys: *mut u64,
        d_precomps: *mut PolyPrecomputation,
        d_slice_data: *const StackedSliceData,
        d_u: *const EF,
        d_r: *const EF,
        d_lambda_pows: *const EF,
        num_commits: u32,
        num_slices: u32,
        n_stack: u32,
        l_skip: u32,
        d_temp_buffer: *mut core::ffi::c_void,
        temp_bytes: usize,
        d_num_coeffs: *mut usize,
    ) -> i32;

    // stacking/claims.cu
    fn _stacking_claims_tracegen_temp_bytes(
        d_trace: *mut F,
        height: usize,
        h_temp_bytes_out: *mut usize,
    ) -> i32;

    fn _stacking_claims_tracegen(
        d_trace: *mut F,
        height: usize,
        width: usize,
        h_row_bounds: *const u32,
        d_claims: *const *const StackingClaim,
        d_coeffs: *const *const EF,
        d_mu_pows: *const *const EF,
        d_records: *const ClaimsRecordsPerProof,
        num_proofs: u32,
        d_temp_buffer: *mut core::ffi::c_void,
        temp_bytes: usize,
    ) -> i32;

    // stacking/opening.cu
    fn _opening_claims_tracegen_temp_bytes(
        d_trace: *mut F,
        height: usize,
        d_keys_buffer: *mut F,
        h_temp_bytes_out: *mut usize,
    ) -> i32;

    fn _opening_claims_tracegen(
        d_trace: *mut F,
        height: usize,
        width: usize,
        h_row_bounds: *const u32,
        d_claims: *const *const ColumnOpeningClaims,
        d_slice_data: *const *const StackedSliceData,
        d_precomps: *const *const PolyPrecomputation,
        d_lambda_pows: *const *const EF,
        d_records: *const OpeningRecordsPerProof,
        num_proofs: u32,
        l_skip: u32,
        d_keys_buffer: *mut F,
        d_temp_buffer: *mut core::ffi::c_void,
        temp_bytes: usize,
    ) -> i32;
}

// ============================================================================
// Temp-bytes helpers
// ============================================================================

pub unsafe fn compute_coefficients_temp_bytes(
    d_coeff_terms: &DeviceBuffer<EF>,
    d_coeff_term_keys: &DeviceBuffer<u64>,
    d_coeffs: &DeviceBuffer<EF>,
    d_coeff_keys: &DeviceBuffer<u64>,
    num_slices: u32,
    d_num_coeffs: &DeviceBuffer<usize>,
) -> Result<usize, CudaError> {
    let mut temp_bytes = 0usize;
    CudaError::from_result(_compute_coefficients_temp_bytes(
        d_coeff_terms.as_mut_ptr(),
        d_coeff_term_keys.as_mut_ptr(),
        d_coeffs.as_mut_ptr(),
        d_coeff_keys.as_mut_ptr(),
        num_slices,
        d_num_coeffs.as_mut_ptr(),
        &mut temp_bytes as *mut usize,
    ))?;
    Ok(temp_bytes)
}

pub unsafe fn stacking_claims_tracegen_temp_bytes(
    d_trace: &DeviceBuffer<F>,
    height: usize,
) -> Result<usize, CudaError> {
    let mut temp_bytes = 0usize;
    CudaError::from_result(_stacking_claims_tracegen_temp_bytes(
        d_trace.as_mut_ptr(),
        height,
        &mut temp_bytes as *mut usize,
    ))?;
    Ok(temp_bytes)
}

pub unsafe fn opening_claims_tracegen_temp_bytes(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    d_keys_buffer: &DeviceBuffer<F>,
) -> Result<usize, CudaError> {
    let mut temp_bytes = 0usize;
    CudaError::from_result(_opening_claims_tracegen_temp_bytes(
        d_trace.as_mut_ptr(),
        height,
        d_keys_buffer.as_mut_ptr(),
        &mut temp_bytes as *mut usize,
    ))?;
    Ok(temp_bytes)
}

// ============================================================================
// Launchers
// ============================================================================

#[allow(clippy::too_many_arguments)]
pub unsafe fn stacked_slice_data(
    d_out: &DeviceBuffer<StackedSliceData>,
    d_slice_offsets: &DeviceBuffer<u32>,
    d_stacked_trace_data: &DeviceBuffer<StackedTraceData>,
    num_airs: u32,
    num_commits: u32,
    num_slices: u32,
    n_stack: u32,
    l_skip: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_stacked_slice_data(
        d_out.as_mut_ptr(),
        d_slice_offsets.as_ptr(),
        d_stacked_trace_data.as_ptr(),
        num_airs,
        num_commits,
        num_slices,
        n_stack,
        l_skip,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn compute_coefficients(
    d_coeff_terms: &DeviceBuffer<EF>,
    d_coeff_term_keys: &DeviceBuffer<u64>,
    d_coeffs: &DeviceBuffer<EF>,
    d_coeff_keys: &DeviceBuffer<u64>,
    d_precomps: &DeviceBuffer<PolyPrecomputation>,
    d_slice_data: &DeviceBuffer<StackedSliceData>,
    d_u: &DeviceBuffer<EF>,
    d_r: &DeviceBuffer<EF>,
    d_lambda_pows: &DeviceBuffer<EF>,
    num_commits: u32,
    num_slices: u32,
    n_stack: u32,
    l_skip: u32,
    d_temp_buffer: &DeviceBuffer<u8>,
    temp_bytes: usize,
    d_num_coeffs: &DeviceBuffer<usize>,
) -> Result<(), CudaError> {
    CudaError::from_result(_compute_coefficients(
        d_coeff_terms.as_mut_ptr(),
        d_coeff_term_keys.as_mut_ptr(),
        d_coeffs.as_mut_ptr(),
        d_coeff_keys.as_mut_ptr(),
        d_precomps.as_mut_ptr(),
        d_slice_data.as_ptr(),
        d_u.as_ptr(),
        d_r.as_ptr(),
        d_lambda_pows.as_ptr(),
        num_commits,
        num_slices,
        n_stack,
        l_skip,
        d_temp_buffer.as_mut_ptr() as *mut core::ffi::c_void,
        temp_bytes,
        d_num_coeffs.as_mut_ptr(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn stacking_claims_tracegen(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    width: usize,
    h_row_bounds: &[u32],
    d_claims: Vec<*const StackingClaim>,
    d_coeffs: Vec<*const EF>,
    d_mu_pows: Vec<*const EF>,
    d_records: &DeviceBuffer<ClaimsRecordsPerProof>,
    num_proofs: u32,
    d_temp_buffer: &DeviceBuffer<u8>,
    temp_bytes: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_stacking_claims_tracegen(
        d_trace.as_mut_ptr(),
        height,
        width,
        h_row_bounds.as_ptr(),
        d_claims.as_ptr(),
        d_coeffs.as_ptr(),
        d_mu_pows.as_ptr(),
        d_records.as_ptr(),
        num_proofs,
        d_temp_buffer.as_mut_ptr() as *mut core::ffi::c_void,
        temp_bytes,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn opening_claims_tracegen(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    width: usize,
    h_row_bounds: &[u32],
    d_claims: Vec<*const ColumnOpeningClaims>,
    d_slice_data: Vec<*const StackedSliceData>,
    d_precomps: Vec<*const PolyPrecomputation>,
    d_lambda_pows: Vec<*const EF>,
    d_records: &DeviceBuffer<OpeningRecordsPerProof>,
    num_proofs: u32,
    l_skip: u32,
    d_keys_buffer: &DeviceBuffer<F>,
    d_temp_buffer: &DeviceBuffer<u8>,
    temp_bytes: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_opening_claims_tracegen(
        d_trace.as_mut_ptr(),
        height,
        width,
        h_row_bounds.as_ptr(),
        d_claims.as_ptr(),
        d_slice_data.as_ptr(),
        d_precomps.as_ptr(),
        d_lambda_pows.as_ptr(),
        d_records.as_ptr(),
        num_proofs,
        l_skip,
        d_keys_buffer.as_mut_ptr(),
        d_temp_buffer.as_mut_ptr() as *mut core::ffi::c_void,
        temp_bytes,
    ))
}
