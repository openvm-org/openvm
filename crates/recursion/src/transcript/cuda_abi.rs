#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::{base::DeviceMatrix, prelude::F};
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};
use openvm_stark_backend::prover::MatrixDimensions;

use crate::cuda::types::MerkleVerifyRecord;

extern "C" {
    fn _poseidon2_tracegen(
        d_trace: *mut F,
        height: usize,
        width: usize,
        d_records: *mut F,
        d_counts: *mut u32,
        num_records: usize,
        sbox_regs: usize,
    ) -> i32;

    fn _poseidon2_deduplicate_records_get_temp_bytes(
        d_records: *mut F,
        d_counts: *mut u32,
        num_records: usize,
        d_num_records: *mut usize,
        h_temp_bytes_out: *mut usize,
    ) -> i32;

    fn _poseidon2_deduplicate_records(
        d_records: *mut F,
        d_counts: *mut u32,
        num_records: usize,
        d_num_records: *mut usize,
        d_temp_storage: *mut std::ffi::c_void,
        temp_storage_bytes: usize,
    ) -> i32;

    fn _merkle_verify_tracegen(
        d_trace: *mut F,
        height: usize,
        width: usize,
        d_records: *const MerkleVerifyRecord,
        num_records: usize,
        d_leaf_hashes: *const F,
        d_siblings: *const F,
        num_leaves: usize,
        k: usize,
        d_poseidon_inputs: *mut F,
        num_valid_rows: usize,
        d_proof_start_rows: *const usize,
        num_proofs: usize,
        d_leaf_scratch: *mut F,
    ) -> i32;
}

pub unsafe fn poseidon2_tracegen(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    width: usize,
    d_records: &DeviceBuffer<F>,
    d_counts: &DeviceBuffer<u32>,
    num_records: usize,
    sbox_regs: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_poseidon2_tracegen(
        d_trace.as_mut_ptr(),
        height,
        width,
        d_records.as_mut_ptr(),
        d_counts.as_mut_ptr(),
        num_records,
        sbox_regs,
    ))
}

pub unsafe fn poseidon2_deduplicate_records_get_temp_bytes(
    d_records: &DeviceBuffer<F>,
    d_counts: &DeviceBuffer<u32>,
    num_records: usize,
    d_num_records: &DeviceBuffer<usize>,
    h_temp_bytes_out: &mut usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_poseidon2_deduplicate_records_get_temp_bytes(
        d_records.as_mut_ptr(),
        d_counts.as_mut_ptr(),
        num_records,
        d_num_records.as_mut_ptr(),
        h_temp_bytes_out,
    ))
}

pub unsafe fn poseidon2_deduplicate_records(
    d_records: &DeviceBuffer<F>,
    d_counts: &DeviceBuffer<u32>,
    num_records: usize,
    d_num_records: &DeviceBuffer<usize>,
    d_temp_storage: &DeviceBuffer<u8>,
    temp_storage_bytes: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_poseidon2_deduplicate_records(
        d_records.as_mut_ptr(),
        d_counts.as_mut_ptr(),
        num_records,
        d_num_records.as_mut_ptr(),
        d_temp_storage.as_mut_ptr() as *mut std::ffi::c_void,
        temp_storage_bytes,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn merkle_verify_tracegen(
    d_trace: &mut DeviceMatrix<F>,
    d_records: &DeviceBuffer<MerkleVerifyRecord>,
    d_leaf_hashes: &DeviceBuffer<F>,
    d_siblings: &DeviceBuffer<F>,
    num_leaves: usize,
    k: usize,
    d_poseidon_inputs: &mut DeviceBuffer<F>,
    num_valid_rows: usize,
    d_proof_start_rows: &DeviceBuffer<usize>,
    num_proofs: usize,
    d_leaf_scratch: &mut DeviceBuffer<F>,
) -> Result<(), CudaError> {
    CudaError::from_result(_merkle_verify_tracegen(
        d_trace.buffer().as_mut_ptr(),
        d_trace.height(),
        d_trace.width(),
        d_records.as_ptr(),
        d_records.len(),
        d_leaf_hashes.as_ptr(),
        d_siblings.as_ptr(),
        num_leaves,
        k,
        d_poseidon_inputs.as_mut_ptr(),
        num_valid_rows,
        d_proof_start_rows.as_ptr(),
        num_proofs,
        d_leaf_scratch.as_mut_ptr(),
    ))
}
