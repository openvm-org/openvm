#![allow(clippy::missing_safety_doc)]

use cuda_backend_v2::Digest;
use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

use crate::{
    cuda::types::{AirData, PublicValueData, TraceMetadata},
    proof_shape::air::cuda::{ProofShapePerProof, ProofShapeTracegenInputs},
};

extern "C" {
    fn _proof_shape_tracegen(
        d_trace: *mut F,
        height: usize,
        d_air_data: *const AirData,
        d_sorted_trace_data: *const *const TraceMetadata,
        d_cached_commits: *const *const Digest,
        d_per_proof: *const ProofShapePerProof,
        num_proofs: usize,
        inputs: *const ProofShapeTracegenInputs,
    ) -> i32;
    fn _public_values_recursion_tracegen(
        d_trace: *mut F,
        height: usize,
        d_pvs_data: *const *const PublicValueData,
        d_pvs_tidx: *const *const usize,
        num_proofs: usize,
        num_pvs: usize,
    ) -> i32;
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn proof_shape_tracegen(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    d_air_data: &DeviceBuffer<AirData>,
    d_sorted_trace_data: Vec<*const TraceMetadata>,
    d_cached_commits: Vec<*const Digest>,
    d_per_proof: &DeviceBuffer<ProofShapePerProof>,
    num_proofs: usize,
    inputs: &ProofShapeTracegenInputs,
) -> Result<(), CudaError> {
    CudaError::from_result(_proof_shape_tracegen(
        d_trace.as_mut_ptr(),
        height,
        d_air_data.as_ptr(),
        d_sorted_trace_data.as_ptr(),
        d_cached_commits.as_ptr(),
        d_per_proof.as_ptr(),
        num_proofs,
        inputs as *const ProofShapeTracegenInputs,
    ))
}

pub unsafe fn public_values_tracegen(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    d_pvs_data: Vec<*const PublicValueData>,
    d_pvs_tidx: Vec<*const usize>,
    num_proofs: usize,
    num_pvs: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_public_values_recursion_tracegen(
        d_trace.as_mut_ptr(),
        height,
        d_pvs_data.as_ptr(),
        d_pvs_tidx.as_ptr(),
        num_proofs,
        num_pvs,
    ))
}
