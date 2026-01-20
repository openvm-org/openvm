#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

extern "C" {
    fn _merkle_precomputation_hash_vectors(
        d_data: *const F,
        d_descriptors: *const VectorDescriptor,
        num_vectors: usize,
        d_pre_states: *mut F,
        d_post_states: *mut F,
    ) -> i32;
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VectorDescriptor {
    pub data_offset: usize,
    pub len: usize,
    pub output_offset: usize,
}

pub unsafe fn merkle_precomputation_hash_vectors(
    d_data: &DeviceBuffer<F>,
    d_descriptors: &DeviceBuffer<VectorDescriptor>,
    num_vectors: usize,
    d_pre_states: &DeviceBuffer<F>,
    d_post_states: &DeviceBuffer<F>,
) -> Result<(), CudaError> {
    CudaError::from_result(_merkle_precomputation_hash_vectors(
        d_data.as_ptr(),
        d_descriptors.as_ptr(),
        num_vectors,
        d_pre_states.as_mut_ptr(),
        d_post_states.as_mut_ptr(),
    ))
}
