use stark_backend_gpu::cuda::{d_buffer::DeviceBuffer, error::CudaError, stream::cudaStream_t};

pub mod merkle_tree {
    use super::*;

    extern "C" {
        fn _build_merkle_subtree(
            d_data: *mut u8,
            size: usize,
            d_tree: *mut std::ffi::c_void,
            tree_offset: usize,
            addr_space_idx: u32,
            stream: cudaStream_t,
        ) -> i32;

        fn _restore_merkle_subtree_path(
            d_in_out: *mut std::ffi::c_void,
            d_zero_hash: *mut std::ffi::c_void,
            remaining_size: usize,
            full_size: usize,
            stream: cudaStream_t,
        ) -> i32;

        fn _calculate_zero_hash(d_zero_hash: *mut std::ffi::c_void, size: usize) -> i32;

        fn _finalize_merkle_tree(
            d_roots: *mut usize,
            d_out: *mut std::ffi::c_void,
            num_roots: usize,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn build_merkle_subtree<T>(
        d_data: &DeviceBuffer<u8>,
        size: usize,
        d_tree: &DeviceBuffer<T>,
        tree_offset: usize,
        addr_space_idx: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_build_merkle_subtree(
            d_data.as_mut_ptr(),
            size,
            d_tree.as_mut_raw_ptr(),
            tree_offset,
            addr_space_idx,
            stream,
        ))
    }

    pub unsafe fn restore_merkle_subtree_path<T>(
        d_in_out: &DeviceBuffer<T>,
        d_zero_hash: &DeviceBuffer<T>,
        remaining_size: usize,
        full_size: usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_restore_merkle_subtree_path(
            d_in_out.as_mut_raw_ptr(),
            d_zero_hash.as_mut_raw_ptr(),
            remaining_size,
            full_size,
            stream,
        ))
    }

    pub unsafe fn calculate_zero_hash<T>(
        d_zero_hash: &DeviceBuffer<T>,
        size: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_calculate_zero_hash(d_zero_hash.as_mut_raw_ptr(), size))
    }

    pub unsafe fn finalize_merkle_tree<T>(
        d_roots: &DeviceBuffer<usize>,
        d_out: &DeviceBuffer<T>,
        num_roots: usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_finalize_merkle_tree(
            d_roots.as_mut_ptr(),
            d_out.as_mut_raw_ptr(),
            num_roots,
            stream,
        ))
    }
}
