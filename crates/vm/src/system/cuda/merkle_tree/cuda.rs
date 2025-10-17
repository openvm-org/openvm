#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::{base::DeviceMatrix, types::F};
use openvm_cuda_common::{
    copy::MemCopyH2D, d_buffer::DeviceBuffer, error::CudaError, stream::cudaStream_t,
};

use super::{SharedBuffer, DIGEST_WIDTH, TIMESTAMPED_BLOCK_WIDTH};

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

        fn _update_merkle_tree(
            num_leaves: usize,
            layer: *mut u32, // are actually `(u32, u32, u32, H)`s
            subtree_height: usize,
            child_buf: *mut u32,
            tmp_buf: *mut u32,
            merkle_trace: *mut u32,
            trace_height: usize,
            num_subtrees: usize,
            subtrees: *mut usize,        // is actually H**
            top_roots: *mut u32,         // are actually `H`s
            zero_hashes_end: *const u32, // are actually `H`s
            actual_subtree_heights: *const usize,
            d_poseidon2_raw_buffer: *mut std::ffi::c_void,
            d_poseidon2_buffer_idx: *mut u32,
            poseidon2_capacity: usize,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn update_merkle_tree<T>(
        trace: &DeviceMatrix<T>,
        subtree_ptrs: &DeviceBuffer<usize>,
        top_roots: &DeviceBuffer<[T; DIGEST_WIDTH]>,
        zero_hash: &DeviceBuffer<[T; DIGEST_WIDTH]>,
        touched_blocks: &DeviceBuffer<u32>,
        subtree_height: usize,
        actual_heights: &[usize],
        unpadded_height: usize,
        hasher_buffer: &SharedBuffer<F>,
    ) -> Result<(), CudaError> {
        let num_leaves = touched_blocks.len() / TIMESTAMPED_BLOCK_WIDTH;
        let num_subtrees = subtree_ptrs.len();
        let tmp_buffer = DeviceBuffer::<u32>::with_capacity(4 * num_leaves);
        let actual_heights = actual_heights.to_device().unwrap();
        CudaError::from_result(_update_merkle_tree(
            num_leaves,
            touched_blocks.as_mut_ptr(),
            subtree_height,
            tmp_buffer.as_mut_ptr(),
            tmp_buffer.as_mut_ptr().add(2 * num_leaves),
            trace.buffer().as_ptr() as *mut u32,
            unpadded_height,
            num_subtrees,
            subtree_ptrs.as_mut_ptr(),
            top_roots.as_mut_ptr() as *mut u32,
            zero_hash.as_ptr() as *mut u32,
            actual_heights.as_ptr(),
            hasher_buffer.buffer.as_mut_raw_ptr(),
            hasher_buffer.idx.as_mut_ptr(),
            hasher_buffer.buffer.len(),
        ))
    }
}
