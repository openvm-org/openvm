use std::sync::Arc;

use openvm_circuit::utils::next_power_of_two_or_zero;
use openvm_stark_backend::p3_util::log2_ceil_usize;
use stark_backend_gpu::{
    cuda::{
        copy::MemCopyH2D,
        d_buffer::DeviceBuffer,
        stream::{CudaEvent, CudaStream},
    },
    prelude::F,
};

mod cuda;
use cuda::merkle_tree::*;

const DIGEST_WIDTH: usize = 8;
type H = [F; DIGEST_WIDTH];

/// A Merkle subtree stored in a single flat buffer, combining a vertical path and a heap-ordered binary tree.
///
/// Memory layout:
/// - The first `path_len` elements form a vertical path (one node per level), used when the actual size is smaller than the max size.
/// - The remaining elements store the subtree nodes in heap-order (breadth-first), with `size` leaves and `2 * size - 1` total nodes.
///
/// The call of filling the buffer is done async on the new stream. Option<CudaEvent> is used to wait for the completion.
pub struct MemoryMerkleSubTree {
    pub stream: Arc<CudaStream>,
    pub event: Option<CudaEvent>,
    pub buf: DeviceBuffer<H>,
    pub height: usize,
    pub path_len: usize,
}

impl MemoryMerkleSubTree {
    /// Constructs a new Merkle subtree with a vertical path and heap-ordered tree.
    /// The buffer is sized based on the actual address space and the maximum size.
    pub fn new(addr_space_size: usize, max_size: usize) -> Self {
        assert!(
            max_size.is_power_of_two(),
            "Max address space size must be a power of two"
        );
        let size = next_power_of_two_or_zero(addr_space_size);
        let height = log2_ceil_usize(size);
        let path_len = log2_ceil_usize(max_size) - height;
        let buf = DeviceBuffer::<H>::with_capacity(path_len + (2 * size - 1));
        let stream = Arc::new(CudaStream::new().unwrap());
        Self {
            stream,
            event: None,
            height,
            buf,
            path_len,
        }
    }

    /// Asynchronously builds the Merkle subtree on its dedicated CUDA stream.
    /// Also reconstructs the vertical path if `path_len > 0`, and records a completion event.
    pub fn build_async(&mut self, data: &[u8], addr_space_idx: u32, zero_hash: &DeviceBuffer<H>) {
        let d_data = data.to_device().unwrap();
        let event = CudaEvent::new().unwrap();
        unsafe {
            build_merkle_subtree(
                &d_data,
                1 << self.height,
                &self.buf,
                self.path_len,
                addr_space_idx,
                self.stream.as_raw(),
            )
            .unwrap();

            if self.path_len > 0 {
                restore_merkle_subtree_path(
                    &self.buf,
                    zero_hash,
                    self.path_len,
                    self.height + self.path_len,
                    self.stream.as_raw(),
                )
                .unwrap();
            }
            event.record(self.stream.as_raw()).unwrap();
        }
        self.event = Some(event);
    }

    /// Returns the bounds [start, end) of the layer at the given depth.
    /// These bounds correspond to the indices of the layer in the buffer.
    /// depth: 0 = root, 1 = root's children, ..., height-1 = leaves
    pub fn layer_bounds(&self, depth: usize) -> (usize, usize) {
        let global_height = self.height + self.path_len;
        assert!(
            depth < global_height,
            "Depth {} out of bounds for height {}",
            depth,
            global_height
        );
        if depth >= self.path_len {
            // depth is within the heap-ordered subtree
            let d = depth - self.path_len;
            let start = self.path_len + ((1 << d) - 1);
            let end = self.path_len + ((1 << (d + 1)) - 1);
            (start, end)
        } else {
            // vertical path layer: single node per level
            (depth, depth + 1)
        }
    }
}

/// A Memory Merkle tree composed of independent subtrees (one per address space),
/// each built asynchronously and finalized into a top-level Merkle root.
///
/// Layout:
/// - The memory is split across multiple `MemoryMerkleSubTree` instances, one per address space.
/// - The top-level tree is formed by hashing all subtree roots into a single buffer (`top_roots`).
/// - - top_roots layout: [root, hash(root_addr_space_1, root_addr_space_2), hash(root_addr_space_3), hash(root_addr_space_4)]
/// - - if we have > 4 address spaces, top_roots will be extended with the next hash, etc.
///
/// Execution:
/// - Subtrees are built asynchronously on individual CUDA streams.
/// - The final root is computed after all subtrees complete, on a shared stream.
/// - `CudaEvent`s are used to synchronize subtree completion.
pub struct MemoryMerkleTree {
    pub stream: Arc<CudaStream>,
    pub subtrees: Vec<MemoryMerkleSubTree>,
    pub top_roots: DeviceBuffer<H>,
    zero_hash: DeviceBuffer<H>,
    pub height: usize,
}

impl MemoryMerkleTree {
    /// Creates a full Merkle tree with one subtree per address space.
    /// Initializes all buffers and precomputes the zero hash chain.
    pub fn new(addr_space_sizes: Vec<usize>) -> Self {
        assert!(!(addr_space_sizes.is_empty()), "No address spaces given?");

        let num_addr_spaces = addr_space_sizes.len();
        assert!(
            num_addr_spaces.is_power_of_two(),
            "Number of address spaces must be a power of two"
        );

        let max_size = next_power_of_two_or_zero(*addr_space_sizes.iter().max().unwrap());

        let zero_hash = DeviceBuffer::<H>::with_capacity(log2_ceil_usize(max_size));
        let top_roots = DeviceBuffer::<H>::with_capacity(num_addr_spaces - 1);
        let subtrees: Vec<MemoryMerkleSubTree> = addr_space_sizes
            .iter()
            .map(|size| MemoryMerkleSubTree::new(*size, max_size))
            .collect();
        unsafe {
            calculate_zero_hash(&zero_hash, log2_ceil_usize(max_size)).unwrap();
        }

        Self {
            stream: subtrees.first().unwrap().stream.clone(),
            subtrees,
            top_roots,
            height: log2_ceil_usize(max_size) + log2_ceil_usize(num_addr_spaces),
            zero_hash,
        }
    }

    /// Starts asynchronous construction of the specified address space's Merkle subtree.
    /// Uses internal zero hashes and launches kernels on the subtree's own CUDA stream.
    pub fn build_async(&mut self, data: &[u8], addr_space_idx: u32) {
        if let Some(subtree) = self.subtrees.get_mut(addr_space_idx as usize - 1) {
            subtree.build_async(data, addr_space_idx, &self.zero_hash);
        } else {
            panic!("Invalid address space index");
        }
    }

    /// Finalizes the Merkle tree by collecting all subtree roots and computing the final root.
    /// Waits for all subtrees to complete and then performs the final hash operation.
    pub fn finalize_async(&self) {
        let roots: Vec<usize> = self
            .subtrees
            .iter()
            .map(|subtree| subtree.buf.as_ptr() as usize)
            .collect();
        let d_roots = roots.to_device().unwrap();
        for subtree in self.subtrees.iter() {
            self.stream.wait(subtree.event.as_ref().unwrap()).unwrap();
        }
        unsafe {
            finalize_merkle_tree(
                &d_roots,
                &self.top_roots,
                self.subtrees.len(),
                self.stream.as_raw(),
            )
            .unwrap();
        }
    }
}
