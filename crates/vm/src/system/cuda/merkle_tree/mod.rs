use std::{ffi::c_void, sync::Arc};

use openvm_circuit::{
    arch::{MemoryConfig, ADDR_SPACE_OFFSET},
    system::memory::{merkle::MemoryMerkleCols, TimestampedEquipartition},
    utils::next_power_of_two_or_zero,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, prover_backend::GpuBackend};
use openvm_cuda_common::{
    copy::{cuda_memcpy, MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::{cudaStreamPerThread, default_stream_wait, CudaEvent, CudaStream},
};
use openvm_stark_backend::{
    p3_maybe_rayon::prelude::{IntoParallelIterator, ParallelIterator},
    p3_util::log2_ceil_usize,
    prover::types::AirProvingContext,
};
use p3_field::FieldAlgebra;

use super::{poseidon2::SharedBuffer, Poseidon2PeripheryChipGPU, DIGEST_WIDTH};

pub mod cuda;
use cuda::merkle_tree::*;

type H = [F; DIGEST_WIDTH];
pub const TIMESTAMPED_BLOCK_WIDTH: usize = 11;

/// A Merkle subtree stored in a single flat buffer, combining a vertical path and a heap-ordered
/// binary tree.
///
/// Memory layout:
/// - The first `path_len` elements form a vertical path (one node per level), used when the actual
///   size is smaller than the max size.
/// - The remaining elements store the subtree nodes in heap-order (breadth-first), with `size`
///   leaves and `2 * size - 1` total nodes.
///
/// The call of filling the buffer is done async on the new stream. Option<CudaEvent> is used to
/// wait for the completion.
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
    ///
    /// `addr_space_size` is the number of leaf digest nodes necessary for this address space. The
    /// `max_size` is the number of leaf digest nodes in the full balanced tree dictated by
    /// `addr_space_height` from the `MemoryConfig`.
    pub fn new(addr_space_size: usize, max_size: usize) -> Self {
        assert!(
            max_size.is_power_of_two(),
            "Max address space size must be a power of two"
        );
        let size = next_power_of_two_or_zero(addr_space_size);
        if addr_space_size == 0 {
            let mut res = MemoryMerkleSubTree::dummy();
            res.height = log2_ceil_usize(max_size);
            return res;
        }
        let height = log2_ceil_usize(size);
        let path_len = log2_ceil_usize(max_size).checked_sub(height).unwrap();
        tracing::debug!(
            "Creating a subtree buffer, size is {} (addr space size is {})",
            path_len + (2 * size - 1),
            addr_space_size
        );
        let buf = DeviceBuffer::<H>::with_capacity(path_len + (2 * size - 1));

        let created_buffer_event = CudaEvent::new().unwrap();
        unsafe {
            created_buffer_event.record(cudaStreamPerThread).unwrap();
        }

        let stream = Arc::new(CudaStream::new().unwrap());
        stream.wait(&created_buffer_event).unwrap();
        Self {
            stream,
            event: None,
            height,
            buf,
            path_len,
        }
    }

    pub fn dummy() -> Self {
        Self {
            stream: Arc::new(CudaStream::new().unwrap()),
            event: None,
            height: 0,
            buf: DeviceBuffer::new(),
            path_len: 0,
        }
    }

    /// Asynchronously builds the Merkle subtree on its dedicated CUDA stream.
    /// Also reconstructs the vertical path if `path_len > 0`, and records a completion event.
    ///
    /// Here `addr_space_idx` is the address space _shifted_ by ADDR_SPACE_OFFSET = 1
    pub fn build_async(
        &mut self,
        d_data: &DeviceBuffer<u8>,
        addr_space_idx: usize,
        zero_hash: &DeviceBuffer<H>,
    ) {
        let event = CudaEvent::new().unwrap();
        if self.buf.is_empty() {
            // TODO not really async in this branch is it
            self.buf = DeviceBuffer::with_capacity(1);
            unsafe {
                cuda_memcpy::<true, true>(
                    self.buf.as_mut_raw_ptr(),
                    zero_hash.as_ptr().add(self.height) as *mut c_void,
                    size_of::<H>(),
                )
                .unwrap();
                event.record(cudaStreamPerThread).unwrap();
            }
        } else {
            unsafe {
                build_merkle_subtree(
                    d_data,
                    1 << self.height,
                    &self.buf,
                    self.path_len,
                    addr_space_idx as u32,
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
///     - top_roots layout: \[root, hash(root_addr_space_1, root_addr_space_2),
///       hash(root_addr_space_3), hash(root_addr_space_4), ...\]
///     - if we have > 4 address spaces, top_roots will be extended with the next hash, etc.
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
    pub hasher_buffer: SharedBuffer<F>,
    mem_config: MemoryConfig,
}

impl MemoryMerkleTree {
    /// Creates a full Merkle tree with one subtree per address space.
    /// Initializes all buffers and precomputes the zero hash chain.
    pub fn new(mem_config: MemoryConfig, hasher_chip: Arc<Poseidon2PeripheryChipGPU>) -> Self {
        let addr_space_sizes = mem_config
            .addr_spaces
            .iter()
            .map(|ashc| {
                assert!(
                    ashc.num_cells % DIGEST_WIDTH == 0,
                    "the number of cells must be divisible by `DIGEST_WIDTH`"
                );
                ashc.num_cells / DIGEST_WIDTH
            })
            .collect::<Vec<_>>();
        assert!(!(addr_space_sizes.is_empty()), "Invalid config");

        let num_addr_spaces = addr_space_sizes.len() - ADDR_SPACE_OFFSET as usize;
        assert!(
            num_addr_spaces.is_power_of_two(),
            "Number of address spaces must be a one plus power of two"
        );
        for &sz in addr_space_sizes.iter().take(ADDR_SPACE_OFFSET as usize) {
            assert!(
                sz == 0,
                "The first `ADDR_SPACE_OFFSET` address spaces are assumed to be empty"
            );
        }

        let label_max_bits = mem_config.pointer_max_bits - log2_ceil_usize(DIGEST_WIDTH);

        let zero_hash = DeviceBuffer::<H>::with_capacity(label_max_bits + 1);
        let top_roots = DeviceBuffer::<H>::with_capacity(2 * num_addr_spaces - 1);
        unsafe {
            calculate_zero_hash(&zero_hash, label_max_bits).unwrap();
        }

        Self {
            stream: Arc::new(CudaStream::new().unwrap()),
            subtrees: Vec::new(),
            top_roots,
            height: label_max_bits + log2_ceil_usize(num_addr_spaces),
            zero_hash,
            hasher_buffer: hasher_chip.shared_buffer(),
            mem_config,
        }
    }

    pub fn mem_config(&self) -> &MemoryConfig {
        &self.mem_config
    }

    /// Starts asynchronous construction of the specified address space's Merkle subtree.
    /// Uses internal zero hashes and launches kernels on the subtree's own CUDA stream.
    ///
    /// Here `addr_space` is the _unshifted_ address space, so `addr_space = 0` is the immediate
    /// address space, which should be ignored.
    pub fn build_async(&mut self, d_data: &DeviceBuffer<u8>, addr_space: usize) {
        if addr_space < ADDR_SPACE_OFFSET as usize {
            return;
        }
        let addr_space_idx = addr_space - ADDR_SPACE_OFFSET as usize;
        if addr_space < self.mem_config.addr_spaces.len() && addr_space_idx == self.subtrees.len() {
            let mut subtree = MemoryMerkleSubTree::new(
                self.mem_config.addr_spaces[addr_space].num_cells / DIGEST_WIDTH,
                1 << (self.zero_hash.len() - 1), /* label_max_bits */
            );
            subtree.build_async(d_data, addr_space_idx, &self.zero_hash);
            self.subtrees.push(subtree);
        } else {
            panic!("Invalid address space index");
        }
    }

    /// Finalizes the Merkle tree by collecting all subtree roots and computing the final root.
    /// Waits for all subtrees to complete and then performs the final hash operation.
    pub fn finalize(&self) {
        for subtree in self.subtrees.iter() {
            self.stream.wait(subtree.event.as_ref().unwrap()).unwrap();
        }

        let we_can_gather_bufs_event = CudaEvent::new().unwrap();
        unsafe {
            we_can_gather_bufs_event
                .record(self.stream.as_raw())
                .unwrap();
        }
        default_stream_wait(&we_can_gather_bufs_event).unwrap();

        let roots: Vec<usize> = self
            .subtrees
            .iter()
            .map(|subtree| subtree.buf.as_ptr() as usize)
            .collect();
        let d_roots = roots.to_device().unwrap();
        let to_device_event = CudaEvent::new().unwrap();
        unsafe {
            to_device_event.record(cudaStreamPerThread).unwrap();
        }
        self.stream.wait(&to_device_event).unwrap();

        unsafe {
            finalize_merkle_tree(
                &d_roots,
                &self.top_roots,
                self.subtrees.len(),
                self.stream.as_raw(),
            )
            .unwrap();
        }

        self.stream.synchronize().unwrap();
    }

    /// Drops all massive buffers to free memory. Used at the end of an execution segment.
    pub fn drop_subtrees(&mut self) {
        self.subtrees = Vec::new();
    }

    /// Updates the tree and returns the merkle trace.
    pub fn update_with_touched_blocks(
        &self,
        unpadded_height: usize,
        d_touched_blocks: &DeviceBuffer<u32>, // consists of (as, label, ts, [F; 8])
        empty_touched_blocks: bool,
    ) -> AirProvingContext<GpuBackend> {
        let mut public_values = self.top_roots.to_host().unwrap()[0].to_vec();
        let merkle_trace = {
            let width = MemoryMerkleCols::<u8, DIGEST_WIDTH>::width();
            let padded_height = next_power_of_two_or_zero(unpadded_height);
            let output = DeviceMatrix::<F>::with_capacity(padded_height, width);
            output.buffer().fill_zero().unwrap();

            let actual_heights = self.subtrees.iter().map(|s| s.height).collect::<Vec<_>>();
            let subtrees_pointers = self
                .subtrees
                .iter()
                .map(|st| st.buf.as_ptr() as usize)
                .collect::<Vec<_>>()
                .to_device()
                .unwrap();
            unsafe {
                update_merkle_tree(
                    &output,
                    &subtrees_pointers,
                    &self.top_roots,
                    &self.zero_hash,
                    d_touched_blocks,
                    self.height - log2_ceil_usize(self.subtrees.len()),
                    &actual_heights,
                    unpadded_height,
                    &self.hasher_buffer,
                )
                .unwrap();
            }

            if empty_touched_blocks {
                // The trace is small then
                let mut output_vec = output.to_host().unwrap();
                output_vec[unpadded_height - 1 + (width - 2) * padded_height] = F::ONE; // left_direction_different
                output_vec[unpadded_height - 1 + (width - 1) * padded_height] = F::ONE; // right_direction_different
                DeviceMatrix::new(
                    Arc::new(output_vec.to_device().unwrap()),
                    padded_height,
                    width,
                )
            } else {
                output
            }
        };
        public_values.extend(self.top_roots.to_host().unwrap()[0].to_vec());

        AirProvingContext::new(Vec::new(), Some(merkle_trace), public_values)
    }

    /// An auxiliary function to calculate the required number of rows for the merkle trace.
    pub fn calculate_unpadded_height(
        &self,
        touched_memory: &TimestampedEquipartition<F, DIGEST_WIDTH>,
    ) -> usize {
        let md = self.mem_config.memory_dimensions();
        let tree_height = md.overall_height();
        let shift_address = |(sp, ptr): (u32, u32)| (sp, ptr / DIGEST_WIDTH as u32);
        2 * if touched_memory.is_empty() {
            tree_height
        } else {
            tree_height
                + (0..(touched_memory.len() - 1))
                    .into_par_iter()
                    .map(|i| {
                        let x = md.label_to_index(shift_address(touched_memory[i].0));
                        let y = md.label_to_index(shift_address(touched_memory[i + 1].0));
                        (x ^ y).ilog2() as usize
                    })
                    .sum::<usize>()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openvm_circuit::{
        arch::{
            testing::POSEIDON2_DIRECT_BUS, vm_poseidon2_config, AddressSpaceHostLayout,
            MemoryCellType, MemoryConfig,
        },
        system::{
            memory::{
                merkle::MerkleTree,
                online::{GuestMemory, LinearMemory},
                AddressMap, TimestampedValues,
            },
            poseidon2::Poseidon2PeripheryChip,
        },
    };
    use openvm_cuda_backend::prelude::F;
    use openvm_cuda_common::{
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
    };
    use openvm_instructions::{
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
        NATIVE_AS,
    };
    use openvm_stark_sdk::utils::create_seeded_rng;
    use p3_field::{FieldAlgebra, PrimeField32};
    use rand::Rng;

    use super::MemoryMerkleTree;
    use crate::system::cuda::{Poseidon2PeripheryChipGPU, DIGEST_WIDTH};

    #[test]
    fn test_cuda_merkle_tree_cpu_gpu_root_equivalence() {
        let mut rng = create_seeded_rng();
        let mem_config = {
            let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
            let max_cells = 1 << 16;
            addr_spaces[RV32_REGISTER_AS as usize].num_cells = 32 * size_of::<u32>();
            addr_spaces[RV32_MEMORY_AS as usize].num_cells = max_cells;
            addr_spaces[NATIVE_AS as usize].num_cells = max_cells;
            MemoryConfig::new(2, addr_spaces, max_cells.ilog2() as usize, 29, 17, 32)
        };

        let mut initial_memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
        for (idx, space) in mem_config.addr_spaces.iter().enumerate() {
            unsafe {
                match space.layout {
                    MemoryCellType::Null => {}
                    MemoryCellType::U8 => {
                        for i in 0..space.num_cells {
                            initial_memory.write::<u8, 1>(
                                idx as u32,
                                i as u32,
                                [rng.gen_range(0..space.layout.size()) as u8],
                            );
                        }
                    }
                    MemoryCellType::U16 => {
                        for i in 0..space.num_cells {
                            initial_memory.write::<u16, 1>(
                                idx as u32,
                                i as u32,
                                [rng.gen_range(0..space.layout.size()) as u16],
                            );
                        }
                    }
                    MemoryCellType::U32 => {
                        for i in 0..space.num_cells {
                            initial_memory.write::<u32, 1>(
                                idx as u32,
                                i as u32,
                                [rng.gen_range(0..space.layout.size()) as u32],
                            );
                        }
                    }
                    MemoryCellType::Native { .. } => {
                        for i in 0..space.num_cells {
                            initial_memory.write::<F, 1>(
                                idx as u32,
                                i as u32,
                                [F::from_canonical_u32(rng.gen_range(0..F::ORDER_U32))],
                            );
                        }
                    }
                }
            }
        }

        let gpu_hasher_chip = Arc::new(Poseidon2PeripheryChipGPU::new(
            (mem_config
                .addr_spaces
                .iter()
                .map(|ashc| ashc.num_cells * 2 + mem_config.memory_dimensions().overall_height())
                .sum::<usize>()
                * 2)
            .next_power_of_two()
                * 2
                * DIGEST_WIDTH, // max_buffer_size
            1, // sbox_regs
        ));
        let mut gpu_merkle_tree = MemoryMerkleTree::new(mem_config.clone(), gpu_hasher_chip);
        for (i, mem) in initial_memory.memory.get_memory().iter().enumerate() {
            let mem_slice = mem.as_slice();
            gpu_merkle_tree.build_async(
                &(if !mem_slice.is_empty() {
                    mem_slice.to_device().unwrap()
                } else {
                    DeviceBuffer::new()
                }),
                i,
            );
        }
        gpu_merkle_tree.finalize();

        let cpu_hasher_chip =
            Poseidon2PeripheryChip::new(vm_poseidon2_config(), POSEIDON2_DIRECT_BUS, 3);
        let mut cpu_merkle_tree = MerkleTree::<F, DIGEST_WIDTH>::from_memory(
            &initial_memory.memory,
            &mem_config.memory_dimensions(),
            &cpu_hasher_chip,
        );

        assert_eq!(
            cpu_merkle_tree.root(),
            gpu_merkle_tree.top_roots.to_host().unwrap()[0]
        );
        eprintln!("{:?}", cpu_merkle_tree.root());
        eprintln!("{:?}", gpu_merkle_tree.top_roots.to_host().unwrap()[0]);

        // Now we add some touched memory
        // We don't care about the memory layout and whatnot, because neither implementation uses
        // any special form of the touched blocks
        let touched_ptrs = mem_config
            .addr_spaces
            .iter()
            .enumerate()
            .flat_map(|(i, cnf)| {
                let mut ptrs = Vec::new();
                for j in 0..(cnf.num_cells / DIGEST_WIDTH) {
                    if rng.gen_bool(0.333) {
                        ptrs.push((i as u32, (j * DIGEST_WIDTH) as u32));
                    }
                }
                ptrs
            })
            .collect::<Vec<_>>();
        let new_data = touched_ptrs
            .iter()
            .map(|_| std::array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..F::ORDER_U32))))
            .collect::<Vec<[F; DIGEST_WIDTH]>>();
        assert!(!touched_ptrs.is_empty());
        cpu_merkle_tree.finalize(
            &cpu_hasher_chip,
            &(touched_ptrs
                .iter()
                .copied()
                .zip(new_data.iter().copied())
                .collect()),
            &mem_config.memory_dimensions(),
        );
        let touched_blocks = touched_ptrs
            .into_iter()
            .zip(new_data)
            .map(|(address, data)| {
                (
                    address,
                    TimestampedValues {
                        timestamp: rng.gen_range(0..(1u32 << mem_config.timestamp_max_bits)),
                        values: data,
                    },
                )
            })
            .collect::<Vec<_>>();
        let d_touched_blocks = touched_blocks.to_device().unwrap().as_buffer::<u32>();

        gpu_merkle_tree.update_with_touched_blocks(
            gpu_merkle_tree.calculate_unpadded_height(&touched_blocks),
            &d_touched_blocks,
            false,
        );

        assert_eq!(
            cpu_merkle_tree.root(),
            gpu_merkle_tree.top_roots.to_host().unwrap()[0]
        );
        eprintln!("{:?}", cpu_merkle_tree.root());
        eprintln!("{:?}", gpu_merkle_tree.top_roots.to_host().unwrap()[0]);
    }
}
