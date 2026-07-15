use std::{ffi::c_void, sync::Arc};

use openvm_circuit::{
    arch::{MemoryConfig, ADDR_SPACE_OFFSET, BLOCK_FE_WIDTH},
    system::memory::merkle::MemoryMerkleCols,
    utils::next_power_of_two_or_zero,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{
    copy::{cuda_memcpy_on, MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::{CudaEvent, GpuDeviceCtx},
};
use openvm_instructions::VM_DIGEST_WIDTH;
use openvm_stark_backend::{
    p3_maybe_rayon::prelude::{IntoParallelIterator, ParallelIterator},
    p3_util::log2_ceil_usize,
    prover::AirProvingContext,
};
use p3_field::PrimeCharacteristicRing;

use super::{poseidon2::SharedBuffer, Poseidon2PeripheryChipGPU};

pub mod cuda;
use cuda::merkle_tree::*;

type H = [F; VM_DIGEST_WIDTH];
/// Width of `((u32, u32), TimestampedValues<F, BLOCK_FE_WIDTH>)` in u32 units.
/// = 2 (key) + 1 (timestamp) + BLOCK_FE_WIDTH (values)
pub const TIMESTAMPED_BLOCK_WIDTH: usize = 3 + BLOCK_FE_WIDTH;
/// Width of `((u32, u32), TimestampedValues<F, VM_DIGEST_WIDTH>)` in u32 units.
/// = 2 (key) + 1 (timestamp) + VM_DIGEST_WIDTH (values)
pub const MERKLE_TOUCHED_BLOCK_WIDTH: usize = 3 + VM_DIGEST_WIDTH;
pub(crate) const OMITTED_BOTTOM_LEVELS: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum MemoryMerkleSubTreeLayout {
    Full = 0,
    OmitBottomLevels = 1,
}

/// A Merkle subtree stored in a single flat buffer, combining a vertical path and a heap-ordered
/// retained tree.
///
/// Memory layout:
/// - The first `path_len` elements form a vertical path (one node per level), used when the actual
///   size is smaller than the max size.
/// - `Full` subtrees store the remaining nodes as the complete subtree heap.
/// - `OmitBottomLevels` subtrees omit the bottom `OMITTED_BOTTOM_LEVELS` levels and store only the
///   retained heap whose leaves are the first stored hashes above the omitted levels.
///
/// All GPU work is issued on the subtree's `GpuDeviceCtx` stream.
/// `build_completion_event` records when the build kernels finish so that downstream consumers can
/// synchronize.
pub struct MemoryMerkleSubTree {
    build_completion_event: Option<CudaEvent>,
    pub buf: DeviceBuffer<H>,
    pub height: usize,
    pub path_len: usize,
    layout: MemoryMerkleSubTreeLayout,
    /// Shared handle to the initial-memory buffer (`d_data`) from [`Self::build_async`], or
    /// `None` for empty/dummy subtrees. Co-owning the buffer keeps the host from freeing it: under
    /// `OmitBottomLevels` the omitted levels aren't in `buf` and are recomputed from this buffer
    /// during [`MemoryMerkleTree::update_with_touched_blocks`] (`recompute_omitted_node` in
    /// `merkle_tree.cu`).
    ///
    /// This only covers host-side ownership; the buffer also feeds GPU kernels on the stream, so
    /// drop the subtrees (releasing these handles) only after the `stream.synchronize()` in
    /// [`MemoryMerkleTree::drop_subtrees`].
    initial_data: Option<Arc<DeviceBuffer<u8>>>,
}

impl MemoryMerkleSubTree {
    fn layout_for_height(height: usize) -> MemoryMerkleSubTreeLayout {
        if height > OMITTED_BOTTOM_LEVELS {
            MemoryMerkleSubTreeLayout::OmitBottomLevels
        } else {
            MemoryMerkleSubTreeLayout::Full
        }
    }

    fn heap_len(height: usize, layout: MemoryMerkleSubTreeLayout) -> usize {
        let retained_height = match layout {
            MemoryMerkleSubTreeLayout::Full => height,
            MemoryMerkleSubTreeLayout::OmitBottomLevels => height - OMITTED_BOTTOM_LEVELS,
        };
        2 * (1 << retained_height) - 1
    }

    fn buffer_len(
        addr_space_size: usize,
        path_len: usize,
        layout: MemoryMerkleSubTreeLayout,
    ) -> usize {
        let height = log2_ceil_usize(addr_space_size);
        path_len + Self::heap_len(height, layout)
    }

    /// Constructs a new Merkle subtree with a vertical path and heap-ordered tree.
    /// The buffer is sized based on the actual address space and the maximum size.
    ///
    /// `addr_space_size` is the number of leaf digest nodes necessary for this address space. The
    /// `max_size` is the number of leaf digest nodes in the full balanced tree dictated by
    /// `addr_space_height` from the `MemoryConfig`.
    ///
    /// `addr_space_size` must be a power of two or zero.
    /// `max_size` must be a power of two.
    pub fn new(addr_space_size: usize, max_size: usize, device_ctx: &GpuDeviceCtx) -> Self {
        assert!(
            addr_space_size == 0 || addr_space_size.is_power_of_two(),
            "The actual address space size must be a power of two"
        );
        assert!(
            max_size.is_power_of_two(),
            "Max address space size must be a power of two"
        );
        assert!(
            addr_space_size <= max_size,
            "Address space needs {addr_space_size} leaf digests but the tree supports at most \
             {max_size}; check that every address space's `num_cells` fits within \
             `pointer_max_bits`"
        );
        if addr_space_size == 0 {
            let mut res = MemoryMerkleSubTree::dummy();
            res.height = log2_ceil_usize(max_size);
            return res;
        }
        let height = log2_ceil_usize(addr_space_size);
        let path_len = log2_ceil_usize(max_size).checked_sub(height).unwrap();
        let layout = Self::layout_for_height(height);
        let buffer_len = Self::buffer_len(addr_space_size, path_len, layout);
        tracing::debug!(
            "Creating a subtree buffer, size is {} (addr space size is {})",
            buffer_len,
            addr_space_size
        );
        let buf = DeviceBuffer::<H>::with_capacity_on(buffer_len, device_ctx);

        Self {
            build_completion_event: None,
            height,
            buf,
            path_len,
            layout,
            initial_data: None,
        }
    }

    pub fn dummy() -> Self {
        Self {
            build_completion_event: None,
            height: 0,
            buf: DeviceBuffer::new(),
            path_len: 0,
            layout: MemoryMerkleSubTreeLayout::Full,
            initial_data: None,
        }
    }

    fn layout_tag(&self) -> u8 {
        self.layout as u8
    }

    fn stored_heap_height(&self) -> usize {
        match self.layout {
            MemoryMerkleSubTreeLayout::Full => self.height,
            MemoryMerkleSubTreeLayout::OmitBottomLevels => self.height - OMITTED_BOTTOM_LEVELS,
        }
    }

    /// Builds the Merkle subtree on the provided `GpuDeviceCtx` stream.
    /// Also reconstructs the vertical path if `path_len > 0`, and records a completion event.
    ///
    /// Here `addr_space_idx` is the address space _shifted_ by ADDR_SPACE_OFFSET = 1
    pub fn build_async(
        &mut self,
        d_data: Arc<DeviceBuffer<u8>>,
        addr_space_idx: usize,
        zero_hash: &DeviceBuffer<H>,
        device_ctx: &GpuDeviceCtx,
    ) {
        let event = CudaEvent::new().unwrap();
        // Co-own the buffer; it must outlive `update_with_touched_blocks`, which re-reads it under
        // the `OmitBottomLevels` layout (see the `initial_data` field).
        self.initial_data = Some(d_data.clone());
        if self.buf.is_empty() {
            self.buf = DeviceBuffer::with_capacity_on(1, device_ctx);
            unsafe {
                cuda_memcpy_on::<true, true>(
                    self.buf.as_mut_raw_ptr(),
                    zero_hash.as_ptr().add(self.height) as *mut c_void,
                    size_of::<H>(),
                    device_ctx,
                )
                .unwrap();
                event.record(device_ctx.stream.as_raw()).unwrap();
            }
        } else {
            unsafe {
                build_merkle_subtree(
                    &d_data,
                    1 << self.stored_heap_height(),
                    &self.buf,
                    self.path_len,
                    addr_space_idx as u32,
                    self.layout_tag(),
                    device_ctx.stream.as_raw(),
                )
                .unwrap();

                if self.path_len > 0 {
                    restore_merkle_subtree_path(
                        &self.buf,
                        zero_hash,
                        self.path_len,
                        self.height + self.path_len,
                        device_ctx.stream.as_raw(),
                    )
                    .unwrap();
                }
                event.record(device_ctx.stream.as_raw()).unwrap();
            }
        }
        self.build_completion_event = Some(event);
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
/// - Subtrees are built on the tree's `GpuDeviceCtx` stream.
/// - The final root is computed after all subtrees complete on that same stream.
pub struct MemoryMerkleTree {
    pub device_ctx: GpuDeviceCtx,
    pub subtrees: Vec<MemoryMerkleSubTree>,
    pub top_roots: DeviceBuffer<H>,
    zero_hash: DeviceBuffer<H>,
    pub height: usize,
    pub hasher_buffer: SharedBuffer<F>,
    mem_config: MemoryConfig,
    pub(crate) top_roots_host: Vec<H>,
}

impl MemoryMerkleTree {
    /// Creates a full Merkle tree with one subtree per address space.
    /// Initializes all buffers and precomputes the zero hash chain.
    pub fn new(
        mem_config: MemoryConfig,
        hasher_chip: Arc<Poseidon2PeripheryChipGPU>,
        device_ctx: GpuDeviceCtx,
    ) -> Self {
        let addr_space_sizes = mem_config
            .addr_spaces
            .iter()
            .map(|ashc| {
                assert!(
                    ashc.num_cells % VM_DIGEST_WIDTH == 0,
                    "the number of cells must be divisible by `VM_DIGEST_WIDTH`"
                );
                ashc.num_cells / VM_DIGEST_WIDTH
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

        let label_max_bits = mem_config.memory_dimensions().address_height;

        let zero_hash = DeviceBuffer::<H>::with_capacity_on(label_max_bits + 1, &device_ctx);
        let top_roots = DeviceBuffer::<H>::with_capacity_on(2 * num_addr_spaces - 1, &device_ctx);
        unsafe {
            calculate_zero_hash(&zero_hash, label_max_bits, device_ctx.stream.as_raw()).unwrap();
        }

        Self {
            device_ctx,
            subtrees: Vec::new(),
            top_roots,
            height: label_max_bits + log2_ceil_usize(num_addr_spaces),
            zero_hash,
            hasher_buffer: hasher_chip.shared_buffer(),
            mem_config,
            top_roots_host: vec![],
        }
    }

    pub fn mem_config(&self) -> &MemoryConfig {
        &self.mem_config
    }

    /// Starts construction of the specified address space's Merkle subtree.
    /// Uses internal zero hashes and launches kernels on the tree's `GpuDeviceCtx` stream.
    ///
    /// Here `addr_space` is the _unshifted_ address space, so `addr_space = 0` is the immediate
    /// address space, which should be ignored.
    ///
    /// **Note:** the caller MUST ENSURE that `d_data` lives long enough to be there
    /// when the enqueued task actually starts. Moreover, when the subtree uses the
    /// `OmitBottomLevels` layout, `d_data` is also re-read during
    /// [`Self::update_with_touched_blocks`] to recompute the omitted bottom levels, so it must
    /// remain valid until that update completes — not just until the build kernel runs. See
    /// [`MemoryMerkleSubTree`]'s `initial_data` field for details.
    pub fn build_async(&mut self, d_data: Arc<DeviceBuffer<u8>>, addr_space: usize) {
        if addr_space < ADDR_SPACE_OFFSET as usize {
            return;
        }
        let addr_space_idx = addr_space - ADDR_SPACE_OFFSET as usize;
        if addr_space < self.mem_config.addr_spaces.len() && addr_space_idx == self.subtrees.len() {
            let mut subtree = MemoryMerkleSubTree::new(
                self.mem_config.addr_spaces[addr_space].num_cells / VM_DIGEST_WIDTH,
                1 << (self.zero_hash.len() - 1), /* label_max_bits */
                &self.device_ctx,
            );
            subtree.build_async(d_data, addr_space_idx, &self.zero_hash, &self.device_ctx);
            self.subtrees.push(subtree);
        } else {
            panic!("Invalid address space ID");
        }
    }

    /// Finalizes the Merkle tree by collecting all subtree roots and computing the final root.
    /// All subtree builds were issued on the same `GpuDeviceCtx` stream, so stream ordering
    /// guarantees they are complete before the finalize kernel runs.
    pub fn finalize(&mut self) {
        let roots: Vec<usize> = self
            .subtrees
            .iter()
            .map(|subtree| subtree.buf.as_ptr() as usize)
            .collect();
        let d_roots = roots.to_device_on(&self.device_ctx).unwrap();

        unsafe {
            finalize_merkle_tree(
                &d_roots,
                &self.top_roots,
                self.subtrees.len(),
                self.device_ctx.stream.as_raw(),
            )
            .unwrap();
        }
    }

    /// Drops all massive buffers to free memory. Used at the end of an execution segment.
    ///
    /// Synchronizes the tree's `GpuDeviceCtx` stream before deallocating buffers and destroying
    /// events.
    pub fn drop_subtrees(&mut self) {
        self.device_ctx.stream.synchronize().unwrap();
        self.subtrees.clear();
    }

    /// Updates the tree and returns the merkle trace.
    ///
    /// `d_touched_blocks` consists of `(as, ptr, ts, [F; VM_DIGEST_WIDTH])`.
    pub fn update_with_touched_blocks(
        &mut self,
        unpadded_height: usize,
        d_touched_blocks: &DeviceBuffer<u32>,
        empty_touched_blocks: bool,
    ) -> AirProvingContext<GpuBackend> {
        let mut public_values = self.top_roots.to_host_on(&self.device_ctx).unwrap()[0].to_vec();
        // .to_host() calls cudaEventSynchronize on the D2H memcpy, which also means all subtree
        // events are now completed, so we can clean up the events.
        for subtree in &mut self.subtrees {
            subtree.build_completion_event = None;
        }
        let merkle_trace = {
            let width = MemoryMerkleCols::<u8, VM_DIGEST_WIDTH>::width();
            let padded_height = next_power_of_two_or_zero(unpadded_height);
            let output =
                DeviceMatrix::<F>::with_capacity_on(padded_height, width, &self.device_ctx);
            output.buffer().fill_zero_on(&self.device_ctx).unwrap();

            let actual_heights = self.subtrees.iter().map(|s| s.height).collect::<Vec<_>>();
            let subtree_layouts = self
                .subtrees
                .iter()
                .map(|s| s.layout_tag())
                .collect::<Vec<_>>();
            let initial_data_ptrs = self
                .subtrees
                .iter()
                .map(|s| s.initial_data.as_ref().map_or(0, |b| b.as_ptr() as usize))
                .collect::<Vec<_>>();
            let subtrees_pointers = self
                .subtrees
                .iter()
                .map(|st| st.buf.as_ptr() as usize)
                .collect::<Vec<_>>()
                .to_device_on(&self.device_ctx)
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
                    &subtree_layouts,
                    &initial_data_ptrs,
                    unpadded_height,
                    &self.hasher_buffer,
                    &self.device_ctx,
                )
                .unwrap();
            }

            if empty_touched_blocks {
                // The trace is small then
                let mut output_vec = output.buffer().to_host_on(&self.device_ctx).unwrap();
                output_vec[unpadded_height - 1 + (width - 2) * padded_height] = F::ONE; // left_direction_different
                output_vec[unpadded_height - 1 + (width - 1) * padded_height] = F::ONE; // right_direction_different
                DeviceMatrix::new(
                    Arc::new(output_vec.to_device_on(&self.device_ctx).unwrap()),
                    padded_height,
                    width,
                )
            } else {
                output
            }
        };
        self.top_roots_host = self.top_roots.to_host_on(&self.device_ctx).unwrap();
        public_values.extend(self.top_roots_host[0]);

        AirProvingContext::new(Vec::new(), merkle_trace, public_values)
    }

    /// An auxiliary function to calculate the required number of rows for the merkle trace.
    /// Generic over BLOCK_SIZE since only addresses are used, not values.
    pub fn calculate_unpadded_height<A: Sync>(
        &self,
        touched_memory: &[A],
        address: impl Fn(&A) -> (u32, u32) + Sync,
    ) -> usize {
        let md = self.mem_config.memory_dimensions();
        let tree_height = md.overall_height();
        let shift_address = |(sp, ptr): (u32, u32)| (sp, ptr / VM_DIGEST_WIDTH as u32);
        2 * if touched_memory.is_empty() {
            tree_height
        } else {
            tree_height
                + (0..(touched_memory.len() - 1))
                    .into_par_iter()
                    .map(|i| {
                        let x = md.label_to_index(shift_address(address(&touched_memory[i])));
                        let y = md.label_to_index(shift_address(address(&touched_memory[i + 1])));
                        let xor = x ^ y;
                        if xor == 0 {
                            0
                        } else {
                            xor.ilog2() as usize
                        }
                    })
                    .sum::<usize>()
        }
    }

    /// Device replay reaches the memory inventory as already-merged Merkle
    /// leaf pointers. Computing the height from those pointers is equivalent to
    /// the block-label form above: consecutive blocks in the same leaf add a
    /// zero transition and therefore do not affect the sum.
    pub fn calculate_unpadded_height_from_leaf_ptrs(&self, leaf_ptrs: &[(u32, u32)]) -> usize {
        let md = self.mem_config.memory_dimensions();
        let tree_height = md.overall_height();
        let to_label = |(sp, ptr): (u32, u32)| (sp, ptr / DIGEST_WIDTH as u32);
        2 * if leaf_ptrs.is_empty() {
            tree_height
        } else {
            tree_height
                + (0..(leaf_ptrs.len() - 1))
                    .into_par_iter()
                    .map(|i| {
                        let x = md.label_to_index(to_label(leaf_ptrs[i]));
                        let y = md.label_to_index(to_label(leaf_ptrs[i + 1]));
                        let xor = x ^ y;
                        if xor == 0 {
                            0
                        } else {
                            xor.ilog2() as usize
                        }
                    })
                    .sum::<usize>()
        }
    }
}

impl Drop for MemoryMerkleTree {
    fn drop(&mut self) {
        self.drop_subtrees();
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use openvm_circuit::{
        arch::{vm_poseidon2_config, MemoryCellType, MemoryConfig, U16_CELL_SIZE},
        system::{
            cuda::merkle_tree::MERKLE_TOUCHED_BLOCK_WIDTH,
            memory::{
                merkle::{MemoryMerkleChip, MerkleTree},
                online::{GuestMemory, LinearMemory},
                AddressMap, TimestampedValues,
            },
            poseidon2::Poseidon2PeripheryChip,
        },
    };
    use openvm_cuda_backend::prelude::{F, SC};
    use openvm_cuda_common::{
        common::get_device,
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use openvm_instructions::{
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
        DEFERRAL_AS, VM_DIGEST_WIDTH,
    };
    use openvm_stark_backend::{interaction::PermutationCheckBus, prover::MatrixDimensions};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use p3_field::{PrimeCharacteristicRing, PrimeField32};
    use rand::Rng;

    use super::{
        MemoryMerkleSubTree, MemoryMerkleSubTreeLayout, MemoryMerkleTree, OMITTED_BOTTOM_LEVELS,
    };
    use crate::{
        arch::testing::{MEMORY_MERKLE_BUS, POSEIDON2_DIRECT_BUS},
        system::cuda::Poseidon2PeripheryChipGPU,
    };

    #[test]
    fn test_cuda_merkle_subtree_layout_and_buffer_sizes() {
        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let max_size = 1 << (OMITTED_BOTTOM_LEVELS + 3);

        let below =
            MemoryMerkleSubTree::new(1 << (OMITTED_BOTTOM_LEVELS - 1), max_size, &device_ctx);
        assert_eq!(below.layout, MemoryMerkleSubTreeLayout::Full);
        assert_eq!(
            below.buf.len(),
            below.path_len + (2 * (1 << (OMITTED_BOTTOM_LEVELS - 1)) - 1)
        );

        let equal = MemoryMerkleSubTree::new(1 << OMITTED_BOTTOM_LEVELS, max_size, &device_ctx);
        assert_eq!(equal.layout, MemoryMerkleSubTreeLayout::Full);
        assert_eq!(
            equal.buf.len(),
            equal.path_len + (2 * (1 << OMITTED_BOTTOM_LEVELS) - 1)
        );

        let above =
            MemoryMerkleSubTree::new(1 << (OMITTED_BOTTOM_LEVELS + 1), max_size, &device_ctx);
        let full_len = above.path_len + (2 * (1 << (OMITTED_BOTTOM_LEVELS + 1)) - 1);
        let optimized_len = above.path_len + (2 * (1 << 1) - 1);
        assert_eq!(above.layout, MemoryMerkleSubTreeLayout::OmitBottomLevels);
        assert_eq!(above.buf.len(), optimized_len);
        assert!(above.buf.len() < full_len);
    }

    #[test]
    fn test_cuda_merkle_tree_cpu_gpu_root_equivalence() {
        let mut rng = create_seeded_rng();
        let mem_config = {
            let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
            let max_ptr_bits = 16;
            let max_cells = 1 << max_ptr_bits;
            // RV64_REGISTER_AS uses u16 storage cells.
            addr_spaces[RV64_REGISTER_AS as usize].num_cells =
                32 * size_of::<u64>() / U16_CELL_SIZE;
            addr_spaces[RV64_MEMORY_AS as usize].num_cells = max_cells;
            addr_spaces[DEFERRAL_AS as usize].num_cells = max_cells;
            MemoryConfig::new(2, addr_spaces, max_ptr_bits, 29, 17)
        };

        let mut initial_memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
        for (idx, space) in mem_config.addr_spaces.iter().enumerate() {
            unsafe {
                match space.layout {
                    MemoryCellType::Null => {}
                    MemoryCellType::U8 => {
                        for i in 0..space.num_cells {
                            initial_memory.write_bytes::<1>(idx as u32, i as u32, [rng.random()]);
                        }
                    }
                    MemoryCellType::U16 => {
                        for i in 0..space.num_cells {
                            initial_memory.write::<u16, 1>(idx as u32, i as u32, [rng.random()]);
                        }
                    }
                    MemoryCellType::U32 => {
                        for i in 0..space.num_cells {
                            initial_memory.write::<u32, 1>(idx as u32, i as u32, [rng.random()]);
                        }
                    }
                    MemoryCellType::F { .. } => {
                        for i in 0..space.num_cells {
                            initial_memory.write::<F, 1>(
                                idx as u32,
                                i as u32,
                                [F::from_u32(rng.random_range(0..F::ORDER_U32))],
                            );
                        }
                    }
                }
            }
        }

        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let gpu_hasher_chip = Arc::new(Poseidon2PeripheryChipGPU::new(
            1, // sbox_regs
            device_ctx.clone(),
        ));
        let mut gpu_merkle_tree = MemoryMerkleTree::new(
            mem_config.clone(),
            gpu_hasher_chip.clone(),
            device_ctx.clone(),
        );
        let mem_slices = initial_memory
            .memory
            .get_memory()
            .iter()
            .map(|mem| {
                let mem_slice = mem.as_slice();
                Arc::new(if !mem_slice.is_empty() {
                    mem_slice.to_device_on(&gpu_merkle_tree.device_ctx).unwrap()
                } else {
                    DeviceBuffer::new()
                })
            })
            .collect::<Vec<_>>();
        for (i, mem_slice) in mem_slices.iter().enumerate() {
            gpu_merkle_tree.build_async(mem_slice.clone(), i);
        }
        assert_eq!(
            gpu_merkle_tree.subtrees[RV64_REGISTER_AS as usize - 1].layout,
            MemoryMerkleSubTreeLayout::OmitBottomLevels
        );
        assert_eq!(
            gpu_merkle_tree.subtrees[RV64_MEMORY_AS as usize - 1].layout,
            MemoryMerkleSubTreeLayout::OmitBottomLevels
        );
        assert_eq!(
            gpu_merkle_tree.subtrees[DEFERRAL_AS as usize - 1].layout,
            MemoryMerkleSubTreeLayout::OmitBottomLevels
        );
        gpu_merkle_tree.finalize();

        let cpu_hasher_chip = Poseidon2PeripheryChip::new(vm_poseidon2_config(), 3);
        let mut cpu_merkle_tree = MerkleTree::<F, VM_DIGEST_WIDTH>::from_memory(
            &initial_memory.memory,
            &mem_config.memory_dimensions(),
            &cpu_hasher_chip,
        );

        assert_eq!(
            cpu_merkle_tree.root(),
            gpu_merkle_tree
                .top_roots
                .to_host_on(&gpu_merkle_tree.device_ctx)
                .unwrap()[0]
        );
        eprintln!("{:?}", cpu_merkle_tree.root());
        eprintln!(
            "{:?}",
            gpu_merkle_tree
                .top_roots
                .to_host_on(&gpu_merkle_tree.device_ctx)
                .unwrap()[0]
        );

        // Now we add some touched memory
        // We don't care about the memory layout and whatnot, because neither implementation uses
        // any special form of the touched blocks
        let touched_ptrs = mem_config
            .addr_spaces
            .iter()
            .enumerate()
            .flat_map(|(i, cnf)| {
                let mut ptrs = Vec::new();
                for j in 0..(cnf.num_cells / VM_DIGEST_WIDTH) {
                    if rng.random_bool(0.333) {
                        ptrs.push((i as u32, (j * VM_DIGEST_WIDTH) as u32));
                    }
                }
                ptrs
            })
            .collect::<Vec<_>>();
        let new_data = touched_ptrs
            .iter()
            .map(|_| std::array::from_fn(|_| F::from_u32(rng.random_range(0..F::ORDER_U32))))
            .collect::<Vec<[F; VM_DIGEST_WIDTH]>>();
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
                        timestamp: rng.random_range(0..(1u32 << mem_config.timestamp_max_bits)),
                        values: data,
                    },
                )
            })
            .collect::<Vec<_>>();
        let mut merkle_records =
            Vec::<u32>::with_capacity(touched_blocks.len() * MERKLE_TOUCHED_BLOCK_WIDTH);
        for (address, ts_values) in &touched_blocks {
            let (address_space, ptr) = *address;
            merkle_records.push(address_space);
            merkle_records.push(ptr);
            merkle_records.push(ts_values.timestamp);
            for &v in &ts_values.values {
                merkle_records.push(unsafe { std::mem::transmute::<F, u32>(v) });
            }
        }
        let d_touched_blocks = merkle_records
            .to_device_on(&gpu_merkle_tree.device_ctx)
            .unwrap();

        let unpadded_height =
            gpu_merkle_tree.calculate_unpadded_height(&touched_blocks, |(addr, _)| *addr);
        gpu_hasher_chip.prepare_records(unpadded_height);
        gpu_merkle_tree.update_with_touched_blocks(unpadded_height, &d_touched_blocks, false);

        assert_eq!(
            cpu_merkle_tree.root(),
            gpu_merkle_tree
                .top_roots
                .to_host_on(&gpu_merkle_tree.device_ctx)
                .unwrap()[0]
        );
        eprintln!("{:?}", cpu_merkle_tree.root());
        eprintln!(
            "{:?}",
            gpu_merkle_tree
                .top_roots
                .to_host_on(&gpu_merkle_tree.device_ctx)
                .unwrap()[0]
        );
    }

    /// Checks that the *trace* (not just the root) produced by the GPU
    /// `update_with_touched_blocks` contains exactly the same rows as the canonical trace
    /// produced by the CPU `MemoryMerkleChip`. The CPU trace is known to satisfy the
    /// `MemoryMerkleAir` constraints (covered by the CPU-side merkle tests), so matching every
    /// row content-for-content implies the GPU emits the correct merkle trace. This exercises the
    /// `OmitBottomLevels` trace-generation path, whose row contents (the reconstructed omitted
    /// levels) are not checked by the root-equivalence test.
    ///
    /// The comparison is order-independent: the `MemoryMerkleAir` permits more than one valid row
    /// ordering and the GPU lays rows out differently than the CPU, so we compare the two traces
    /// as multisets of rows rather than positionally.
    #[test]
    fn test_cuda_merkle_tree_cpu_gpu_trace_equivalence() {
        let mut rng = create_seeded_rng();
        let mem_config = {
            let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
            let max_ptr_bits = 16;
            let max_cells = 1 << max_ptr_bits;
            // RV64_REGISTER_AS uses u16 storage cells.
            addr_spaces[RV64_REGISTER_AS as usize].num_cells =
                32 * size_of::<u64>() / U16_CELL_SIZE;
            addr_spaces[RV64_MEMORY_AS as usize].num_cells = max_cells;
            addr_spaces[DEFERRAL_AS as usize].num_cells = max_cells;
            MemoryConfig::new(2, addr_spaces, max_ptr_bits, 29, 17)
        };

        let mut initial_memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
        for (idx, space) in mem_config.addr_spaces.iter().enumerate() {
            unsafe {
                match space.layout {
                    MemoryCellType::Null => {}
                    MemoryCellType::U8 => {
                        for i in 0..space.num_cells {
                            initial_memory.write::<u8, 1>(idx as u32, i as u32, [rng.random()]);
                        }
                    }
                    MemoryCellType::U16 => {
                        for i in 0..space.num_cells {
                            initial_memory.write::<u16, 1>(idx as u32, i as u32, [rng.random()]);
                        }
                    }
                    MemoryCellType::U32 => {
                        for i in 0..space.num_cells {
                            initial_memory.write::<u32, 1>(idx as u32, i as u32, [rng.random()]);
                        }
                    }
                    MemoryCellType::F { .. } => {
                        for i in 0..space.num_cells {
                            initial_memory.write::<F, 1>(
                                idx as u32,
                                i as u32,
                                [F::from_u32(rng.random_range(0..F::ORDER_U32))],
                            );
                        }
                    }
                }
            }
        }

        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let gpu_hasher_chip = Arc::new(Poseidon2PeripheryChipGPU::new(1, device_ctx.clone()));
        let mut gpu_merkle_tree = MemoryMerkleTree::new(
            mem_config.clone(),
            gpu_hasher_chip.clone(),
            device_ctx.clone(),
        );
        let mem_slices = initial_memory
            .memory
            .get_memory()
            .iter()
            .map(|mem| {
                let mem_slice = mem.as_slice();
                Arc::new(if !mem_slice.is_empty() {
                    mem_slice.to_device_on(&gpu_merkle_tree.device_ctx).unwrap()
                } else {
                    DeviceBuffer::new()
                })
            })
            .collect::<Vec<_>>();
        for (i, mem_slice) in mem_slices.iter().enumerate() {
            gpu_merkle_tree.build_async(mem_slice.clone(), i);
        }
        gpu_merkle_tree.finalize();

        // Touched blocks: ~1/3 of digest-aligned pointers get fresh random values.
        let touched_ptrs = mem_config
            .addr_spaces
            .iter()
            .enumerate()
            .flat_map(|(i, cnf)| {
                let mut ptrs = Vec::new();
                for j in 0..(cnf.num_cells / VM_DIGEST_WIDTH) {
                    if rng.random_bool(0.333) {
                        ptrs.push((i as u32, (j * VM_DIGEST_WIDTH) as u32));
                    }
                }
                ptrs
            })
            .collect::<Vec<_>>();
        let new_data = touched_ptrs
            .iter()
            .map(|_| std::array::from_fn(|_| F::from_u32(rng.random_range(0..F::ORDER_U32))))
            .collect::<Vec<[F; VM_DIGEST_WIDTH]>>();
        assert!(!touched_ptrs.is_empty());

        // Build the canonical CPU trace from the same initial memory and touched blocks, using a
        // Poseidon2 hasher equivalent to the GPU one.
        let cpu_hasher_chip = Poseidon2PeripheryChip::new(vm_poseidon2_config(), 3);
        let mut cpu_merkle_chip = MemoryMerkleChip::<VM_DIGEST_WIDTH, F>::new(
            mem_config.memory_dimensions(),
            PermutationCheckBus::new(MEMORY_MERKLE_BUS),
            PermutationCheckBus::new(POSEIDON2_DIRECT_BUS),
        );
        let final_partition: BTreeMap<(u32, u32), [F; VM_DIGEST_WIDTH]> = touched_ptrs
            .iter()
            .copied()
            .zip(new_data.iter().copied())
            .collect();
        cpu_merkle_chip.finalize(&initial_memory.memory, &final_partition, &cpu_hasher_chip);
        let cpu_ctx = cpu_merkle_chip.generate_proving_ctx::<SC>();

        // Run the GPU update and capture the resulting trace.
        let touched_blocks = touched_ptrs
            .into_iter()
            .zip(new_data)
            .map(|(address, data)| {
                (
                    address,
                    TimestampedValues {
                        timestamp: rng.random_range(0..(1u32 << mem_config.timestamp_max_bits)),
                        values: data,
                    },
                )
            })
            .collect::<Vec<_>>();
        let mut merkle_records =
            Vec::<u32>::with_capacity(touched_blocks.len() * MERKLE_TOUCHED_BLOCK_WIDTH);
        for (address, ts_values) in &touched_blocks {
            let (address_space, ptr) = *address;
            merkle_records.push(address_space);
            merkle_records.push(ptr);
            merkle_records.push(ts_values.timestamp);
            for &v in &ts_values.values {
                merkle_records.push(unsafe { std::mem::transmute::<F, u32>(v) });
            }
        }
        let d_touched_blocks = merkle_records
            .to_device_on(&gpu_merkle_tree.device_ctx)
            .unwrap();

        let unpadded_height =
            gpu_merkle_tree.calculate_unpadded_height(&touched_blocks, |(addr, _)| *addr);
        gpu_hasher_chip.prepare_records(unpadded_height);
        let merkle_ctx =
            gpu_merkle_tree.update_with_touched_blocks(unpadded_height, &d_touched_blocks, false);

        // The GPU trace must contain exactly the same rows as the constraint-valid CPU trace.
        let width = cpu_ctx.common_main.width;
        let height = cpu_ctx.common_main.values.len() / width;
        let gpu_trace = &merkle_ctx.common_main;
        assert_eq!(gpu_trace.width(), width, "trace width mismatch");
        assert_eq!(gpu_trace.height(), height, "trace (padded) height mismatch");

        // CPU trace is row-major; GPU trace is column-major on device.
        let cpu_vals = &cpu_ctx.common_main.values;
        let gpu_vals = gpu_trace
            .buffer()
            .to_host_on(&gpu_merkle_tree.device_ctx)
            .unwrap();
        let row_to_u32 = |get: &dyn Fn(usize, usize) -> F, r: usize| -> Vec<u32> {
            (0..width).map(|c| get(r, c).as_canonical_u32()).collect()
        };
        let cpu_get = |r: usize, c: usize| cpu_vals[r * width + c];
        let gpu_get = |r: usize, c: usize| gpu_vals[c * height + r];
        let mut cpu_rows: Vec<Vec<u32>> = (0..height).map(|r| row_to_u32(&cpu_get, r)).collect();
        let mut gpu_rows: Vec<Vec<u32>> = (0..height).map(|r| row_to_u32(&gpu_get, r)).collect();
        cpu_rows.sort_unstable();
        gpu_rows.sort_unstable();
        assert_eq!(
            gpu_rows, cpu_rows,
            "GPU merkle trace rows do not match the CPU reference trace"
        );
    }
}
