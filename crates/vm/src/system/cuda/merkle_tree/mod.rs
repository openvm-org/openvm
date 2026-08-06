use std::{ffi::c_void, sync::Arc};

use openvm_circuit::{
    arch::{AddressSpaceHostLayout, MemoryConfig, ADDR_SPACE_OFFSET, BLOCK_FE_WIDTH},
    system::memory::{
        merkle::MemoryMerkleCols,
        online::{LinearMemory, PAGE_SIZE},
        AddressMap,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{
    copy::{cuda_memcpy_on, MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::{CudaEvent, GpuDeviceCtx},
};
use openvm_instructions::VM_DIGEST_WIDTH;
use openvm_stark_backend::{p3_util::log2_ceil_usize, prover::AirProvingContext};
use p3_field::PrimeCharacteristicRing;

use super::{poseidon2::SharedBuffer, GpuMemoryCellType, Poseidon2PeripheryChipGPU};

pub mod cuda;
use cuda::merkle_tree::*;

type H = [F; VM_DIGEST_WIDTH];
/// Width of `((u32, u32), TimestampedValues<F, BLOCK_FE_WIDTH>)` in u32 units.
/// = 2 (key) + 1 (timestamp) + BLOCK_FE_WIDTH (values)
pub const TIMESTAMPED_BLOCK_WIDTH: usize = 3 + BLOCK_FE_WIDTH;
/// Width of one merkle touched-block record in u32 units.
/// = 2 (key) + 1 (is_dirty) + VM_DIGEST_WIDTH (values); see `MemoryMerkleRecord`.
pub const MERKLE_TOUCHED_BLOCK_WIDTH: usize = 3 + VM_DIGEST_WIDTH;
pub(crate) const OMITTED_BOTTOM_LEVELS: usize = 3;

#[derive(Debug)]
pub(crate) enum InitialMerkleBuild {
    DensePrefix(usize),
    SparsePages(SparseMerklePlan),
}

#[derive(Debug)]
pub(crate) struct SparseMerklePlan {
    /// Node labels grouped by height, with the root first. Digests use the same ordering.
    labels: Vec<u32>,
    /// For each conceptual node height, `[start, count]` in `labels`.
    levels: Vec<[u32; 2]>,
    base_height: usize,
}

/// Number of leaf digests a subtree must store densely so that every page of `addr_sp` that may
/// contain non-zero data lies inside it, rounded up to a power of two. Returns 0 for an empty
/// address space and at least 1 otherwise, so that stored-node reads during
/// [`MemoryMerkleTree::update_with_touched_blocks`] stay in bounds. All leaves at or beyond the
/// watermark are guaranteed zero (see
/// [`TouchedPages`](openvm_circuit::system::memory::online::TouchedPages)), which is exactly the
/// invariant [`MemoryMerkleTree::build_async`] requires of its `addr_space_size` argument.
pub(crate) fn touched_leaf_watermark(memory: &AddressMap, addr_sp: usize) -> usize {
    let raw_len = memory.mem[addr_sp].as_slice().len();
    if raw_len == 0 {
        return 0;
    }
    let cell_size = memory.config[addr_sp].layout.size();
    let watermark_bytes = memory.touched_pages[addr_sp]
        .touched_byte_ranges(raw_len)
        .last()
        .map_or(0, |&(_, end)| end);
    watermark_bytes
        .div_ceil(cell_size * VM_DIGEST_WIDTH)
        .next_power_of_two()
}

/// Chooses between the fast dense-prefix builder and a page-dense, upper-level-sparse builder.
///
/// The sparse representation starts at height [`OMITTED_BOTTOM_LEVELS`], so each base node is the
/// root of eight raw-memory leaves. Touched 4 KiB pages therefore remain dense and coalesced while
/// gaps between pages are represented by precomputed zero hashes. Dense-prefix construction stays
/// preferable for the ordinary contiguous guest image because it avoids sparse labels and lookups.
pub(crate) fn initial_merkle_build(
    memory: &AddressMap,
    addr_sp: usize,
    full_height: usize,
) -> InitialMerkleBuild {
    let raw_len = memory.mem[addr_sp].as_slice().len();
    let dense_size = touched_leaf_watermark(memory, addr_sp);
    if raw_len == 0 || dense_size == 0 {
        return InitialMerkleBuild::DensePrefix(dense_size);
    }

    let ranges = memory.touched_pages[addr_sp].touched_byte_ranges(raw_len);
    if ranges.is_empty() || (ranges.len() == 1 && ranges[0].0 == 0) {
        return InitialMerkleBuild::DensePrefix(dense_size);
    }

    let cell_size = memory.config[addr_sp].layout.size();
    let leaf_bytes = cell_size * VM_DIGEST_WIDTH;
    let base_leaf_count = 1usize << OMITTED_BOTTOM_LEVELS;
    let base_node_bytes = leaf_bytes * base_leaf_count;
    debug_assert_eq!(PAGE_SIZE % base_node_bytes, 0);

    let mut by_height = vec![Vec::<u32>::new(); full_height + 1];
    for (start, end) in ranges {
        let first = start / base_node_bytes;
        let last = end.div_ceil(base_node_bytes);
        by_height[OMITTED_BOTTOM_LEVELS].extend((first..last).map(|label| label as u32));
    }
    by_height[OMITTED_BOTTOM_LEVELS].sort_unstable();
    by_height[OMITTED_BOTTOM_LEVELS].dedup();
    for height in OMITTED_BOTTOM_LEVELS + 1..=full_height {
        let mut parents = by_height[height - 1]
            .iter()
            .map(|label| label >> 1)
            .collect::<Vec<_>>();
        parents.dedup();
        by_height[height] = parents;
    }

    let sparse_nodes = by_height.iter().map(Vec::len).sum::<usize>();
    let dense_height = log2_ceil_usize(dense_size);
    let dense_path_len = full_height - dense_height;
    let dense_layout = MemoryMerkleSubTree::layout_for_height(dense_height);
    let dense_nodes = MemoryMerkleSubTree::buffer_len(dense_size, dense_path_len, dense_layout);
    // Sparse nodes carry a u32 label and use binary-search lookup during updates. Require a
    // meaningful reduction rather than selecting sparse storage for small holes in a dense image.
    if sparse_nodes.saturating_mul(2) >= dense_nodes {
        return InitialMerkleBuild::DensePrefix(dense_size);
    }

    let mut labels = Vec::with_capacity(sparse_nodes);
    let mut levels = vec![[0u32; 2]; full_height + 1];
    for height in (OMITTED_BOTTOM_LEVELS..=full_height).rev() {
        let start = labels.len();
        labels.extend_from_slice(&by_height[height]);
        levels[height] = [start as u32, by_height[height].len() as u32];
    }
    debug_assert_eq!(levels[full_height], [0, 1]);
    InitialMerkleBuild::SparsePages(SparseMerklePlan {
        labels,
        levels,
        base_height: OMITTED_BOTTOM_LEVELS,
    })
}

/// Exact number of distinct internal merkle nodes (heights `1..=tree_height`) on the
/// paths from a sorted stream of global leaf indices to the root: the first leaf's path
/// has `tree_height` nodes, and each subsequent leaf adds nodes only below the height at
/// which its path merges with the previous one (`log2` of the index xor). Exact, not an
/// upper bound.
#[derive(Default)]
pub(crate) struct SpanningNodeCounter {
    prev_leaf_index: Option<u64>,
    pub(crate) nodes: usize,
}

impl SpanningNodeCounter {
    #[inline]
    pub(crate) fn push(&mut self, leaf_index: u64, tree_height: usize) {
        self.nodes += match self.prev_leaf_index {
            None => tree_height,
            Some(prev) => {
                debug_assert!(prev <= leaf_index);
                let xor = prev ^ leaf_index;
                if xor == 0 {
                    0
                } else {
                    xor.ilog2() as usize
                }
            }
        };
        self.prev_leaf_index = Some(leaf_index);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum MemoryMerkleSubTreeLayout {
    Full = 0,
    OmitBottomLevels = 1,
    SparsePages = 2,
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
/// - `SparsePages` subtrees store sorted `(height, label) -> digest` levels for touched pages and
///   their ancestors; absent labels resolve to the precomputed zero hash for that height.
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
    cell_type: GpuMemoryCellType,
    sparse_labels: DeviceBuffer<u32>,
    sparse_levels: DeviceBuffer<[u32; 2]>,
    /// Shared handle to the initial-memory buffer (`d_data`) from [`Self::build_async`], or
    /// `None` for empty/dummy subtrees. Co-owning the buffer keeps the host from freeing it: under
    /// `OmitBottomLevels` and `SparsePages`, the omitted levels aren't in `buf` and are recomputed
    /// from this buffer during [`MemoryMerkleTree::update_with_touched_blocks`]
    /// (`recompute_omitted_node` in `merkle_tree.cu`).
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
            MemoryMerkleSubTreeLayout::SparsePages => {
                unreachable!("sparse buffers are sized by SparseMerklePlan")
            }
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
    fn new(
        addr_space_size: usize,
        max_size: usize,
        cell_type: GpuMemoryCellType,
        device_ctx: &GpuDeviceCtx,
    ) -> Self {
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
        assert!(
            addr_space_size == 0 || cell_type != GpuMemoryCellType::Unsupported,
            "nonempty CUDA memory address spaces require U8, U16, or Field32 cells"
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
            cell_type,
            sparse_labels: DeviceBuffer::new(),
            sparse_levels: DeviceBuffer::new(),
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
            cell_type: GpuMemoryCellType::Unsupported,
            sparse_labels: DeviceBuffer::new(),
            sparse_levels: DeviceBuffer::new(),
            initial_data: None,
        }
    }

    fn new_sparse(
        full_height: usize,
        plan: &SparseMerklePlan,
        cell_type: GpuMemoryCellType,
        device_ctx: &GpuDeviceCtx,
    ) -> Self {
        debug_assert_eq!(plan.levels[full_height], [0, 1]);
        assert!(
            cell_type != GpuMemoryCellType::Unsupported,
            "nonempty CUDA memory address spaces require U8, U16, or Field32 cells"
        );
        tracing::debug!(
            nodes = plan.labels.len(),
            full_height,
            "Creating a page-sparse subtree buffer"
        );
        Self {
            build_completion_event: None,
            buf: DeviceBuffer::<H>::with_capacity_on(plan.labels.len(), device_ctx),
            height: full_height,
            path_len: 0,
            layout: MemoryMerkleSubTreeLayout::SparsePages,
            cell_type,
            sparse_labels: plan.labels.to_device_on(device_ctx).unwrap(),
            sparse_levels: plan.levels.to_device_on(device_ctx).unwrap(),
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
            MemoryMerkleSubTreeLayout::SparsePages => {
                unreachable!("sparse subtrees do not use heap height")
            }
        }
    }

    /// Builds the Merkle subtree on the provided `GpuDeviceCtx` stream.
    /// Also reconstructs the vertical path if `path_len > 0`, and records a completion event.
    pub fn build_async(
        &mut self,
        d_data: Arc<DeviceBuffer<u8>>,
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
                    self.cell_type as u8,
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

    fn build_sparse_async(
        &mut self,
        d_data: Arc<DeviceBuffer<u8>>,
        plan: &SparseMerklePlan,
        zero_hash: &DeviceBuffer<H>,
        device_ctx: &GpuDeviceCtx,
    ) {
        let event = CudaEvent::new().unwrap();
        self.initial_data = Some(d_data.clone());
        unsafe {
            build_sparse_merkle_subtree(
                &d_data,
                &self.buf,
                &self.sparse_labels,
                &plan.levels,
                plan.base_height,
                self.height,
                self.cell_type as u8,
                zero_hash,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
            event.record(device_ctx.stream.as_raw()).unwrap();
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
    /// `build` selects either a dense prefix or page-dense sparse construction. Both require every
    /// unrepresented leaf to be all-zero, as guaranteed by the touched-page metadata.
    ///
    /// **Note:** the caller MUST ENSURE that `d_data` lives long enough to be there
    /// when the enqueued task actually starts. Moreover, when the subtree uses the
    /// `OmitBottomLevels` or `SparsePages` layout, `d_data` is also re-read during
    /// [`Self::update_with_touched_blocks`] to recompute the omitted bottom levels, so it must
    /// remain valid until that update completes — not just until the build kernel runs. See
    /// [`MemoryMerkleSubTree`]'s `initial_data` field for details.
    pub(crate) fn build_async(
        &mut self,
        d_data: Arc<DeviceBuffer<u8>>,
        addr_space: usize,
        build: InitialMerkleBuild,
    ) {
        if addr_space < ADDR_SPACE_OFFSET as usize {
            return;
        }
        let addr_space_idx = addr_space - ADDR_SPACE_OFFSET as usize;
        if addr_space < self.mem_config.addr_spaces.len() && addr_space_idx == self.subtrees.len() {
            let full_height = self.zero_hash.len() - 1;
            let cell_type = self.mem_config.addr_spaces[addr_space].layout.into();
            let mut subtree = match &build {
                InitialMerkleBuild::DensePrefix(addr_space_size) => {
                    assert!(
                        *addr_space_size
                            <= self.mem_config.addr_spaces[addr_space].num_cells / VM_DIGEST_WIDTH,
                        "subtree size exceeds the address space's configured leaf count"
                    );
                    MemoryMerkleSubTree::new(
                        *addr_space_size,
                        1 << full_height, /* label_max_bits */
                        cell_type,
                        &self.device_ctx,
                    )
                }
                InitialMerkleBuild::SparsePages(plan) => {
                    MemoryMerkleSubTree::new_sparse(full_height, plan, cell_type, &self.device_ctx)
                }
            };
            match &build {
                InitialMerkleBuild::DensePrefix(_) => {
                    subtree.build_async(d_data, &self.zero_hash, &self.device_ctx)
                }
                InitialMerkleBuild::SparsePages(plan) => {
                    subtree.build_sparse_async(d_data, plan, &self.zero_hash, &self.device_ctx)
                }
            }
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
    /// `d_touched_blocks` consists of `(as, ptr, is_dirty, [F; VM_DIGEST_WIDTH])`.
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
            let cell_types = self
                .subtrees
                .iter()
                .map(|s| s.cell_type as u8)
                .collect::<Vec<_>>();
            let initial_data_ptrs = self
                .subtrees
                .iter()
                .map(|s| s.initial_data.as_ref().map_or(0, |b| b.as_ptr() as usize))
                .collect::<Vec<_>>();
            let sparse_label_ptrs = self
                .subtrees
                .iter()
                .map(|s| s.sparse_labels.as_ptr() as usize)
                .collect::<Vec<_>>();
            let sparse_level_ptrs = self
                .subtrees
                .iter()
                .map(|s| s.sparse_levels.as_ptr() as usize)
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
                    &cell_types,
                    &initial_data_ptrs,
                    &sparse_label_ptrs,
                    &sparse_level_ptrs,
                    unpadded_height,
                    &self.hasher_buffer,
                    &self.device_ctx,
                )
                .unwrap();
            }

            if empty_touched_blocks {
                // The artificial touch (see the caller) seeds the walk so the root pair
                // exists, but no boundary row supplies the leaf's claim, so the height-1
                // initial row (the last row) must treat the leaf as *untouched*: consume
                // nothing, i.e. `left_child_mode = 0` (the kernel wrote 1 for the seeded
                // leaf).
                let mut output_vec = output.buffer().to_host_on(&self.device_ctx).unwrap();
                let left_child_mode_col = std::mem::offset_of!(
                    MemoryMerkleCols<F, VM_DIGEST_WIDTH>,
                    left_child_mode
                ) / std::mem::size_of::<F>();
                output_vec[unpadded_height - 1 + left_child_mode_col * padded_height] = F::ZERO;
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
                persistent::DirtyLeaves,
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
        riscv::{MEMORY_AS, REGISTER_AS},
        DEFERRAL_AS, PUBLIC_VALUES_AS, VM_DIGEST_WIDTH,
    };
    use openvm_stark_backend::{interaction::PermutationCheckBus, prover::MatrixDimensions};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use p3_field::{PrimeCharacteristicRing, PrimeField32};
    use rand::Rng;

    use super::{
        initial_merkle_build, GpuMemoryCellType, InitialMerkleBuild, MemoryMerkleSubTree,
        MemoryMerkleSubTreeLayout, MemoryMerkleTree, SpanningNodeCounter, OMITTED_BOTTOM_LEVELS,
    };
    use crate::{
        arch::testing::{MEMORY_MERKLE_BUS, POSEIDON2_DIRECT_BUS},
        system::cuda::Poseidon2PeripheryChipGPU,
    };

    /// Builds the device merkle-record words and the exact unpadded merkle trace height
    /// they imply. Dirtiness is per *write*; the tests treat exactly the leaves whose
    /// values changed as written (the minimal valid dirty set). `touched_blocks` must be
    /// sorted by (address space, pointer).
    fn build_records_and_height(
        initial_memory: &GuestMemory,
        touched_blocks: &[((u32, u32), TimestampedValues<F, VM_DIGEST_WIDTH>)],
        mem_config: &MemoryConfig,
    ) -> (Vec<u32>, usize) {
        let md = mem_config.memory_dimensions();
        let tree_height = md.overall_height();
        let mut words = Vec::with_capacity(touched_blocks.len() * MERKLE_TOUCHED_BLOCK_WIDTH);
        let mut touched_nodes = SpanningNodeCounter::default();
        let mut dirty_nodes = SpanningNodeCounter::default();
        let mut dirty_leaves = 0usize;
        for ((address_space, ptr), ts_values) in touched_blocks {
            let init_values: [F; VM_DIGEST_WIDTH] = std::array::from_fn(|i| unsafe {
                initial_memory
                    .memory
                    .get_f::<F>(*address_space, *ptr + i as u32)
            });
            let is_dirty = u32::from(init_values != ts_values.values);
            let leaf_index = md.label_to_index((*address_space, *ptr / VM_DIGEST_WIDTH as u32));
            touched_nodes.push(leaf_index, tree_height);
            if is_dirty != 0 {
                dirty_nodes.push(leaf_index, tree_height);
                dirty_leaves += 1;
            }
            words.push(*address_space);
            words.push(*ptr);
            words.push(is_dirty);
            for &v in &ts_values.values {
                words.push(unsafe { std::mem::transmute::<F, u32>(v) });
            }
        }
        let rows = touched_nodes.nodes
            + if dirty_leaves == 0 {
                1
            } else {
                dirty_nodes.nodes
            };
        (words, rows)
    }

    #[test]
    fn test_cuda_merkle_subtree_layout_and_buffer_sizes() {
        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let max_size = 1 << (OMITTED_BOTTOM_LEVELS + 3);

        let below = MemoryMerkleSubTree::new(
            1 << (OMITTED_BOTTOM_LEVELS - 1),
            max_size,
            GpuMemoryCellType::U16,
            &device_ctx,
        );
        assert_eq!(below.layout, MemoryMerkleSubTreeLayout::Full);
        assert_eq!(
            below.buf.len(),
            below.path_len + (2 * (1 << (OMITTED_BOTTOM_LEVELS - 1)) - 1)
        );

        let equal = MemoryMerkleSubTree::new(
            1 << OMITTED_BOTTOM_LEVELS,
            max_size,
            GpuMemoryCellType::U16,
            &device_ctx,
        );
        assert_eq!(equal.layout, MemoryMerkleSubTreeLayout::Full);
        assert_eq!(
            equal.buf.len(),
            equal.path_len + (2 * (1 << OMITTED_BOTTOM_LEVELS) - 1)
        );

        let above = MemoryMerkleSubTree::new(
            1 << (OMITTED_BOTTOM_LEVELS + 1),
            max_size,
            GpuMemoryCellType::U16,
            &device_ctx,
        );
        let full_len = above.path_len + (2 * (1 << (OMITTED_BOTTOM_LEVELS + 1)) - 1);
        let optimized_len = above.path_len + (2 * (1 << 1) - 1);
        assert_eq!(above.layout, MemoryMerkleSubTreeLayout::OmitBottomLevels);
        assert_eq!(above.buf.len(), optimized_len);
        assert!(above.buf.len() < full_len);
    }

    #[test]
    #[should_panic(expected = "nonempty CUDA memory address spaces require")]
    fn test_nonempty_unsupported_cuda_merkle_subtree_is_rejected() {
        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let _ = MemoryMerkleSubTree::new(
            1,
            1 << OMITTED_BOTTOM_LEVELS,
            GpuMemoryCellType::Unsupported,
            &device_ctx,
        );
    }

    #[test]
    fn test_cuda_merkle_tree_cpu_gpu_root_equivalence() {
        let mut rng = create_seeded_rng();
        let mem_config = {
            let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
            let max_ptr_bits = 16;
            let max_cells = 1 << max_ptr_bits;
            // REGISTER_AS uses u16 storage cells.
            addr_spaces[REGISTER_AS as usize].num_cells = 32 * size_of::<u64>() / U16_CELL_SIZE;
            addr_spaces[MEMORY_AS as usize].num_cells = max_cells;
            addr_spaces[DEFERRAL_AS as usize].num_cells = max_cells;
            addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = max_cells;
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
            gpu_merkle_tree.build_async(
                mem_slice.clone(),
                i,
                InitialMerkleBuild::DensePrefix(
                    mem_config.addr_spaces[i].num_cells / VM_DIGEST_WIDTH,
                ),
            );
        }
        assert_eq!(
            gpu_merkle_tree.subtrees[REGISTER_AS as usize - 1].layout,
            MemoryMerkleSubTreeLayout::OmitBottomLevels
        );
        assert_eq!(
            gpu_merkle_tree.subtrees[MEMORY_AS as usize - 1].layout,
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
        // Dirtiness is per *write*; the test scenario treats exactly the leaves whose
        // values changed as written (the minimal valid dirty set).
        let dirty_leaves: DirtyLeaves = touched_ptrs
            .iter()
            .zip(new_data.iter())
            .filter(|(&(address_space, ptr), values)| {
                let init_values: [F; VM_DIGEST_WIDTH] = std::array::from_fn(|i| unsafe {
                    initial_memory
                        .memory
                        .get_f::<F>(address_space, ptr + i as u32)
                });
                init_values != **values
            })
            .map(|(&key, _)| key)
            .collect();
        cpu_merkle_tree.finalize(
            &cpu_hasher_chip,
            &(touched_ptrs
                .iter()
                .copied()
                .zip(new_data.iter().copied())
                .collect()),
            &dirty_leaves,
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
        let (merkle_records, unpadded_height) =
            build_records_and_height(&initial_memory, &touched_blocks, &mem_config);
        let d_touched_blocks = merkle_records
            .to_device_on(&gpu_merkle_tree.device_ctx)
            .unwrap();
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
            // REGISTER_AS uses u16 storage cells.
            addr_spaces[REGISTER_AS as usize].num_cells = 32 * size_of::<u64>() / U16_CELL_SIZE;
            addr_spaces[MEMORY_AS as usize].num_cells = max_cells;
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
            gpu_merkle_tree.build_async(
                mem_slice.clone(),
                i,
                InitialMerkleBuild::DensePrefix(
                    mem_config.addr_spaces[i].num_cells / VM_DIGEST_WIDTH,
                ),
            );
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
        let mut new_data = touched_ptrs
            .iter()
            .map(|_| std::array::from_fn(|_| F::from_u32(rng.random_range(0..F::ORDER_U32))))
            .collect::<Vec<[F; VM_DIGEST_WIDTH]>>();
        // Make every third touched leaf *clean* (final values equal to initial ones):
        // random values are almost surely dirty, and the mixed case is what exercises
        // skipped final rows, initial-state borrows, and the child modes.
        for (i, (&(address_space, ptr), values)) in
            touched_ptrs.iter().zip(new_data.iter_mut()).enumerate()
        {
            if i % 3 == 0 {
                *values = std::array::from_fn(|j| unsafe {
                    initial_memory
                        .memory
                        .get_f::<F>(address_space, ptr + j as u32)
                });
            }
        }
        let new_data = new_data;
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
        // Dirtiness is per *write*; the test scenario treats exactly the leaves whose
        // values changed as written (the minimal valid dirty set).
        let dirty_leaves: DirtyLeaves = final_partition
            .iter()
            .filter(|((address_space, ptr), values)| {
                let init_values: [F; VM_DIGEST_WIDTH] = std::array::from_fn(|i| unsafe {
                    initial_memory
                        .memory
                        .get_f::<F>(*address_space, *ptr + i as u32)
                });
                init_values != **values
            })
            .map(|(&key, _)| key)
            .collect();
        cpu_merkle_chip.finalize(
            &initial_memory.memory,
            &final_partition,
            &dirty_leaves,
            &cpu_hasher_chip,
        );
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
        let (merkle_records, unpadded_height) =
            build_records_and_height(&initial_memory, &touched_blocks, &mem_config);
        let d_touched_blocks = merkle_records
            .to_device_on(&gpu_merkle_tree.device_ctx)
            .unwrap();
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

    /// Page-sparse variant of the equivalence tests: the large address spaces contain a low prefix
    /// and data in their final page. A prefix-only builder would therefore hash the entire address
    /// space, while the page builder stores only those page subtrees and their ancestors. The
    /// update then touches dirty and clean blocks across the whole pointer range. Checks the
    /// initial root, final root, and full trace against the CPU reference.
    #[test]
    fn test_cuda_merkle_tree_sparse_pages_cpu_gpu_equivalence() {
        let mut rng = create_seeded_rng();
        let mem_config = {
            let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
            let max_ptr_bits = 16;
            let max_cells = 1 << max_ptr_bits;
            // REGISTER_AS uses u16 storage cells.
            addr_spaces[REGISTER_AS as usize].num_cells = 32 * size_of::<u64>() / U16_CELL_SIZE;
            addr_spaces[MEMORY_AS as usize].num_cells = max_cells;
            addr_spaces[DEFERRAL_AS as usize].num_cells = max_cells;
            MemoryConfig::new(2, addr_spaces, max_ptr_bits, 29, 17)
        };

        // Fill REGISTER_AS fully, plus a low prefix and the final cell of each large address
        // space. The distant page forces selection of SparsePages (asserted below).
        let prefix_cells = 1 << 10;
        let mut initial_memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
        for (idx, space) in mem_config.addr_spaces.iter().enumerate() {
            let num_filled = if idx == REGISTER_AS as usize {
                space.num_cells
            } else {
                space.num_cells.min(prefix_cells)
            };
            unsafe {
                match space.layout {
                    MemoryCellType::Null => {}
                    MemoryCellType::U8 => {
                        for i in 0..num_filled {
                            initial_memory.write::<u8, 1>(idx as u32, i as u32, [rng.random()]);
                        }
                    }
                    MemoryCellType::U16 => {
                        for i in 0..num_filled {
                            initial_memory.write::<u16, 1>(idx as u32, i as u32, [rng.random()]);
                        }
                    }
                    MemoryCellType::U32 => {
                        for i in 0..num_filled {
                            initial_memory.write::<u32, 1>(idx as u32, i as u32, [rng.random()]);
                        }
                    }
                    MemoryCellType::F { .. } => {
                        for i in 0..num_filled {
                            initial_memory.write::<F, 1>(
                                idx as u32,
                                i as u32,
                                [F::from_u32(rng.random_range(0..F::ORDER_U32))],
                            );
                        }
                    }
                }
                if (idx == MEMORY_AS as usize || idx == DEFERRAL_AS as usize)
                    && space.num_cells != 0
                {
                    let ptr = (space.num_cells - 1) as u32;
                    match space.layout {
                        MemoryCellType::U8 => initial_memory.write::<u8, 1>(idx as u32, ptr, [1]),
                        MemoryCellType::U16 => initial_memory.write::<u16, 1>(idx as u32, ptr, [1]),
                        MemoryCellType::U32 => initial_memory.write::<u32, 1>(idx as u32, ptr, [1]),
                        MemoryCellType::F { .. } => {
                            initial_memory.write::<F, 1>(idx as u32, ptr, [F::ONE])
                        }
                        MemoryCellType::Null => {}
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
            gpu_merkle_tree.build_async(
                mem_slice.clone(),
                i,
                initial_merkle_build(
                    &initial_memory.memory,
                    i,
                    mem_config.memory_dimensions().address_height,
                ),
            );
        }
        // The large address spaces must actually select page sparsity, or this test degenerates
        // into the dense equivalence tests.
        for addr_space in [MEMORY_AS as usize, DEFERRAL_AS as usize] {
            let subtree = &gpu_merkle_tree.subtrees[addr_space - 1];
            assert_eq!(subtree.layout, MemoryMerkleSubTreeLayout::SparsePages);
        }
        gpu_merkle_tree.finalize();

        let cpu_hasher_chip = Poseidon2PeripheryChip::new(vm_poseidon2_config(), 3);
        let cpu_merkle_tree = MerkleTree::<F, VM_DIGEST_WIDTH>::from_memory(
            &initial_memory.memory,
            &mem_config.memory_dimensions(),
            &cpu_hasher_chip,
        );
        assert_eq!(
            cpu_merkle_tree.root(),
            gpu_merkle_tree
                .top_roots
                .to_host_on(&gpu_merkle_tree.device_ctx)
                .unwrap()[0],
            "initial roots diverge"
        );

        // Touch ~1/3 of digest-aligned pointers across the whole pointer range, always including
        // the very last leaf of the large address spaces.
        let touched_ptrs = mem_config
            .addr_spaces
            .iter()
            .enumerate()
            .flat_map(|(i, cnf)| {
                let mut ptrs = Vec::new();
                let num_leaves = cnf.num_cells / VM_DIGEST_WIDTH;
                for j in 0..num_leaves {
                    let is_last_large = j + 1 == num_leaves
                        && (i == MEMORY_AS as usize || i == DEFERRAL_AS as usize);
                    if is_last_large || rng.random_bool(0.333) {
                        ptrs.push((i as u32, (j * VM_DIGEST_WIDTH) as u32));
                    }
                }
                ptrs
            })
            .collect::<Vec<_>>();
        let mut new_data = touched_ptrs
            .iter()
            .map(|_| std::array::from_fn(|_| F::from_u32(rng.random_range(0..F::ORDER_U32))))
            .collect::<Vec<[F; VM_DIGEST_WIDTH]>>();
        // Make every third touched leaf *clean* (final values equal to initial ones). Above the
        // watermark the initial values are all zero, so this exercises clean zero-hash leaves.
        for (i, (&(address_space, ptr), values)) in
            touched_ptrs.iter().zip(new_data.iter_mut()).enumerate()
        {
            if i % 3 == 0 {
                *values = std::array::from_fn(|j| unsafe {
                    initial_memory
                        .memory
                        .get_f::<F>(address_space, ptr + j as u32)
                });
            }
        }
        let new_data = new_data;
        assert!(!touched_ptrs.is_empty());

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
        // Dirtiness is per *write*; the test scenario treats exactly the leaves whose
        // values changed as written (the minimal valid dirty set).
        let dirty_leaves: DirtyLeaves = final_partition
            .iter()
            .filter(|((address_space, ptr), values)| {
                let init_values: [F; VM_DIGEST_WIDTH] = std::array::from_fn(|i| unsafe {
                    initial_memory
                        .memory
                        .get_f::<F>(*address_space, *ptr + i as u32)
                });
                init_values != **values
            })
            .map(|(&key, _)| key)
            .collect();
        cpu_merkle_chip.finalize(
            &initial_memory.memory,
            &final_partition,
            &dirty_leaves,
            &cpu_hasher_chip,
        );
        let cpu_ctx = cpu_merkle_chip.generate_proving_ctx::<SC>();

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
        let (merkle_records, unpadded_height) =
            build_records_and_height(&initial_memory, &touched_blocks, &mem_config);
        let d_touched_blocks = merkle_records
            .to_device_on(&gpu_merkle_tree.device_ctx)
            .unwrap();
        gpu_hasher_chip.prepare_records(unpadded_height);
        let merkle_ctx =
            gpu_merkle_tree.update_with_touched_blocks(unpadded_height, &d_touched_blocks, false);

        // Final roots and the full trace must match the CPU reference.
        let gpu_final_root = gpu_merkle_tree
            .top_roots
            .to_host_on(&gpu_merkle_tree.device_ctx)
            .unwrap()[0];
        let width = cpu_ctx.common_main.width;
        let height = cpu_ctx.common_main.values.len() / width;
        let cpu_vals = &cpu_ctx.common_main.values;
        // The CPU chip's final root is the parent hash of the height-section final root row
        // (row 1 by the AIR's pinning of the first two rows).
        let gpu_trace = &merkle_ctx.common_main;
        assert_eq!(gpu_trace.width(), width, "trace width mismatch");
        assert_eq!(gpu_trace.height(), height, "trace (padded) height mismatch");
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
        // Cross-check the final root through an independently finalized CPU tree.
        let mut cpu_final_tree = MerkleTree::<F, VM_DIGEST_WIDTH>::from_memory(
            &initial_memory.memory,
            &mem_config.memory_dimensions(),
            &cpu_hasher_chip,
        );
        cpu_final_tree.finalize(
            &cpu_hasher_chip,
            &final_partition,
            &dirty_leaves,
            &mem_config.memory_dimensions(),
        );
        assert_eq!(cpu_final_tree.root(), gpu_final_root, "final roots diverge");
    }
}
