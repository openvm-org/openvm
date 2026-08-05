use std::sync::Arc;

use openvm_circuit::{
    arch::{AddressSpaceHostLayout, MemoryConfig, ADDR_SPACE_OFFSET, BLOCK_FE_WIDTH},
    system::{
        memory::{persistent::BLOCKS_PER_LEAF, AddressMap},
        TouchedBlock, TouchedMemory,
    },
};
use openvm_circuit_primitives::Chip;
use openvm_cuda_backend::{prelude::F, GpuBackend};
use openvm_cuda_common::{
    copy::{cuda_memcpy_on, MemCopyD2H, MemCopyH2D},
    d_buffer::{DeviceBuffer, DeviceBufferView},
    memory_manager::MemTracker,
    pinned,
    stream::GpuDeviceCtx,
};
use openvm_instructions::VM_DIGEST_WIDTH;
#[cfg(feature = "parallel")]
use openvm_stark_backend::p3_maybe_rayon::prelude::IndexedParallelIterator;
use openvm_stark_backend::{
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice, ParallelSliceMut},
    prover::AirProvingContext,
};
use tracing::instrument;

use super::{
    boundary::BoundaryChipGPU,
    merkle_tree::{MemoryMerkleTree, SpanningNodeCounter, MERKLE_TOUCHED_BLOCK_WIDTH},
    Poseidon2PeripheryChipGPU,
};
use crate::{
    arch::cuda::postflight::{GpuPostflightError, GpuPostflightTranscript},
    cuda_abi::inventory,
    system::memory::online::LinearMemory,
};

/// Chunk size for the parallel pack into the upload staging buffer.
const UPLOAD_PACK_CHUNK: usize = 8 << 20;

#[inline]
fn copy_into_upload_staging(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    if src.len() <= UPLOAD_PACK_CHUNK {
        dst.copy_from_slice(src);
    } else {
        dst.par_chunks_mut(UPLOAD_PACK_CHUNK)
            .zip(src.par_chunks(UPLOAD_PACK_CHUNK))
            .for_each(|(dst, src)| dst.copy_from_slice(src));
    }
}

// The CUDA merge kernel in `inventory.cu` is hardcoded to a 2-way merge of
// `<IN_BLOCK_SIZE=4, 1>` records into `<OUT_BLOCK_SIZE=8, 2>` records, so the only
// supported `(BLOCK_FE_WIDTH, VM_DIGEST_WIDTH)` shape is `(4, 8)`.
const _: () = assert!(
    BLOCK_FE_WIDTH == 4 && VM_DIGEST_WIDTH == 8,
    "CUDA memory inventory only supports (BLOCK_FE_WIDTH, VM_DIGEST_WIDTH) == (4, 8)"
);

// `TouchedBlock` must exactly match the CUDA `MemoryTouchedBlock` layout so
// the merge path can upload the vector's bytes without repacking.
const _: () = assert!(
    std::mem::size_of::<TouchedBlock>() == (4 + BLOCK_FE_WIDTH) * std::mem::size_of::<u32>()
        && std::mem::offset_of!(TouchedBlock, address_space) == 0
        && std::mem::offset_of!(TouchedBlock, ptr) == std::mem::size_of::<u32>()
        && std::mem::offset_of!(TouchedBlock, is_dirty) == 2 * std::mem::size_of::<u32>()
        && std::mem::offset_of!(TouchedBlock, timestamp) == 3 * std::mem::size_of::<u32>()
        && std::mem::offset_of!(TouchedBlock, values) == 4 * std::mem::size_of::<u32>(),
    "TouchedBlock must match MemoryTouchedBlock in system/memory/touched_block.cuh"
);

pub struct MemoryInventoryGPU {
    pub device_ctx: GpuDeviceCtx,
    pub boundary: BoundaryChipGPU,
    pub merkle_tree: MemoryMerkleTree,
    pub hasher_chip: Arc<Poseidon2PeripheryChipGPU>,
    pub initial_memory: Vec<Arc<DeviceBuffer<u8>>>,
    pub merkle_records: Option<DeviceBuffer<u32>>,
    upload_staging: PinnedStaging,
    #[cfg(feature = "metrics")]
    pub(super) unpadded_merkle_height: usize,
}

/// Page-locked host staging for the per-segment memory-image upload.
///
/// Copies from pageable memory run at staging-pipeline speed and only return
/// once the source is consumed; copies from registered memory take the DMA
/// fast path (~2x) and return immediately. Registering the guest memory
/// itself would tie a registration to an allocation this module does not own
/// (freed-while-registered is undefined), so the image is packed into this
/// owned, once-registered buffer instead: the pack memcpy is parallel and
/// fully consumes the guest memory before returning, so preflight may mutate
/// it right away, while the DMA reads the staging asynchronously.
#[derive(Default)]
struct PinnedStaging {
    buf: Vec<u8>,
    registered: bool,
}

impl PinnedStaging {
    /// Returns a staging slice of exactly `len` bytes, growing and
    /// re-registering the underlying buffer if needed.
    fn ensure(&mut self, len: usize) -> &mut [u8] {
        if self.buf.len() < len {
            if self.registered {
                pinned::unregister_region(self.buf.as_mut_ptr());
                self.registered = false;
            }
            self.buf = vec![0u8; len];
            self.registered = pinned::register_region(self.buf.as_mut_ptr(), len);
            if !self.registered {
                tracing::debug!("memory-image staging stays pageable ({len} bytes)");
            }
        }
        &mut self.buf[..len]
    }
}

impl Drop for PinnedStaging {
    fn drop(&mut self) {
        if self.registered {
            pinned::unregister_region(self.buf.as_mut_ptr());
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryInventoryRecord<const CHUNK: usize, const BLOCKS: usize> {
    address_space: u32,
    ptr: u32,
    is_dirty: u32,
    timestamps: [u32; BLOCKS],
    values: [u32; CHUNK],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryMerkleRecord {
    address_space: u32,
    ptr: u32,
    is_dirty: u32,
    values: [u32; VM_DIGEST_WIDTH],
}

impl MemoryInventoryGPU {
    #[inline]
    fn field_to_raw_u32(value: F) -> u32 {
        unsafe { std::mem::transmute::<F, u32>(value) }
    }

    fn clear_initial_memory(&mut self) {
        // Initial-memory buffers are also owned by in-flight Merkle subtrees.
        // Fence and release those handles before dropping or replacing the
        // buffers themselves.
        self.merkle_tree.drop_subtrees();
        self.boundary.initial_leaves.clear();
        self.initial_memory.clear();
    }

    pub fn new(
        config: MemoryConfig,
        hasher_chip: Arc<Poseidon2PeripheryChipGPU>,
        device_ctx: GpuDeviceCtx,
    ) -> Self {
        Self {
            device_ctx: device_ctx.clone(),
            boundary: BoundaryChipGPU::new(hasher_chip.shared_buffer(), device_ctx.clone()),
            merkle_tree: MemoryMerkleTree::new(config.clone(), hasher_chip.clone(), device_ctx),
            hasher_chip,
            initial_memory: Vec::new(),
            merkle_records: None,
            upload_staging: PinnedStaging::default(),
            #[cfg(feature = "metrics")]
            unpadded_merkle_height: 0,
        }
    }

    #[instrument(name = "set_initial_memory", skip_all)]
    pub fn set_initial_memory(&mut self, initial_memory: &AddressMap) {
        let mem = MemTracker::start("set initial memory");
        if !self.initial_memory.is_empty() {
            self.clear_initial_memory();
        }
        // Only transfer pages that may contain non-zero data; the rest are zero-filled
        // on-device. The merkle kernel reads the full address-space region, so the device
        // buffer is full-size and the skipped pages must read as zero.
        let per_as: Vec<_> = initial_memory
            .get_memory()
            .iter()
            .enumerate()
            .map(|(addr_sp, mem)| {
                let raw = mem.as_slice();
                let runs = initial_memory.touched_pages[addr_sp].touched_byte_ranges(raw.len());
                (raw, runs)
            })
            .collect();
        let total: usize = per_as
            .iter()
            .flat_map(|(_, runs)| runs.iter().map(|(s, e)| e - s))
            .sum();
        let staging = self.upload_staging.ensure(total);
        let mut offset = 0usize;
        for (addr_sp, (raw_mem, runs)) in per_as.into_iter().enumerate() {
            tracing::debug!(
                "Setting initial memory for address space {}: {} bytes, {} touched run(s)",
                addr_sp,
                raw_mem.len(),
                runs.len()
            );
            // Sparse H2D zero-fills every unmarked page, whereas the CPU backend keeps the full
            // host image. Fail at this boundary if a writer mutated memory without marking its
            // page; otherwise the divergence surfaces much later as a memory-bus imbalance.
            // The scan is linear in the host address-space size, so keep it out of release builds.
            #[cfg(any(debug_assertions, feature = "stark-debug"))]
            {
                let mut cursor = 0usize;
                let sentinel = (raw_mem.len(), raw_mem.len());
                for &(start, end) in runs.iter().chain(std::iter::once(&sentinel)) {
                    if let Some(position) =
                        raw_mem[cursor..start].iter().position(|&byte| byte != 0)
                    {
                        let offset = cursor + position;
                        panic!(
                            "address space {addr_sp}: nonzero byte at offset {offset} (page {}) \
                             is outside touched_pages; sparse H2D would zero it on device",
                            offset / crate::system::memory::online::PAGE_SIZE,
                        );
                    }
                    cursor = end;
                }
            }
            self.initial_memory.push(Arc::new(if raw_mem.is_empty() {
                DeviceBuffer::new()
            } else {
                let buf = DeviceBuffer::<u8>::with_capacity_on(raw_mem.len(), &self.device_ctx);
                buf.fill_zero_on(&self.device_ctx)
                    .expect("failed to zero device memory");
                for (start, end) in runs {
                    let dst = &mut staging[offset..offset + (end - start)];
                    offset += end - start;
                    copy_into_upload_staging(dst, &raw_mem[start..end]);
                    // SAFETY: runs are clamped to raw_mem.len() and buf has the same
                    // length; dst is exactly end-start bytes of the staging.
                    unsafe {
                        cuda_memcpy_on::<false, true>(
                            buf.as_mut_ptr().add(start) as *mut std::ffi::c_void,
                            dst.as_ptr() as *const std::ffi::c_void,
                            end - start,
                            &self.device_ctx,
                        )
                        .expect("failed to copy memory to device");
                    }
                }
                buf
            }));
            self.merkle_tree
                .build_async(self.initial_memory[addr_sp].clone(), addr_sp);
        }
        self.boundary.initial_leaves = self
            .initial_memory
            .iter()
            .skip(1)
            .map(|per_as| per_as.as_raw_ptr())
            .collect();
        mem.emit_metrics();
    }

    /// Differential reference for memory traces from host-resident interpreter history.
    #[instrument(name = "generate_proving_ctxs", skip_all)]
    pub fn generate_proving_ctxs(
        &mut self,
        touched_memory: TouchedMemory,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        assert!(
            touched_memory
                .iter()
                .all(|block| block.values.iter().all(|&value| value < F::ORDER_U32)),
            "touched memory contains a non-canonical field value"
        );
        let in_num_records = touched_memory.len();
        if in_num_records == 0 {
            // SAFETY: the exact empty prefix has no backing allocation to keep
            // alive, and the empty path never dereferences the null view.
            return unsafe {
                self.generate_proving_ctxs_from_device(
                    DeviceBufferView {
                        ptr: std::ptr::null(),
                        size: 0,
                    },
                    0,
                )
            };
        }
        let in_bytes = in_num_records * std::mem::size_of::<TouchedBlock>();
        let mut h_in = pinned::take(in_bytes + 4);
        let align_offset = h_in.as_ptr().align_offset(std::mem::size_of::<u32>());
        let dirty_len = align_offset + in_bytes;
        let src: &[u8] =
            unsafe { std::slice::from_raw_parts(touched_memory.as_ptr() as *const u8, in_bytes) };
        let dst = &mut h_in[align_offset..align_offset + in_bytes];
        copy_into_upload_staging(dst, src);
        // SAFETY: 4-aligned by `align_offset`, within the buffer.
        let in_words: &[u32] = unsafe {
            std::slice::from_raw_parts(
                h_in.as_ptr().add(align_offset) as *const u32,
                in_bytes / std::mem::size_of::<u32>(),
            )
        };
        let d_in_records = in_words.to_device_on(&self.device_ctx).unwrap();
        pinned::give_back(h_in, dirty_len);
        // SAFETY: d_in_records owns this same-context view through the call.
        unsafe {
            self.generate_proving_ctxs_from_device_inner(
                d_in_records.view(),
                in_num_records,
                Some(&touched_memory),
            )
        }
    }

    #[instrument(name = "generate_proving_ctxs_from_device", skip_all)]
    pub(super) fn generate_proving_ctxs_from_transcript(
        &mut self,
        transcript: &GpuPostflightTranscript,
    ) -> Result<Vec<AirProvingContext<GpuBackend>>, GpuPostflightError> {
        let (touched_memory, in_num_records) = transcript.touched_blocks_on(&self.device_ctx)?;
        // SAFETY: the transcript owns this typed allocation on the context
        // validated above and remains borrowed until this method returns.
        Ok(unsafe {
            self.generate_proving_ctxs_from_device_inner(
                touched_memory.view(),
                in_num_records,
                None,
            )
        })
    }

    /// Consumes the initialized prefix of sorted unique RVR touched blocks
    /// directly on device. The caller retains ownership of the backing buffer
    /// until this method returns.
    ///
    /// # Safety
    ///
    /// `touched_memory` must point to `in_num_records` valid `TouchedBlock`
    /// records on `self.device_ctx`, every value must be canonical for the CUDA
    /// proof field, and the allocation must remain alive until this method returns.
    #[instrument(name = "generate_proving_ctxs_from_device", skip_all)]
    pub(crate) unsafe fn generate_proving_ctxs_from_device(
        &mut self,
        touched_memory: DeviceBufferView,
        in_num_records: usize,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        // SAFETY: forwarded from this method's caller.
        unsafe {
            self.generate_proving_ctxs_from_device_inner(touched_memory, in_num_records, None)
        }
    }

    unsafe fn generate_proving_ctxs_from_device_inner(
        &mut self,
        touched_memory: DeviceBufferView,
        in_num_records: usize,
        host_touched_memory: Option<&[TouchedBlock]>,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let expected_bytes = in_num_records
            .checked_mul(std::mem::size_of::<TouchedBlock>())
            .expect("touched-memory byte length overflow");
        assert_eq!(
            touched_memory.size, expected_bytes,
            "touched-block view must be its exact initialized prefix"
        );
        assert!(
            in_num_records == 0 || !touched_memory.ptr.is_null(),
            "nonempty touched-block view has a null pointer"
        );
        let mem = MemTracker::start("generate mem proving ctxs");
        // Exact merkle trace rows: one initial row per touched-spanning node, one final
        // row per dirty-spanning node (or just the forced root final row when nothing
        // is dirty).
        let merkle_rows = if in_num_records == 0 {
            let leftmost_values = 'left: {
                let mut res = [F::ZERO; VM_DIGEST_WIDTH];
                if self.initial_memory[ADDR_SPACE_OFFSET as usize].is_empty() {
                    break 'left res;
                }
                let layout =
                    &self.merkle_tree.mem_config().addr_spaces[ADDR_SPACE_OFFSET as usize].layout;
                let one_cell_size = layout.size();
                let mut values = vec![0u8; one_cell_size * VM_DIGEST_WIDTH];
                unsafe {
                    cuda_memcpy_on::<true, false>(
                        values.as_mut_ptr() as *mut std::ffi::c_void,
                        self.initial_memory[ADDR_SPACE_OFFSET as usize].as_ptr()
                            as *const std::ffi::c_void,
                        values.len(),
                        &self.device_ctx,
                    )
                    .unwrap();
                    for i in 0..VM_DIGEST_WIDTH {
                        res[i] = layout.to_field::<F>(&values[i * one_cell_size..]);
                    }
                }
                res
            };

            let values_u32 = leftmost_values.map(Self::field_to_raw_u32);
            let merkle_record = MemoryMerkleRecord {
                address_space: ADDR_SPACE_OFFSET,
                ptr: 0,
                is_dirty: 0,
                values: values_u32,
            };
            let merkle_records = [merkle_record];
            let merkle_words: &[u32] = unsafe {
                std::slice::from_raw_parts(
                    merkle_records.as_ptr() as *const u32,
                    MERKLE_TOUCHED_BLOCK_WIDTH,
                )
            };
            self.merkle_records = Some(merkle_words.to_device_on(&self.device_ctx).unwrap());

            // The artificial clean touch spans one root-to-leaf path of initial rows,
            // plus the always-present final root row.
            let merkle_rows = self
                .merkle_tree
                .mem_config()
                .memory_dimensions()
                .overall_height()
                + 1;
            #[cfg(feature = "metrics")]
            {
                self.unpadded_merkle_height = merkle_rows;
            }
            self.boundary
                .finalize_records::<VM_DIGEST_WIDTH>(Vec::new());
            self.prepare_poseidon2_records(0, 0, merkle_rows);
            merkle_rows
        } else {
            let _span = tracing::info_span!("mem_merge_records").entered();
            let out_words = in_num_records
                * (std::mem::size_of::<MemoryInventoryRecord<VM_DIGEST_WIDTH, BLOCKS_PER_LEAF>>()
                    / std::mem::size_of::<u32>());
            let d_tmp_records = DeviceBuffer::<u32>::with_capacity_on(out_words, &self.device_ctx);
            let d_out_records = DeviceBuffer::<u32>::with_capacity_on(out_words, &self.device_ctx);
            let d_metadata =
                DeviceBuffer::<inventory::MergeMetadata>::with_capacity_on(1, &self.device_ctx);
            if host_touched_memory.is_none() {
                d_metadata.fill_zero_on(&self.device_ctx).unwrap();
            }
            let d_flags = DeviceBuffer::<u32>::with_capacity_on(in_num_records, &self.device_ctx);
            let d_positions =
                DeviceBuffer::<u32>::with_capacity_on(in_num_records, &self.device_ctx);
            let d_initial_mem = self
                .boundary
                .initial_leaves
                .to_device_on(&self.device_ctx)
                .unwrap();
            let mut temp_bytes = 0usize;
            unsafe {
                inventory::merge_records_get_temp_bytes(
                    &d_flags,
                    in_num_records,
                    &mut temp_bytes,
                    self.device_ctx.stream.as_raw(),
                )
                .expect("merge_records_get_temp_bytes failed");
            }
            let d_temp_storage = if temp_bytes == 0 {
                DeviceBuffer::<u8>::new()
            } else {
                DeviceBuffer::<u8>::with_capacity_on(temp_bytes, &self.device_ctx)
            };
            unsafe {
                let memory_dimensions = self.merkle_tree.mem_config().memory_dimensions();
                inventory::merge_records(
                    touched_memory,
                    in_num_records,
                    memory_dimensions.address_height,
                    &d_initial_mem,
                    &d_tmp_records,
                    &d_out_records,
                    &d_flags,
                    &d_positions,
                    &d_temp_storage,
                    temp_bytes,
                    &d_metadata,
                    host_touched_memory.is_none(),
                    self.device_ctx.stream.as_raw(),
                )
                .expect("merge_records failed");
            }

            let (out_num_records, num_dirty_leaves, merkle_rows) = if let Some(partition) =
                host_touched_memory
            {
                // The compacted leaf keys and dirty bits are pure functions of the
                // sorted host partition, so compute the exact trace shape while the
                // merge kernels run.
                let memory_dimensions = self.merkle_tree.mem_config().memory_dimensions();
                let tree_height = memory_dimensions.overall_height();
                let mut touched_nodes = SpanningNodeCounter::default();
                let mut dirty_nodes = SpanningNodeCounter::default();
                let mut num_touched_leaves = 0usize;
                let mut num_dirty_leaves = 0usize;
                for leaf_blocks in partition.chunk_by(|a, b| {
                    (a.address_space, a.ptr / VM_DIGEST_WIDTH as u32)
                        == (b.address_space, b.ptr / VM_DIGEST_WIDTH as u32)
                }) {
                    let block = &leaf_blocks[0];
                    let key = (block.address_space, block.ptr / VM_DIGEST_WIDTH as u32);
                    let leaf_index = memory_dimensions.label_to_index(key);
                    touched_nodes.push(leaf_index, tree_height);
                    num_touched_leaves += 1;
                    if leaf_blocks.iter().any(|b| b.is_dirty != 0) {
                        dirty_nodes.push(leaf_index, tree_height);
                        num_dirty_leaves += 1;
                    }
                }
                let merkle_rows = touched_nodes.nodes
                    + if num_dirty_leaves == 0 {
                        1
                    } else {
                        dirty_nodes.nodes
                    };
                (num_touched_leaves, num_dirty_leaves, merkle_rows)
            } else {
                // This fixed-size record is the only host data required from the
                // RVR device-resident touched-block list. Its copy also completes
                // all reads from the borrowed input view.
                let metadata = d_metadata.to_host_on(&self.device_ctx).unwrap()[0];
                let out_num_records = metadata.out_num_records;
                assert!(
                    (1..=in_num_records).contains(&out_num_records),
                    "GPU memory merge returned invalid record count {out_num_records} for {in_num_records} inputs"
                );
                assert!(
                    metadata.dirty_leaves <= out_num_records,
                    "GPU memory merge returned {} dirty leaves for {out_num_records} records",
                    metadata.dirty_leaves
                );
                let tree_height = self
                    .merkle_tree
                    .mem_config()
                    .memory_dimensions()
                    .overall_height();
                let touched_nodes = usize::try_from(metadata.touched_path_sum)
                    .ok()
                    .and_then(|sum| tree_height.checked_add(sum))
                    .expect("memory Merkle touched-node count overflow");
                let final_nodes = if metadata.dirty_leaves == 0 {
                    1
                } else {
                    usize::try_from(metadata.dirty_path_sum)
                        .ok()
                        .and_then(|sum| tree_height.checked_add(sum))
                        .expect("memory Merkle dirty-node count overflow")
                };
                let merkle_rows = touched_nodes
                    .checked_add(final_nodes)
                    .expect("memory Merkle trace height overflow");
                (out_num_records, metadata.dirty_leaves, merkle_rows)
            };
            #[cfg(feature = "metrics")]
            {
                self.unpadded_merkle_height = merkle_rows;
            }

            // The merge has produced `d_out_records`, and the metadata copy
            // above (when present) has fenced its borrowed scratch. Release
            // every merge-only allocation before Poseidon and Merkle buffers
            // are prepared so trace generation does not retain two phases'
            // working sets at once. DeviceBuffer destruction is ordered on
            // the same stream as the merge kernel.
            drop((
                d_tmp_records,
                d_metadata,
                d_flags,
                d_positions,
                d_initial_mem,
                d_temp_storage,
            ));
            {
                let _span = tracing::info_span!("poseidon2_prepare").entered();
                self.prepare_poseidon2_records(out_num_records, num_dirty_leaves, merkle_rows);
            }

            // Send records to boundary chip
            self.boundary
                .finalize_records_device::<VM_DIGEST_WIDTH>(d_out_records, out_num_records);

            // Send records to memory merkle tree: convert boundary-layout
            // records to Merkle touched-block records on device (the merged
            // records already live there; a host round-trip would serialize
            // on the stream and rebuild the buffer one record at a time).
            let d_merkle_records = DeviceBuffer::<u32>::with_capacity_on(
                out_num_records * MERKLE_TOUCHED_BLOCK_WIDTH,
                &self.device_ctx,
            );
            unsafe {
                inventory::to_merkle_records(
                    self.boundary.records(),
                    out_num_records,
                    &d_merkle_records,
                    self.device_ctx.stream.as_raw(),
                )
                .expect("inventory_to_merkle_records failed");
            }
            self.merkle_records = Some(d_merkle_records);
            merkle_rows
        };

        mem.tracing_info("merkle update");
        let merkle_proof_ctx = {
            let _span = tracing::info_span!("merkle_update").entered();
            self.merkle_tree.finalize();
            let merkle_records = self.merkle_records.take().expect("missing merkle records");
            let ctx = self.merkle_tree.update_with_touched_blocks(
                merkle_rows,
                &merkle_records,
                in_num_records == 0,
            );
            // `update_with_touched_blocks` synchronizes its final device-to-host
            // root copy before returning, so no kernel can still borrow these
            // records when they are released.
            drop(merkle_records);
            ctx
        };
        mem.tracing_info("boundary tracegen");
        let ret = {
            let _span = tracing::info_span!("boundary_trace_gen").entered();
            vec![self.boundary.generate_proving_ctx(), merkle_proof_ctx]
        };
        mem.tracing_info("dropping merkle tree");
        {
            let _span = tracing::info_span!("merkle_drop").entered();
            self.clear_initial_memory();
        }
        mem.emit_metrics();
        ret
    }

    /// Sizes the shared Poseidon2 record buffer to the exact push count: the boundary
    /// kernel records one initial hash per touched leaf plus one final hash per dirty
    /// leaf, and every merkle trace row records exactly one compression. These counts
    /// must stay in lockstep with `boundary.cu` / `merkle_tree.cu`.
    fn prepare_poseidon2_records(
        &self,
        boundary_records: usize,
        dirty_leaves: usize,
        merkle_rows: usize,
    ) {
        let num_records = boundary_records
            .checked_add(dirty_leaves)
            .and_then(|n| n.checked_add(merkle_rows))
            .expect("Poseidon2 records count overflow");
        self.hasher_chip.prepare_records(num_records);
    }
}

impl Drop for MemoryInventoryGPU {
    fn drop(&mut self) {
        self.clear_initial_memory();
    }
}

#[cfg(test)]
mod tests {
    use std::{
        array,
        collections::{BTreeMap, BTreeSet},
        sync::Arc,
    };

    use openvm_circuit::{
        arch::{
            vm_poseidon2_config, AddressSpaceHostConfig, MemoryCellType, MemoryConfig,
            ADDR_SPACE_OFFSET, MEMORY_BLOCK_BYTES,
        },
        system::{
            memory::{
                merkle::{memory_to_vec_partition, MemoryMerkleChip, MerkleTree},
                offline_checker::pack_u8_block_value,
                online::{GuestMemory, PAGE_SIZE},
                persistent::DirtyLeaves,
                ptr_bits_from_address_height, AddressMap,
            },
            poseidon2::Poseidon2PeripheryChip,
            TouchedBlock,
        },
    };
    use openvm_cuda_backend::{
        data_transporter::assert_eq_host_and_device_matrix_col_maj, prelude::F,
    };
    use openvm_cuda_common::{
        common::get_device,
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use openvm_instructions::{
        exe::SparseMemoryImage,
        riscv::{MEMORY_AS, REGISTER_AS},
        DEFERRAL_AS,
    };
    use openvm_stark_backend::{
        interaction::PermutationCheckBus,
        prover::{ColMajorMatrix, MatrixDimensions},
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

    use super::*;

    /// CPU reference Merkle root, for cross-checking the GPU root.
    fn cpu_merkle_root(memory: &AddressMap, mem_config: &MemoryConfig) -> [F; VM_DIGEST_WIDTH] {
        let cpu_hasher = Poseidon2PeripheryChip::new(vm_poseidon2_config(), 3);
        let cpu_merkle_tree = MerkleTree::<F, VM_DIGEST_WIDTH>::from_memory(
            memory,
            &mem_config.memory_dimensions(),
            &cpu_hasher,
        );
        cpu_merkle_tree.root()
    }

    fn pack_u8_block_canonical(bytes: [u8; MEMORY_BLOCK_BYTES]) -> [u32; BLOCK_FE_WIDTH] {
        pack_u8_block_value(&bytes.map(F::from_u8)).map(|value| value.as_canonical_u32())
    }

    /// Builds a GPU inventory, loads `initial_memory`, returns the contexts for `touched_memory`.
    fn run_inventory(
        mem_config: &MemoryConfig,
        initial_memory: &AddressMap,
        touched_memory: TouchedMemory,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let hasher_chip = Arc::new(Poseidon2PeripheryChipGPU::new(1, device_ctx.clone()));
        let mut inventory =
            MemoryInventoryGPU::new(mem_config.clone(), hasher_chip, device_ctx.clone());
        inventory.set_initial_memory(initial_memory);
        let contexts = inventory.generate_proving_ctxs(touched_memory);
        device_ctx.stream.synchronize().unwrap();
        contexts
    }

    fn run_inventory_direct(
        mem_config: &MemoryConfig,
        initial_memory: &AddressMap,
        touched_memory: TouchedMemory,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let hasher_chip = Arc::new(Poseidon2PeripheryChipGPU::new(1, device_ctx.clone()));
        let mut inventory =
            MemoryInventoryGPU::new(mem_config.clone(), hasher_chip, device_ctx.clone());
        inventory.set_initial_memory(initial_memory);
        let d_touched = if touched_memory.is_empty() {
            DeviceBuffer::new()
        } else {
            touched_memory.as_slice().to_device_on(&device_ctx).unwrap()
        };
        // SAFETY: d_touched owns this same-context exact initialized prefix
        // until the method has synchronized and returned.
        let contexts = unsafe {
            inventory.generate_proving_ctxs_from_device(d_touched.view(), touched_memory.len())
        };
        device_ctx.stream.synchronize().unwrap();
        contexts
    }

    fn assert_same_contexts(
        expected: &[AirProvingContext<GpuBackend>],
        actual: &[AirProvingContext<GpuBackend>],
    ) {
        assert_eq!(expected.len(), actual.len());
        for (expected, actual) in expected.iter().zip(actual) {
            assert_eq!(expected.common_main.height(), actual.common_main.height());
            assert_eq!(expected.common_main.width(), actual.common_main.width());
            assert_eq!(expected.public_values, actual.public_values);
            let device_ctx = GpuDeviceCtx {
                device_id: get_device().unwrap() as u32,
                stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
            };
            assert_eq!(
                expected
                    .common_main
                    .buffer()
                    .to_host_on(&device_ctx)
                    .unwrap(),
                actual.common_main.buffer().to_host_on(&device_ctx).unwrap()
            );
        }
    }

    /// Extracts the Merkle root: the merkle chip is the one emitting at least two public-value
    /// digests, and the root is the last one.
    fn gpu_merkle_root(ctxs: &[AirProvingContext<GpuBackend>]) -> [F; VM_DIGEST_WIDTH] {
        let merkle_ctx = ctxs
            .iter()
            .find(|ctx| ctx.public_values.len() >= 2 * VM_DIGEST_WIDTH)
            .expect("missing merkle ctx");
        let gpu_root_slice =
            &merkle_ctx.public_values[merkle_ctx.public_values.len() - VM_DIGEST_WIDTH..];
        gpu_root_slice.try_into().unwrap()
    }

    /// Single-block register + memory config shared by the empty- and touched-memory tests.
    fn single_block_setup() -> (MemoryConfig, GuestMemory) {
        let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
        for addr_space in [REGISTER_AS, MEMORY_AS] {
            // num_cells is in u16 cells; allocate 2 * VM_DIGEST_WIDTH = 16 cells.
            addr_spaces[addr_space as usize].num_cells = 2 * VM_DIGEST_WIDTH;
        }
        let mem_config = MemoryConfig::new(2, addr_spaces, ptr_bits_from_address_height(1), 29, 17);

        let mut memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
        unsafe {
            memory.write_bytes::<MEMORY_BLOCK_BYTES>(REGISTER_AS, 0, [1, 2, 3, 4, 5, 6, 7, 8]);
            memory.write_bytes::<MEMORY_BLOCK_BYTES>(MEMORY_AS, 0, [9, 10, 11, 12, 0, 0, 0, 0]);
        }
        // `write_bytes` doesn't mark pages; mark them so `set_initial_memory` transfers them
        // (see `AddressMap::touched_pages`).
        for addr_space in [REGISTER_AS, MEMORY_AS] {
            memory.memory.touched_pages[addr_space as usize].mark_byte_range(0, MEMORY_BLOCK_BYTES);
        }
        (mem_config, memory)
    }

    #[test]
    fn test_empty_touched_memory_uses_full_chunk_values() {
        let (mem_config, memory) = single_block_setup();

        let expected_root = cpu_merkle_root(&memory.memory, &mem_config);

        let ctxs = run_inventory(&mem_config, &memory.memory, Vec::new());
        let direct_ctxs = run_inventory_direct(&mem_config, &memory.memory, Vec::new());
        assert_same_contexts(&ctxs, &direct_ctxs);
        let boundary_ctx = ctxs.first().expect("missing boundary ctx");
        assert_eq!(
            boundary_ctx.common_main.height(),
            1,
            "boundary trace should be a single padding row for empty touched memory"
        );
        assert!(
            boundary_ctx.public_values.is_empty(),
            "boundary chip should not emit public values"
        );

        assert_eq!(expected_root, gpu_merkle_root(&ctxs));
    }

    // Touched-memory merge path: each address space has one clean and one dirty
    // 8-byte block in the same leaf, routed through the `<4, 1> -> <8, 2>` merge.
    #[test]
    fn test_touched_memory_device_path_matches_legacy_across_address_spaces() {
        let (mem_config, memory) = single_block_setup();

        let mut final_memory = memory.clone();
        let clean_register_bytes = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let clean_memory_bytes = [9u8, 10, 11, 12, 0, 0, 0, 0];
        let touched_register_bytes = [81u8, 82, 83, 84, 85, 86, 87, 88];
        let touched_bytes_late = [111u8, 112, 113, 114, 115, 116, 117, 118];
        unsafe {
            final_memory.write_bytes::<MEMORY_BLOCK_BYTES>(
                REGISTER_AS,
                MEMORY_BLOCK_BYTES as u32,
                touched_register_bytes,
            );
            final_memory.write_bytes::<MEMORY_BLOCK_BYTES>(
                MEMORY_AS,
                MEMORY_BLOCK_BYTES as u32,
                touched_bytes_late,
            );
        }

        let expected_root = cpu_merkle_root(&final_memory.memory, &mem_config);

        let touched_memory = vec![
            TouchedBlock {
                address_space: REGISTER_AS,
                ptr: 0,
                is_dirty: 0,
                timestamp: 1,
                values: pack_u8_block_canonical(clean_register_bytes),
            },
            TouchedBlock {
                address_space: REGISTER_AS,
                ptr: BLOCK_FE_WIDTH as u32,
                is_dirty: 1,
                timestamp: 2,
                values: pack_u8_block_canonical(touched_register_bytes),
            },
            TouchedBlock {
                address_space: MEMORY_AS,
                ptr: 0,
                is_dirty: 0,
                timestamp: 3,
                values: pack_u8_block_canonical(clean_memory_bytes),
            },
            TouchedBlock {
                address_space: MEMORY_AS,
                ptr: BLOCK_FE_WIDTH as u32,
                is_dirty: 1,
                timestamp: 4,
                values: pack_u8_block_canonical(touched_bytes_late),
            },
        ];
        let ctxs = run_inventory(&mem_config, &memory.memory, touched_memory.clone());
        let direct_ctxs = run_inventory_direct(&mem_config, &memory.memory, touched_memory);
        assert_same_contexts(&ctxs, &direct_ctxs);
        let boundary_ctx = ctxs.first().expect("missing boundary ctx");
        assert!(
            boundary_ctx.common_main.height() > 0,
            "boundary trace should be present when touched memory is non-empty"
        );
        assert!(
            boundary_ctx.public_values.is_empty(),
            "boundary chip should not emit public values"
        );

        assert_eq!(expected_root, gpu_merkle_root(&ctxs));
    }

    #[test]
    fn test_canonical_field_touched_memory_matches_cpu_merkle_root() {
        let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
        addr_spaces[DEFERRAL_AS as usize] =
            AddressSpaceHostConfig::new(2 * VM_DIGEST_WIDTH, MemoryCellType::field32());
        let mem_config = MemoryConfig::new(2, addr_spaces, ptr_bits_from_address_height(1), 29, 17);
        let mut memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
        let mut first_leaf = array::from_fn(|i| 100 + i as u32);
        first_leaf[BLOCK_FE_WIDTH] = F::ORDER_U32 - 2;
        let second_leaf = array::from_fn(|i| 1_000 + i as u32);
        unsafe {
            memory.write::<F, VM_DIGEST_WIDTH>(DEFERRAL_AS, 0, first_leaf.map(F::from_u32));
            memory.write::<F, VM_DIGEST_WIDTH>(
                DEFERRAL_AS,
                VM_DIGEST_WIDTH as u32,
                second_leaf.map(F::from_u32),
            );
        }
        memory.memory.recompute_touched_pages();
        let mut final_memory = memory.clone();
        let dirty_values = [0, 1, 123_456, F::ORDER_U32 - 1];
        let clean_values: [u32; BLOCK_FE_WIDTH] = second_leaf[..BLOCK_FE_WIDTH].try_into().unwrap();
        unsafe {
            final_memory.write::<F, BLOCK_FE_WIDTH>(DEFERRAL_AS, 0, dirty_values.map(F::from_u32));
        }

        let expected_root = cpu_merkle_root(&final_memory.memory, &mem_config);
        let touched_memory = vec![
            TouchedBlock {
                address_space: DEFERRAL_AS,
                ptr: 0,
                is_dirty: 1,
                timestamp: 1,
                values: dirty_values,
            },
            TouchedBlock {
                address_space: DEFERRAL_AS,
                ptr: VM_DIGEST_WIDTH as u32,
                is_dirty: 0,
                timestamp: 2,
                values: clean_values,
            },
        ];
        let ctxs = run_inventory(&mem_config, &memory.memory, touched_memory.clone());
        let direct_ctxs = run_inventory_direct(&mem_config, &memory.memory, touched_memory);

        assert_same_contexts(&ctxs, &direct_ctxs);
        assert_eq!(expected_root, gpu_merkle_root(&ctxs));
    }

    // Paged transfer: only pages 0 and 2 of a 4-page AS are populated (via `set_from_sparse`), so
    // the H2D copies just those. Asserts GPU root == CPU root and that paging engaged.
    #[test]
    fn test_set_initial_memory_copies_only_touched_pages() {
        const NUM_PAGES: usize = 4;
        // U16 memory cells (2 bytes), so one PAGE_SIZE-byte page is PAGE_SIZE / 2 cells.
        let num_cells = NUM_PAGES * (PAGE_SIZE / 2);
        // 2^address_height leaf labels per AS must cover num_cells / VM_DIGEST_WIDTH leaves.
        let address_height = (num_cells / VM_DIGEST_WIDTH).ilog2() as usize;

        let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
        addr_spaces[REGISTER_AS as usize].num_cells = 2 * VM_DIGEST_WIDTH;
        addr_spaces[MEMORY_AS as usize].num_cells = num_cells;
        let mem_config = MemoryConfig::new(
            2,
            addr_spaces,
            ptr_bits_from_address_height(address_height),
            29,
            17,
        );

        // Sparse initial image: an 8-byte block at the start of page 0 and another at page 2.
        let mut sparse = SparseMemoryImage::new();
        for (i, b) in [9u8, 10, 11, 12, 13, 14, 15, 16].into_iter().enumerate() {
            sparse.insert((MEMORY_AS, i as u32), b);
        }
        for (i, b) in [101u8, 102, 103, 104, 105, 106, 107, 108]
            .into_iter()
            .enumerate()
        {
            sparse.insert((MEMORY_AS, (2 * PAGE_SIZE + i) as u32), b);
        }
        let mut addr_map = AddressMap::from_mem_config(&mem_config);
        addr_map.set_from_sparse(&sparse);
        let memory = GuestMemory::new(addr_map);

        // Paging engaged: only pages 0 and 2 are marked, coalesced into two single-page runs.
        let mem_bytes = memory.memory.get_memory()[MEMORY_AS as usize]
            .as_slice()
            .len();
        let runs = memory.memory.touched_pages[MEMORY_AS as usize].touched_byte_ranges(mem_bytes);
        assert_eq!(
            runs,
            vec![(0, PAGE_SIZE), (2 * PAGE_SIZE, 3 * PAGE_SIZE)],
            "only the two written pages should be transferred"
        );
        let copied: usize = runs.iter().map(|(s, e)| e - s).sum();
        assert!(
            copied < mem_bytes,
            "paging should copy fewer bytes ({copied}) than the full AS ({mem_bytes})"
        );

        let expected_root = cpu_merkle_root(&memory.memory, &mem_config);

        let ctxs = run_inventory(&mem_config, &memory.memory, Vec::new());

        assert_eq!(expected_root, gpu_merkle_root(&ctxs));
    }

    /// CPU and GPU must generate the *same* Merkle trace, row for row.
    #[test]
    fn test_merkle_trace_matches_between_cpu_and_gpu() {
        let (mem_config, memory) = single_block_setup();
        let mut final_memory = memory.clone();
        let touched_bytes = [101u8, 102, 103, 104, 105, 106, 107, 108];
        let touched_bytes_late = [111u8, 112, 113, 114, 115, 116, 117, 118];
        unsafe {
            final_memory.write_bytes::<MEMORY_BLOCK_BYTES>(MEMORY_AS, 0, touched_bytes);
            final_memory.write_bytes::<MEMORY_BLOCK_BYTES>(
                MEMORY_AS,
                MEMORY_BLOCK_BYTES as u32,
                touched_bytes_late,
            );
        }

        // GPU trace: both blocks written (is_dirty = 1), forming one dirty leaf.
        let touched_memory = vec![
            TouchedBlock {
                address_space: MEMORY_AS,
                ptr: 0,
                is_dirty: 1,
                timestamp: 1,
                values: pack_u8_block_canonical(touched_bytes),
            },
            TouchedBlock {
                address_space: MEMORY_AS,
                ptr: BLOCK_FE_WIDTH as u32,
                is_dirty: 1,
                timestamp: 3,
                values: pack_u8_block_canonical(touched_bytes_late),
            },
        ];
        let ctxs = run_inventory(&mem_config, &memory.memory, touched_memory);
        let gpu_merkle = ctxs
            .iter()
            .find(|ctx| ctx.public_values.len() >= 2 * VM_DIGEST_WIDTH)
            .expect("missing merkle ctx");

        // CPU reference trace via the merkle chip, which applies the reverse+swap that
        // defines the committed row order.
        let md = mem_config.memory_dimensions();
        let hasher = Poseidon2PeripheryChip::new(vm_poseidon2_config(), 3);

        // The two written blocks fall in one 8-cell leaf at (MEMORY_AS, label 0).
        let touched_labels: BTreeSet<(u32, u32)> = BTreeSet::from([(MEMORY_AS, 0)]);
        let final_partition: BTreeMap<(u32, u32), [F; VM_DIGEST_WIDTH]> =
            memory_to_vec_partition::<F, VM_DIGEST_WIDTH>(&final_memory.memory, &md)
                .into_iter()
                .map(|(idx, values)| {
                    let address_space = (idx >> md.address_height) as u32 + ADDR_SPACE_OFFSET;
                    let label = (idx & ((1 << md.address_height) - 1)) as u32;
                    ((address_space, label * VM_DIGEST_WIDTH as u32), values)
                })
                .filter(|((address_space, ptr), _)| {
                    touched_labels.contains(&(*address_space, ptr / VM_DIGEST_WIDTH as u32))
                })
                .collect();
        // Every touched block sets is_dirty = 1, so the leaf is dirty (matches the GPU input).
        let dirty_leaves: DirtyLeaves = final_partition.keys().copied().collect();

        let bus = PermutationCheckBus::new(0); // bus index does not affect the main trace
        let mut cpu_chip = MemoryMerkleChip::<VM_DIGEST_WIDTH, F>::new(md, bus, bus);
        cpu_chip.finalize(&memory.memory, &final_partition, &dirty_leaves, &hasher);
        let cpu_trace = cpu_chip
            .generate_proving_ctx::<BabyBearPoseidon2Config>()
            .common_main;
        let cpu_trace_cm = ColMajorMatrix::from_row_major(&cpu_trace);

        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        assert_eq_host_and_device_matrix_col_maj(
            &cpu_trace_cm,
            &gpu_merkle.common_main,
            &device_ctx,
        );
    }

    #[test]
    fn test_set_initial_memory_replaces_unfinished_upload() {
        let (mem_config, first_memory) = single_block_setup();
        let mut second_memory = first_memory.clone();
        unsafe {
            second_memory.write_bytes::<MEMORY_BLOCK_BYTES>(
                MEMORY_AS,
                0,
                [31, 32, 33, 34, 35, 36, 37, 38],
            );
        }
        second_memory.memory.touched_pages[MEMORY_AS as usize]
            .mark_byte_range(0, MEMORY_BLOCK_BYTES);
        let expected_root = cpu_merkle_root(&second_memory.memory, &mem_config);

        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let hasher_chip = Arc::new(Poseidon2PeripheryChipGPU::new(1, device_ctx.clone()));
        let mut inventory =
            MemoryInventoryGPU::new(mem_config.clone(), hasher_chip, device_ctx.clone());

        inventory.set_initial_memory(&first_memory.memory);
        inventory.set_initial_memory(&second_memory.memory);
        assert_eq!(
            inventory.initial_memory.len(),
            second_memory.memory.get_memory().len()
        );

        let contexts = inventory.generate_proving_ctxs(Vec::new());
        device_ctx.stream.synchronize().unwrap();
        assert_eq!(expected_root, gpu_merkle_root(&contexts));
    }

    #[cfg(any(debug_assertions, feature = "stark-debug"))]
    #[test]
    #[should_panic(expected = "is outside touched_pages; sparse H2D would zero it on device")]
    fn test_set_initial_memory_rejects_nonzero_unmarked_page() {
        let (mem_config, _) = single_block_setup();
        let mut memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
        unsafe {
            memory.write_bytes::<MEMORY_BLOCK_BYTES>(MEMORY_AS, 0, [1, 2, 3, 4, 5, 6, 7, 8]);
        }
        assert!(memory.memory.touched_pages[MEMORY_AS as usize]
            .touched_byte_ranges(MEMORY_BLOCK_BYTES)
            .is_empty());

        run_inventory(&mem_config, &memory.memory, Vec::new());
    }
}
