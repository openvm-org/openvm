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
    d_buffer::DeviceBuffer,
    memory_manager::MemTracker,
    pinned,
    stream::GpuDeviceCtx,
};
use openvm_instructions::VM_DIGEST_WIDTH;
use openvm_stark_backend::{
    p3_field::PrimeCharacteristicRing,
    p3_maybe_rayon::prelude::{
        IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSlice,
        ParallelSliceMut,
    },
    prover::AirProvingContext,
};
use tracing::instrument;

/// Chunk size for packing touched memory runs into pinned upload staging.
const UPLOAD_PACK_CHUNK: usize = 8 << 20;

use super::{
    boundary::BoundaryChipGPU,
    merkle_tree::{MemoryMerkleTree, MERKLE_TOUCHED_BLOCK_WIDTH},
    Poseidon2PeripheryChipGPU,
};
use crate::{cuda_abi::inventory, system::memory::online::LinearMemory};

// The CUDA merge kernel in `inventory.cu` is hardcoded to a 2-way merge of
// `<IN_BLOCK_SIZE=4, 1>` records into `<OUT_BLOCK_SIZE=8, 2>` records, so the only
// supported `(BLOCK_FE_WIDTH, VM_DIGEST_WIDTH)` shape is `(4, 8)`.
const _: () = assert!(
    BLOCK_FE_WIDTH == 4 && VM_DIGEST_WIDTH == 8,
    "CUDA memory inventory only supports (BLOCK_FE_WIDTH, VM_DIGEST_WIDTH) == (4, 8)"
);

// `TouchedBlock<F>` must be exactly the 7-word `InRec` layout in `inventory.cu`
// so the merge path can upload the vector's bytes without repacking.
const _: () = assert!(
    std::mem::size_of::<TouchedBlock<F>>() == (3 + BLOCK_FE_WIDTH) * std::mem::size_of::<u32>(),
    "TouchedBlock<F> must match the 7-u32-word InRec layout in inventory.cu"
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

/// Owned page-locked staging for sparse per-segment memory-image uploads.
///
/// The full device address spaces stay zero-filled, while only touched host
/// runs are packed contiguously here and uploaded to their device offsets.
/// This preserves the sparse-transfer correctness fix and avoids registering
/// memory owned by the executor.
#[derive(Default)]
struct PinnedStaging {
    buf: Vec<u8>,
    registered: bool,
}

impl PinnedStaging {
    fn ensure(&mut self, len: usize) -> &mut [u8] {
        if self.buf.len() < len {
            if self.registered {
                crate::arch::cuda::pinned::unregister_region(self.buf.as_mut_ptr());
                self.registered = false;
            }
            self.buf = vec![0u8; len];
            self.registered =
                crate::arch::cuda::pinned::register_region(self.buf.as_mut_ptr(), len);
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
            crate::arch::cuda::pinned::unregister_region(self.buf.as_mut_ptr());
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryInventoryRecord<const CHUNK: usize, const BLOCKS: usize> {
    address_space: u32,
    ptr: u32,
    timestamps: [u32; BLOCKS],
    values: [u32; CHUNK],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryMerkleRecord {
    address_space: u32,
    ptr: u32,
    timestamp: u32,
    values: [u32; VM_DIGEST_WIDTH],
}

impl MemoryInventoryGPU {
    #[inline]
    fn field_to_raw_u32(value: F) -> u32 {
        unsafe { std::mem::transmute::<F, u32>(value) }
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
            // The CPU backend clones the full host image, while this sparse transfer zero-fills
            // every unmarked page on device. A nonzero byte outside the touched runs therefore
            // means some writer mutated this address space without marking the page (see
            // `AddressMap::extend_touched_pages_from_touched`), and the two backends would
            // silently diverge — surfacing later as an unattributable memory-bus LogUp
            // imbalance. The scan is O(address-space bytes) on host, so it is enabled only in
            // debug builds and `stark-debug` (the feature used by GPU debug gate runs).
            #[cfg(any(debug_assertions, feature = "stark-debug"))]
            {
                let mut cursor = 0usize;
                let sentinel = (raw_mem.len(), raw_mem.len());
                for &(start, end) in runs.iter().chain(std::iter::once(&sentinel)) {
                    if let Some(pos) = raw_mem[cursor..start].iter().position(|&b| b != 0) {
                        let offset = cursor + pos;
                        panic!(
                            "address space {addr_sp}: nonzero byte at offset {offset} (page {}) \
                             is outside touched_pages; the sparse H2D transfer would zero it on \
                             device while the CPU backend keeps it",
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
                    dst.par_chunks_mut(UPLOAD_PACK_CHUNK)
                        .zip(raw_mem[start..end].par_chunks(UPLOAD_PACK_CHUNK))
                        .for_each(|(d, s)| d.copy_from_slice(s));
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

    #[instrument(name = "generate_proving_ctxs", skip_all)]
    pub fn generate_proving_ctxs(
        &mut self,
        touched_memory: TouchedMemory<F>,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let mem = MemTracker::start("generate mem proving ctxs");
        let partition = touched_memory;
        let unpadded_merkle_height = if partition.is_empty() {
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
                timestamp: 0,
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

            let unpadded_merkle_height = self
                .merkle_tree
                .calculate_unpadded_height(&partition, |b| (b.address_space, b.ptr));
            #[cfg(feature = "metrics")]
            {
                self.unpadded_merkle_height = unpadded_merkle_height;
            }
            self.boundary
                .finalize_records::<VM_DIGEST_WIDTH>(Vec::new());
            self.prepare_poseidon2_records(0, unpadded_merkle_height);
            unpadded_merkle_height
        } else {
            let _span = tracing::info_span!("mem_merge_records").entered();
            let in_num_records = partition.len();
            let in_bytes = in_num_records * std::mem::size_of::<TouchedBlock<F>>();
            let mut h_in = pinned::take(in_bytes + 4);
            let align_offset = h_in.as_ptr().align_offset(std::mem::size_of::<u32>());
            let dirty_len = align_offset + in_bytes;
            let src: &[u8] =
                unsafe { std::slice::from_raw_parts(partition.as_ptr() as *const u8, in_bytes) };
            let dst = &mut h_in[align_offset..align_offset + in_bytes];
            dst.par_chunks_mut(UPLOAD_PACK_CHUNK)
                .zip(src.par_chunks(UPLOAD_PACK_CHUNK))
                .for_each(|(d, s)| d.copy_from_slice(s));
            // SAFETY: 4-aligned by `align_offset`, within the buffer.
            let in_words: &[u32] = unsafe {
                std::slice::from_raw_parts(
                    h_in.as_ptr().add(align_offset) as *const u32,
                    in_bytes / std::mem::size_of::<u32>(),
                )
            };
            let out_words = in_num_records
                * (std::mem::size_of::<MemoryInventoryRecord<VM_DIGEST_WIDTH, BLOCKS_PER_LEAF>>()
                    / std::mem::size_of::<u32>());
            let d_in_records = in_words.to_device_on(&self.device_ctx).unwrap();
            pinned::give_back(h_in, dirty_len);
            let d_tmp_records = DeviceBuffer::<u32>::with_capacity_on(out_words, &self.device_ctx);
            let d_out_records = DeviceBuffer::<u32>::with_capacity_on(out_words, &self.device_ctx);
            let d_out_num_records = DeviceBuffer::<usize>::with_capacity_on(1, &self.device_ctx);
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
                inventory::merge_records(
                    &d_in_records,
                    in_num_records,
                    &d_initial_mem,
                    &d_tmp_records,
                    &d_out_records,
                    &d_flags,
                    &d_positions,
                    &d_temp_storage,
                    temp_bytes,
                    &d_out_num_records,
                    self.device_ctx.stream.as_raw(),
                )
                .expect("merge_records failed");
            }

            // The merged record count is a pure function of input adjacency
            // (the device merge flags a record iff its (address_space,
            // ptr / VM_DIGEST_WIDTH) differs from its predecessor's, and the
            // partition is sorted), so it can be computed here and the
            // mid-merge D2H sync dropped entirely.
            let out_num_records = 1
                + (1..in_num_records)
                    .into_par_iter()
                    .filter(|&i| {
                        let (a, b) = (&partition[i], &partition[i - 1]);
                        (a.address_space, a.ptr / VM_DIGEST_WIDTH as u32)
                            != (b.address_space, b.ptr / VM_DIGEST_WIDTH as u32)
                    })
                    .count();

            // Host work overlapping the merge kernels: neither the unpadded
            // height scan nor the Poseidon2 records buffer depends on the
            // merge kernels.
            let unpadded_merkle_height = self
                .merkle_tree
                .calculate_unpadded_height(&partition, |b| (b.address_space, b.ptr));
            #[cfg(feature = "metrics")]
            {
                self.unpadded_merkle_height = unpadded_merkle_height;
            }
            {
                let _span = tracing::info_span!("poseidon2_prepare").entered();
                self.prepare_poseidon2_records(out_num_records, unpadded_merkle_height);
            }

            // Cross-check the host-computed count against the device merge in
            // debug builds (a mismatch would corrupt the boundary trace).
            #[cfg(debug_assertions)]
            {
                let device_count = d_out_num_records.to_host_on(&self.device_ctx).unwrap()[0];
                assert_eq!(device_count, out_num_records, "merged-count mismatch");
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
            unpadded_merkle_height
        };

        mem.tracing_info("merkle update");
        let merkle_proof_ctx = {
            let _span = tracing::info_span!("merkle_update").entered();
            self.merkle_tree.finalize();
            self.merkle_tree.update_with_touched_blocks(
                unpadded_merkle_height,
                self.merkle_records
                    .as_ref()
                    .expect("missing merkle records"),
                partition.is_empty(),
            )
        };
        mem.tracing_info("boundary tracegen");
        let ret = {
            let _span = tracing::info_span!("boundary_trace_gen").entered();
            vec![self.boundary.generate_proving_ctx(()), merkle_proof_ctx]
        };
        mem.tracing_info("dropping merkle tree");
        {
            let _span = tracing::info_span!("merkle_drop").entered();
            self.merkle_tree.drop_subtrees();
        }
        self.initial_memory = Vec::new();
        mem.emit_metrics();
        ret
    }

    fn prepare_poseidon2_records(&self, boundary_records: usize, merkle_height: usize) {
        let num_records = boundary_records
            .checked_mul(2)
            .and_then(|n| n.checked_add(merkle_height))
            .expect("Poseidon2 records count overflow");
        self.hasher_chip.prepare_records(num_records);
    }
}

impl Drop for MemoryInventoryGPU {
    fn drop(&mut self) {
        // WARNING: The merkle subtree events must be completed before dropping the initial memory
        // buffers. This prevents buffers from dropping before build_async completes.
        self.merkle_tree.drop_subtrees();
        self.initial_memory.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openvm_circuit::{
        arch::{vm_poseidon2_config, MemoryConfig, MEMORY_BLOCK_BYTES},
        system::{
            memory::{
                merkle::MerkleTree,
                offline_checker::pack_u8_block_value,
                online::{GuestMemory, PAGE_SIZE},
                ptr_bits_from_address_height, AddressMap,
            },
            poseidon2::Poseidon2PeripheryChip,
            TouchedBlock,
        },
    };
    use openvm_cuda_backend::prelude::F;
    use openvm_cuda_common::{
        common::get_device,
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use openvm_instructions::{
        exe::SparseMemoryImage,
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    };
    use openvm_stark_backend::prover::MatrixDimensions;

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

    /// Builds a GPU inventory, loads `initial_memory`, returns the contexts for `touched_memory`.
    fn run_inventory(
        mem_config: &MemoryConfig,
        initial_memory: &AddressMap,
        touched_memory: TouchedMemory<F>,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let hasher_chip = Arc::new(Poseidon2PeripheryChipGPU::new(1, device_ctx.clone()));
        let mut inventory =
            MemoryInventoryGPU::new(mem_config.clone(), hasher_chip, device_ctx.clone());
        inventory.set_initial_memory(initial_memory);
        inventory.generate_proving_ctxs(touched_memory)
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
        for addr_space in [RV64_REGISTER_AS, RV64_MEMORY_AS] {
            // num_cells is in u16 cells; allocate 2 * VM_DIGEST_WIDTH = 16 cells.
            addr_spaces[addr_space as usize].num_cells = 2 * VM_DIGEST_WIDTH;
        }
        let mem_config = MemoryConfig::new(2, addr_spaces, ptr_bits_from_address_height(1), 29, 17);

        let mut memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
        unsafe {
            memory.write_bytes::<MEMORY_BLOCK_BYTES>(RV64_REGISTER_AS, 0, [1, 2, 3, 4, 5, 6, 7, 8]);
            memory.write_bytes::<MEMORY_BLOCK_BYTES>(
                RV64_MEMORY_AS,
                0,
                [9, 10, 11, 12, 0, 0, 0, 0],
            );
        }
        // `write_bytes` doesn't mark pages; mark them so `set_initial_memory` transfers them
        // (see `AddressMap::touched_pages`).
        for addr_space in [RV64_REGISTER_AS, RV64_MEMORY_AS] {
            memory.memory.touched_pages[addr_space as usize].mark_byte_range(0, MEMORY_BLOCK_BYTES);
        }
        (mem_config, memory)
    }

    #[test]
    fn test_empty_touched_memory_uses_full_chunk_values() {
        let (mem_config, memory) = single_block_setup();

        let expected_root = cpu_merkle_root(&memory.memory, &mem_config);

        let ctxs = run_inventory(&mem_config, &memory.memory, Vec::new());
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

    // Touched-memory merge path: writes two 8-byte (BLOCK_FE_WIDTH = 4 u16 cells) blocks into
    // RV64_MEMORY_AS and routes them through `inventory.cu`'s `<4, 1> -> <8, 2>` merge kernel.
    #[test]
    fn test_touched_memory_updates_memory_address_space() {
        let (mem_config, memory) = single_block_setup();

        let mut final_memory = memory.clone();
        let touched_bytes = [101u8, 102, 103, 104, 105, 106, 107, 108];
        let touched_bytes_late = [111u8, 112, 113, 114, 115, 116, 117, 118];
        unsafe {
            final_memory.write_bytes::<MEMORY_BLOCK_BYTES>(RV64_MEMORY_AS, 0, touched_bytes);
            final_memory.write_bytes::<MEMORY_BLOCK_BYTES>(
                RV64_MEMORY_AS,
                MEMORY_BLOCK_BYTES as u32,
                touched_bytes_late,
            );
        }

        let expected_root = cpu_merkle_root(&final_memory.memory, &mem_config);

        let touched_memory = vec![
            TouchedBlock {
                address_space: RV64_MEMORY_AS,
                ptr: 0,
                timestamp: 1,
                values: pack_u8_block_value(&touched_bytes.map(F::from_u8)),
            },
            TouchedBlock {
                address_space: RV64_MEMORY_AS,
                ptr: BLOCK_FE_WIDTH as u32,
                timestamp: 3,
                values: pack_u8_block_value(&touched_bytes_late.map(F::from_u8)),
            },
        ];
        let ctxs = run_inventory(&mem_config, &memory.memory, touched_memory);
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
        addr_spaces[RV64_REGISTER_AS as usize].num_cells = 2 * VM_DIGEST_WIDTH;
        addr_spaces[RV64_MEMORY_AS as usize].num_cells = num_cells;
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
            sparse.insert((RV64_MEMORY_AS, i as u32), b);
        }
        for (i, b) in [101u8, 102, 103, 104, 105, 106, 107, 108]
            .into_iter()
            .enumerate()
        {
            sparse.insert((RV64_MEMORY_AS, (2 * PAGE_SIZE + i) as u32), b);
        }
        let mut addr_map = AddressMap::from_mem_config(&mem_config);
        addr_map.set_from_sparse(&sparse);
        let memory = GuestMemory::new(addr_map);

        // Paging engaged: only pages 0 and 2 are marked, coalesced into two single-page runs.
        let mem_bytes = memory.memory.get_memory()[RV64_MEMORY_AS as usize]
            .as_slice()
            .len();
        let runs =
            memory.memory.touched_pages[RV64_MEMORY_AS as usize].touched_byte_ranges(mem_bytes);
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
}
