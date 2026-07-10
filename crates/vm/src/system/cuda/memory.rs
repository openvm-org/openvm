use std::sync::Arc;

use openvm_circuit::{
    arch::{AddressSpaceHostLayout, MemoryConfig, ADDR_SPACE_OFFSET},
    system::{memory::AddressMap, TouchedMemory},
};
use openvm_circuit_primitives::Chip;
use openvm_cuda_backend::{prelude::F, GpuBackend};
use openvm_cuda_common::{
    copy::{cuda_memcpy_on, MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    memory_manager::MemTracker,
    stream::GpuDeviceCtx,
};
use openvm_stark_backend::{
    p3_field::PrimeCharacteristicRing,
    p3_maybe_rayon::prelude::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
        ParallelSlice, ParallelSliceMut,
    },
    prover::AirProvingContext,
};

/// Chunk size for the parallel pack into the upload staging buffer.
const UPLOAD_PACK_CHUNK: usize = 8 << 20;
use tracing::instrument;

use super::{
    boundary::BoundaryChipGPU,
    merkle_tree::{MemoryMerkleTree, MERKLE_TOUCHED_BLOCK_WIDTH},
    Poseidon2PeripheryChipGPU, DIGEST_WIDTH,
};
use crate::{cuda_abi::inventory, system::memory::online::LinearMemory};

pub struct MemoryInventoryGPU {
    pub device_ctx: GpuDeviceCtx,
    pub boundary: BoundaryChipGPU,
    pub merkle_tree: MemoryMerkleTree,
    pub hasher_chip: Arc<Poseidon2PeripheryChipGPU>,
    pub initial_memory: Vec<DeviceBuffer<u8>>,
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
    values: [u32; DIGEST_WIDTH],
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
        let slices: Vec<&[u8]> = initial_memory
            .get_memory()
            .iter()
            .map(|mem| mem.as_slice())
            .collect();
        let total: usize = slices.iter().map(|s| s.len()).sum();
        let staging = self.upload_staging.ensure(total);
        let mut offset = 0usize;
        for (addr_sp, raw_mem) in slices.into_iter().enumerate() {
            tracing::debug!(
                "Setting initial memory for address space {}: {} bytes",
                addr_sp,
                raw_mem.len()
            );
            self.initial_memory.push(if raw_mem.is_empty() {
                DeviceBuffer::new()
            } else {
                let dst = &mut staging[offset..offset + raw_mem.len()];
                offset += raw_mem.len();
                // Parallel pack: fully consumes the guest memory before the
                // async DMA from the (registered) staging is enqueued.
                dst.par_chunks_mut(UPLOAD_PACK_CHUNK)
                    .zip(raw_mem.par_chunks(UPLOAD_PACK_CHUNK))
                    .for_each(|(d, s)| d.copy_from_slice(s));
                (*dst)
                    .to_device_on(&self.device_ctx)
                    .expect("failed to copy memory to device")
            });
            self.merkle_tree
                .build_async(&self.initial_memory[addr_sp], addr_sp);
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
        let merkle_proof_ctx = if partition.is_empty() {
            let leftmost_values = 'left: {
                let mut res = [F::ZERO; DIGEST_WIDTH];
                if self.initial_memory[ADDR_SPACE_OFFSET as usize].is_empty() {
                    break 'left res;
                }
                let layout =
                    &self.merkle_tree.mem_config().addr_spaces[ADDR_SPACE_OFFSET as usize].layout;
                let one_cell_size = layout.size();
                let mut values = vec![0u8; one_cell_size * DIGEST_WIDTH];
                unsafe {
                    cuda_memcpy_on::<true, false>(
                        values.as_mut_ptr() as *mut std::ffi::c_void,
                        self.initial_memory[ADDR_SPACE_OFFSET as usize].as_ptr()
                            as *const std::ffi::c_void,
                        values.len(),
                        &self.device_ctx,
                    )
                    .unwrap();
                    for i in 0..DIGEST_WIDTH {
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
            let d_merkle_touched_memory = merkle_words.to_device_on(&self.device_ctx).unwrap();

            let unpadded_merkle_height = self.merkle_tree.calculate_unpadded_height(&partition);
            #[cfg(feature = "metrics")]
            {
                self.unpadded_merkle_height = unpadded_merkle_height;
            }

            self.boundary.finalize_records::<DIGEST_WIDTH>(Vec::new());
            self.prepare_poseidon2_records(0, unpadded_merkle_height);
            mem.tracing_info("merkle update");
            self.merkle_tree.finalize();
            self.merkle_tree.update_with_touched_blocks(
                unpadded_merkle_height,
                &d_merkle_touched_memory,
                true,
            )
        } else {
            let _span = tracing::info_span!("mem_merge_records").entered();
            // Convert to MemoryInventoryRecord<4, 1> layout, packing straight
            // into a pooled page-locked buffer so the upload takes the DMA
            // fast path; giving the buffer back routes through the arena
            // cleaner, which synchronizes the device before reuse, so the
            // in-flight copy is safe.
            const IN_REC_WORDS: usize =
                std::mem::size_of::<MemoryInventoryRecord<4, 1>>() / std::mem::size_of::<u32>();
            let in_num_records = partition.len();
            let mut h_in = crate::arch::cuda::pinned::take(in_num_records * IN_REC_WORDS * 4 + 4);
            let align = h_in.as_ptr().align_offset(std::mem::size_of::<u32>());
            let dirty_len = align + in_num_records * IN_REC_WORDS * 4;
            // SAFETY: the slice is within the buffer and 4-aligned by `align`.
            let in_words: &mut [u32] = unsafe {
                std::slice::from_raw_parts_mut(
                    h_in.as_mut_ptr().add(align) as *mut u32,
                    in_num_records * IN_REC_WORDS,
                )
            };
            in_words
                .par_chunks_mut(IN_REC_WORDS)
                .zip(partition.par_iter())
                .for_each(|(w, &((addr_space, ptr), ts_values))| {
                    w[0] = addr_space;
                    w[1] = ptr;
                    w[2] = ts_values.timestamp;
                    for (dst, v) in w[3..].iter_mut().zip(ts_values.values) {
                        *dst = Self::field_to_raw_u32(v);
                    }
                });
            let out_words = in_num_records
                * (std::mem::size_of::<MemoryInventoryRecord<8, 2>>() / std::mem::size_of::<u32>());
            let d_in_records = (*in_words).to_device_on(&self.device_ctx).unwrap();
            crate::arch::cuda::pinned::give_back(h_in, dirty_len);
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
            // ptr / OUT_BLOCK) differs from its predecessor's, and the
            // partition is sorted), so it can be computed here and the
            // mid-merge D2H sync dropped entirely.
            const OUT_BLOCK_CELLS: u32 = 8;
            let out_num_records = 1
                + (1..in_num_records)
                    .into_par_iter()
                    .filter(|&i| {
                        let (a, p) = partition[i].0;
                        let (pa, pp) = partition[i - 1].0;
                        (a, p / OUT_BLOCK_CELLS) != (pa, pp / OUT_BLOCK_CELLS)
                    })
                    .count();

            // Host work overlapping the merge kernels: neither the unpadded
            // height scan nor the Poseidon2 records buffer depends on the
            // merge kernels.
            let unpadded_merkle_height = self.merkle_tree.calculate_unpadded_height(&partition);
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
                .finalize_records_device::<DIGEST_WIDTH>(d_out_records, out_num_records);

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
            drop(_span);

            mem.tracing_info("merkle update");
            let _span = tracing::info_span!("merkle_update").entered();
            self.merkle_tree.finalize();
            self.merkle_tree.update_with_touched_blocks(
                unpadded_merkle_height,
                self.merkle_records
                    .as_ref()
                    .expect("missing merkle records"),
                false,
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
        arch::{vm_poseidon2_config, MemoryConfig},
        system::{
            memory::{merkle::MerkleTree, online::GuestMemory, AddressMap, TimestampedValues},
            poseidon2::Poseidon2PeripheryChip,
        },
    };
    use openvm_cuda_backend::prelude::F;
    use openvm_cuda_common::{
        common::get_device,
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use openvm_instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS};
    use openvm_stark_backend::prover::MatrixDimensions;

    use super::*;
    #[test]
    fn test_empty_touched_memory_uses_full_chunk_values() {
        let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
        for addr_space in [RV32_REGISTER_AS, RV32_MEMORY_AS] {
            addr_spaces[addr_space as usize].num_cells = 2 * DIGEST_WIDTH;
        }
        let mem_config = MemoryConfig::new(2, addr_spaces, 4, 29, 17);

        let mut memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
        unsafe {
            memory.write::<u8, DIGEST_WIDTH>(RV32_REGISTER_AS, 0, [1, 2, 3, 4, 5, 6, 7, 8]);
            memory.write::<u8, { DIGEST_WIDTH / 2 }>(RV32_MEMORY_AS, 0, [9, 10, 11, 12]);
        }

        let cpu_hasher = Poseidon2PeripheryChip::new(vm_poseidon2_config(), 3);
        let cpu_merkle_tree = MerkleTree::<F, DIGEST_WIDTH>::from_memory(
            &memory.memory,
            &mem_config.memory_dimensions(),
            &cpu_hasher,
        );
        let expected_root = cpu_merkle_tree.root();

        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let hasher_chip = Arc::new(Poseidon2PeripheryChipGPU::new(1, device_ctx.clone()));
        let mut inventory =
            MemoryInventoryGPU::new(mem_config.clone(), hasher_chip, device_ctx.clone());
        inventory.set_initial_memory(&memory.memory);

        let ctxs = inventory.generate_proving_ctxs(Vec::new());
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

        let merkle_ctx = ctxs
            .iter()
            .find(|ctx| ctx.public_values.len() >= 2 * DIGEST_WIDTH)
            .expect("missing merkle ctx");
        let gpu_root_slice =
            &merkle_ctx.public_values[merkle_ctx.public_values.len() - DIGEST_WIDTH..];
        let gpu_root: [F; DIGEST_WIDTH] = gpu_root_slice.try_into().unwrap();

        assert_eq!(expected_root, gpu_root);
    }

    #[test]
    fn test_touched_memory_updates_memory_address_space() {
        let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
        for addr_space in [RV32_REGISTER_AS, RV32_MEMORY_AS] {
            addr_spaces[addr_space as usize].num_cells = 2 * DIGEST_WIDTH;
        }
        let mem_config = MemoryConfig::new(2, addr_spaces, 4, 29, 17);

        let mut memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
        unsafe {
            memory.write::<u8, DIGEST_WIDTH>(RV32_REGISTER_AS, 0, [1, 2, 3, 4, 5, 6, 7, 8]);
            memory.write::<u8, { DIGEST_WIDTH / 2 }>(RV32_MEMORY_AS, 0, [9, 10, 11, 12]);
        }

        let mut final_memory = memory.clone();
        let touched_bytes = [101u8, 102, 103, 104];
        let touched_bytes_late = [111u8, 112, 113, 114];
        unsafe {
            final_memory.write::<u8, { crate::arch::DEFAULT_BLOCK_SIZE }>(
                RV32_MEMORY_AS,
                0,
                touched_bytes,
            );
            final_memory.write::<u8, { crate::arch::DEFAULT_BLOCK_SIZE }>(
                RV32_MEMORY_AS,
                crate::arch::DEFAULT_BLOCK_SIZE as u32,
                touched_bytes_late,
            );
        }

        let cpu_hasher = Poseidon2PeripheryChip::new(vm_poseidon2_config(), 3);
        let cpu_merkle_tree = MerkleTree::<F, DIGEST_WIDTH>::from_memory(
            &final_memory.memory,
            &mem_config.memory_dimensions(),
            &cpu_hasher,
        );
        let expected_root = cpu_merkle_tree.root();

        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let hasher_chip = Arc::new(Poseidon2PeripheryChipGPU::new(1, device_ctx.clone()));
        let mut inventory =
            MemoryInventoryGPU::new(mem_config.clone(), hasher_chip, device_ctx.clone());
        inventory.set_initial_memory(&memory.memory);

        let touched_memory = vec![
            (
                (RV32_MEMORY_AS, 0),
                TimestampedValues {
                    timestamp: 1,
                    values: touched_bytes.map(F::from_u8),
                },
            ),
            (
                (RV32_MEMORY_AS, crate::arch::DEFAULT_BLOCK_SIZE as u32),
                TimestampedValues {
                    timestamp: 3,
                    values: touched_bytes_late.map(F::from_u8),
                },
            ),
        ];
        let ctxs = inventory.generate_proving_ctxs(touched_memory);
        let boundary_ctx = ctxs.first().expect("missing boundary ctx");
        assert!(
            boundary_ctx.common_main.height() > 0,
            "boundary trace should be present when touched memory is non-empty"
        );
        assert!(
            boundary_ctx.public_values.is_empty(),
            "boundary chip should not emit public values"
        );

        let merkle_ctx = ctxs
            .iter()
            .find(|ctx| ctx.public_values.len() >= 2 * DIGEST_WIDTH)
            .expect("missing merkle ctx");
        let gpu_root_slice =
            &merkle_ctx.public_values[merkle_ctx.public_values.len() - DIGEST_WIDTH..];
        let gpu_root: [F; DIGEST_WIDTH] = gpu_root_slice.try_into().unwrap();

        assert_eq!(expected_root, gpu_root);
    }
}
