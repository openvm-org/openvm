use std::sync::Arc;

use openvm_circuit::{
    arch::{AddressSpaceHostLayout, MemoryConfig, ADDR_SPACE_OFFSET, BLOCK_FE_WIDTH},
    system::{
        memory::{persistent::BLOCKS_PER_LEAF, AddressMap},
        TouchedMemory,
    },
};
use openvm_circuit_primitives::Chip;
use openvm_cuda_backend::{prelude::F, GpuBackend};
use openvm_cuda_common::{
    copy::{cuda_memcpy_on, MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    memory_manager::MemTracker,
    stream::GpuDeviceCtx,
};
use openvm_stark_backend::{p3_field::PrimeCharacteristicRing, prover::AirProvingContext};
use tracing::instrument;

use super::{
    boundary::BoundaryChipGPU,
    merkle_tree::{MemoryMerkleTree, MERKLE_TOUCHED_BLOCK_WIDTH},
    Poseidon2PeripheryChipGPU, DIGEST_WIDTH,
};
use crate::{cuda_abi::inventory, system::memory::online::LinearMemory};

// The CUDA merge kernel in `inventory.cu` is hardcoded to a 2-way merge of
// `<IN_BLOCK_SIZE=4, 1>` records into `<OUT_BLOCK_SIZE=8, 2>` records, so the only
// supported `(BLOCK_FE_WIDTH, DIGEST_WIDTH)` shape is `(4, 8)`.
const _: () = assert!(
    BLOCK_FE_WIDTH == 4 && DIGEST_WIDTH == 8,
    "CUDA memory inventory only supports (BLOCK_FE_WIDTH, DIGEST_WIDTH) == (4, 8)"
);

pub struct MemoryInventoryGPU {
    pub device_ctx: GpuDeviceCtx,
    pub boundary: BoundaryChipGPU,
    pub merkle_tree: MemoryMerkleTree,
    pub initial_memory: Vec<DeviceBuffer<u8>>,
    pub merkle_records: Option<DeviceBuffer<u32>>,
    #[cfg(feature = "metrics")]
    pub(super) unpadded_merkle_height: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryInventoryRecord<const DIGEST_WIDTH: usize, const BLOCKS: usize> {
    address_space: u32,
    ptr: u32,
    timestamps: [u32; BLOCKS],
    values: [u32; DIGEST_WIDTH],
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
            initial_memory: Vec::new(),
            merkle_records: None,
            #[cfg(feature = "metrics")]
            unpadded_merkle_height: 0,
        }
    }

    #[instrument(name = "set_initial_memory", skip_all)]
    pub fn set_initial_memory(&mut self, initial_memory: &AddressMap) {
        let mem = MemTracker::start("set initial memory");
        for (addr_sp, raw_mem) in initial_memory
            .get_memory()
            .iter()
            .map(|mem| mem.as_slice())
            .enumerate()
        {
            tracing::debug!(
                "Setting initial memory for address space {}: {} bytes",
                addr_sp,
                raw_mem.len()
            );
            self.initial_memory.push(if raw_mem.is_empty() {
                DeviceBuffer::new()
            } else {
                raw_mem
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
        if partition.is_empty() {
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
            self.merkle_records = Some(merkle_words.to_device_on(&self.device_ctx).unwrap());

            self.boundary.finalize_records::<DIGEST_WIDTH>(Vec::new());
        } else if BLOCK_FE_WIDTH == 4 && DIGEST_WIDTH == 8 {
            // Merge BLOCK_FE_WIDTH-sized input blocks into DIGEST_WIDTH-sized chunks via the
            // hardcoded `<4,1> → <8,2>` kernel in `inventory.cu`.
            let in_records: Vec<MemoryInventoryRecord<BLOCK_FE_WIDTH, 1>> = partition
                .iter()
                .map(|&((addr_space, ptr), ts_values)| MemoryInventoryRecord {
                    address_space: addr_space,
                    ptr,
                    timestamps: [ts_values.timestamp],
                    values: ts_values.values.map(Self::field_to_raw_u32),
                })
                .collect();
            let in_num_records = in_records.len();
            let out_words = in_num_records
                * (std::mem::size_of::<MemoryInventoryRecord<DIGEST_WIDTH, BLOCKS_PER_LEAF>>()
                    / std::mem::size_of::<u32>());
            let d_in_records = in_records
                .to_device_on(&self.device_ctx)
                .unwrap()
                .as_buffer::<u32>();
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

            // Send records to boundary chip
            let out_num_records = d_out_num_records.to_host_on(&self.device_ctx).unwrap()[0];
            self.boundary
                .finalize_records_device::<DIGEST_WIDTH>(d_out_records, out_num_records);

            // Send records to memory merkle tree
            let out_records = self
                .boundary
                .records()
                .to_host_on(&self.device_ctx)
                .unwrap();
            let record_words = 2 + BLOCKS_PER_LEAF + DIGEST_WIDTH;
            let mut merkle_records = Vec::with_capacity(out_num_records);
            for i in 0..out_num_records {
                let base = i * record_words;
                let mut values = [0u32; DIGEST_WIDTH];
                values.copy_from_slice(
                    &out_records
                        [base + 2 + BLOCKS_PER_LEAF..base + 2 + BLOCKS_PER_LEAF + DIGEST_WIDTH],
                );
                let timestamp = *out_records[base + 2..base + 2 + BLOCKS_PER_LEAF]
                    .iter()
                    .max()
                    .unwrap();
                let record = MemoryMerkleRecord {
                    address_space: out_records[base],
                    ptr: out_records[base + 1],
                    timestamp,
                    values,
                };
                merkle_records.push(record);
            }
            let merkle_words: &[u32] = unsafe {
                std::slice::from_raw_parts(
                    merkle_records.as_ptr() as *const u32,
                    merkle_records.len() * MERKLE_TOUCHED_BLOCK_WIDTH,
                )
            };
            self.merkle_records = Some(merkle_words.to_device_on(&self.device_ctx).unwrap());
        } else {
            // Excluded by the module-level const assert on (BLOCK_FE_WIDTH, DIGEST_WIDTH).
            unreachable!()
        }

        let unpadded_merkle_height = self.merkle_tree.calculate_unpadded_height(&partition);
        #[cfg(feature = "metrics")]
        {
            self.unpadded_merkle_height = unpadded_merkle_height;
        }

        mem.tracing_info("merkle update");
        self.merkle_tree.finalize();
        let merkle_proof_ctx = self.merkle_tree.update_with_touched_blocks(
            unpadded_merkle_height,
            self.merkle_records
                .as_ref()
                .expect("missing merkle records"),
            partition.is_empty(),
        );
        mem.tracing_info("boundary tracegen");
        let ret = vec![self.boundary.generate_proving_ctx(()), merkle_proof_ctx];
        mem.tracing_info("dropping merkle tree");
        self.merkle_tree.drop_subtrees();
        self.initial_memory = Vec::new();
        mem.emit_metrics();
        ret
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
        arch::{vm_poseidon2_config, MemoryConfig, MEMORY_BLOCK_BYTES, U16_CELL_SIZE},
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
    use openvm_instructions::riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS};
    use openvm_stark_backend::prover::MatrixDimensions;

    use super::*;
    #[test]
    fn test_empty_touched_memory_uses_full_chunk_values() {
        let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
        for addr_space in [RV64_REGISTER_AS, RV64_MEMORY_AS] {
            addr_spaces[addr_space as usize].num_cells = 2 * DIGEST_WIDTH;
        }
        // pointer_max_bits = 5: address_height = 5 - log2(BUS_PTR_SCALE) -
        // log2(DIGEST_WIDTH) = 5 - 1 - 3 = 1, giving max_size = 2 ≥
        // num_cells / DIGEST_WIDTH = 2.
        let mem_config = MemoryConfig::new(2, addr_spaces, 5, 29, 17);

        let mut memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
        unsafe {
            memory.write::<u8, MEMORY_BLOCK_BYTES>(
                RV64_REGISTER_AS,
                0,
                [1, 2, 3, 4, 5, 6, 7, 8],
            );
            memory.write::<u8, MEMORY_BLOCK_BYTES>(
                RV64_MEMORY_AS,
                0,
                [9, 10, 11, 12, 0, 0, 0, 0],
            );
        }

        let cpu_hasher = Poseidon2PeripheryChip::new(vm_poseidon2_config(), 3);
        let cpu_merkle_tree = MerkleTree::<F, DIGEST_WIDTH>::from_memory(
            &memory.memory,
            &mem_config.memory_dimensions(),
            &cpu_hasher,
        );
        let expected_root = cpu_merkle_tree.root();

        let max_buffer_size = (mem_config
            .addr_spaces
            .iter()
            .map(|ashc| ashc.num_cells * 2 + mem_config.memory_dimensions().overall_height())
            .sum::<usize>()
            * 2)
        .next_power_of_two()
            * 2
            * DIGEST_WIDTH;
        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let hasher_chip = Arc::new(Poseidon2PeripheryChipGPU::new(
            max_buffer_size,
            1,
            device_ctx.clone(),
        ));
        let mut inventory =
            MemoryInventoryGPU::new(mem_config.clone(), hasher_chip, device_ctx.clone());
        inventory.set_initial_memory(&memory.memory);

        let ctxs = inventory.generate_proving_ctxs(Vec::new());
        let boundary_ctx = ctxs.first().expect("missing boundary ctx");
        assert_eq!(
            boundary_ctx.common_main.height(),
            0,
            "boundary trace should be empty for empty touched memory"
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

    // TODO: pre-rv64 this test put two `BLOCK_FE_WIDTH == 4` touched blocks at ptrs 0 and 4,
    // which both fell in Merkle chunk 0 and exercised the 2-way merge path in `inventory.cu`. On
    // rv64 `BLOCK_FE_WIDTH == DIGEST_WIDTH == 8`, so two blocks cannot share a chunk and the test
    // Touched-memory coverage for the merge path: writes two MEMORY_BLOCK_BYTES
    // blocks into RV64_MEMORY_AS (u16-celled, so each block is
    // BLOCK_FE_WIDTH = 4 u16 cells = MEMORY_BLOCK_BYTES = 8 bytes) and routes
    // them through `inventory.cu`'s `<4, 1> → <8, 2>` merge kernel.
    #[test]
    fn test_touched_memory_updates_memory_address_space() {
        let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
        for addr_space in [RV64_REGISTER_AS, RV64_MEMORY_AS] {
            // num_cells is in u16 cells; allocate 2 * DIGEST_WIDTH = 16 cells.
            addr_spaces[addr_space as usize].num_cells = 2 * DIGEST_WIDTH;
        }
        // pointer_max_bits = 5: address_height = 1, max_size = 2 ≥ num_cells / DIGEST_WIDTH.
        let mem_config = MemoryConfig::new(2, addr_spaces, 5, 29, 17);

        let mut memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
        unsafe {
            memory.write::<u8, MEMORY_BLOCK_BYTES>(
                RV64_REGISTER_AS,
                0,
                [1, 2, 3, 4, 5, 6, 7, 8],
            );
            memory.write::<u8, MEMORY_BLOCK_BYTES>(
                RV64_MEMORY_AS,
                0,
                [9, 10, 11, 12, 0, 0, 0, 0],
            );
        }

        let mut final_memory = memory.clone();
        let touched_bytes = [101u8, 102, 103, 104, 105, 106, 107, 108];
        let touched_bytes_late = [111u8, 112, 113, 114, 115, 116, 117, 118];
        unsafe {
            final_memory.write::<u8, MEMORY_BLOCK_BYTES>(RV64_MEMORY_AS, 0, touched_bytes);
            final_memory.write::<u8, MEMORY_BLOCK_BYTES>(
                RV64_MEMORY_AS,
                MEMORY_BLOCK_BYTES as u32,
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

        let max_buffer_size = (mem_config
            .addr_spaces
            .iter()
            .map(|ashc| ashc.num_cells * 2 + mem_config.memory_dimensions().overall_height())
            .sum::<usize>()
            * 2)
        .next_power_of_two()
            * 2
            * DIGEST_WIDTH;
        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let hasher_chip = Arc::new(Poseidon2PeripheryChipGPU::new(
            max_buffer_size,
            1,
            device_ctx.clone(),
        ));
        let mut inventory =
            MemoryInventoryGPU::new(mem_config.clone(), hasher_chip, device_ctx.clone());
        inventory.set_initial_memory(&memory.memory);

        // Pack 8 bytes (one MEMORY_BLOCK_BYTES block) into 4 u16-celled F
        // values via little-endian; matches the storage layout the boundary
        // chip sees post Stage 1.6 flip.
        let pack_block = |bytes: [u8; MEMORY_BLOCK_BYTES]| -> [F; BLOCK_FE_WIDTH] {
            let mut out = [F::ZERO; BLOCK_FE_WIDTH];
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = F::from_u16(u16::from_le_bytes([
                    bytes[i * U16_CELL_SIZE],
                    bytes[i * U16_CELL_SIZE + 1],
                ]));
            }
            out
        };
        let touched_memory = vec![
            (
                (RV64_MEMORY_AS, 0),
                TimestampedValues {
                    timestamp: 1,
                    values: pack_block(touched_bytes),
                },
            ),
            (
                (RV64_MEMORY_AS, BLOCK_FE_WIDTH as u32),
                TimestampedValues {
                    timestamp: 3,
                    values: pack_block(touched_bytes_late),
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
