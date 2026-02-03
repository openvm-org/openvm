use std::sync::Arc;

use openvm_circuit::{
    arch::{
        AddressSpaceHostLayout, DenseRecordArena, MemoryConfig, ADDR_SPACE_OFFSET,
        CONST_BLOCK_SIZE,
    },
    system::{memory::AddressMap, TouchedMemory},
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{prover_backend::GpuBackend, types::F};
use openvm_cuda_common::{
    copy::{cuda_memcpy, MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    memory_manager::MemTracker,
};
use p3_field::FieldAlgebra;
use openvm_stark_backend::{p3_util::log2_ceil_usize, prover::types::AirProvingContext, Chip};

use super::{
    access_adapters::AccessAdapterInventoryGPU,
    boundary::{BoundaryChipGPU, BoundaryFields},
    merkle_tree::{MemoryMerkleTree, MERKLE_TOUCHED_BLOCK_WIDTH},
    Poseidon2PeripheryChipGPU, DIGEST_WIDTH,
};

use crate::cuda_abi::inventory;
use crate::system::memory::online::LinearMemory;

pub struct MemoryInventoryGPU {
    pub boundary: BoundaryChipGPU,
    pub access_adapters: AccessAdapterInventoryGPU,
    pub persistent: Option<PersistentMemoryInventoryGPU>,
    #[cfg(feature = "metrics")]
    pub(super) unpadded_merkle_height: usize,
}

pub struct PersistentMemoryInventoryGPU {
    pub merkle_tree: MemoryMerkleTree,
    pub initial_memory: Vec<DeviceBuffer<u8>>,
    pub merkle_records: Option<DeviceBuffer<u32>>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryInventoryRecord<const CHUNK: usize, const BLOCKS: usize> {
    address_space: u32,
    ptr: u32,
    timestamps: [u32; BLOCKS],
    values: [u32; CHUNK],
}

impl MemoryInventoryGPU {
    #[inline]
    fn field_to_raw_u32(value: F) -> u32 {
        unsafe { std::mem::transmute::<F, u32>(value) }
    }

    pub fn volatile(config: MemoryConfig, range_checker: Arc<VariableRangeCheckerChipGPU>) -> Self {
        let addr_space_max_bits = log2_ceil_usize(
            (ADDR_SPACE_OFFSET + 2u32.pow(config.addr_space_height as u32)) as usize,
        );
        Self {
            boundary: BoundaryChipGPU::volatile(
                range_checker.clone(),
                addr_space_max_bits,
                config.pointer_max_bits,
            ),
            access_adapters: AccessAdapterInventoryGPU::new(
                range_checker,
                config.max_access_adapter_n,
                config.timestamp_max_bits,
            ),
            persistent: None,
            #[cfg(feature = "metrics")]
            unpadded_merkle_height: 0,
        }
    }

    pub fn persistent(
        config: MemoryConfig,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        hasher_chip: Arc<Poseidon2PeripheryChipGPU>,
    ) -> Self {
        Self {
            boundary: BoundaryChipGPU::persistent(hasher_chip.shared_buffer()),
            access_adapters: AccessAdapterInventoryGPU::new(
                range_checker,
                config.max_access_adapter_n,
                config.timestamp_max_bits,
            ),
            persistent: Some(PersistentMemoryInventoryGPU {
                merkle_tree: MemoryMerkleTree::new(config.clone(), hasher_chip.clone()),
                initial_memory: Vec::new(),
                merkle_records: None,
            }),
            #[cfg(feature = "metrics")]
            unpadded_merkle_height: 0,
        }
    }

    pub fn continuation_enabled(&self) -> bool {
        self.persistent.is_some()
    }

    pub fn set_initial_memory(&mut self, initial_memory: &AddressMap) {
        let _mem = MemTracker::start("set initial memory");
        let persistent = self
            .persistent
            .as_mut()
            .expect("`set_initial_memory` requires persistent memory");
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
            persistent.initial_memory.push(if raw_mem.is_empty() {
                DeviceBuffer::new()
            } else {
                raw_mem
                    .to_device()
                    .expect("failed to copy memory to device")
            });
            persistent
                .merkle_tree
                .build_async(&persistent.initial_memory[addr_sp], addr_sp);
        }
        match &mut self.boundary.fields {
            BoundaryFields::Volatile(_) => {
                panic!("`set_initial_memory` requires persistent memory")
            }
            BoundaryFields::Persistent(fields) => {
                fields.initial_leaves = persistent
                    .initial_memory
                    .iter()
                    .skip(1)
                    .map(|per_as| per_as.as_raw_ptr())
                    .collect();
            }
        }
    }

    pub fn generate_proving_ctxs(
        &mut self,
        access_adapter_arena: DenseRecordArena,
        touched_memory: TouchedMemory<F>,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let mem = MemTracker::start("generate mem proving ctxs");
        let merkle_proof_ctx = match touched_memory {
            TouchedMemory::Persistent(partition) => {
                let persistent = self
                    .persistent
                    .as_mut()
                    .expect("persistent touched memory requires persistent memory interface");
                if partition.is_empty() {
                    let leftmost_values = 'left: {
                        let mut res = [F::ZERO; CONST_BLOCK_SIZE];
                        if persistent.initial_memory[ADDR_SPACE_OFFSET as usize].is_empty() {
                            break 'left res;
                        }
                        let layout = &persistent.merkle_tree.mem_config().addr_spaces
                            [ADDR_SPACE_OFFSET as usize]
                            .layout;
                        let one_cell_size = layout.size();
                        let values = vec![0u8; one_cell_size * CONST_BLOCK_SIZE];
                        unsafe {
                            cuda_memcpy::<true, false>(
                                values.as_ptr() as *mut std::ffi::c_void,
                                persistent.initial_memory[ADDR_SPACE_OFFSET as usize].as_ptr()
                                    as *const std::ffi::c_void,
                                values.len(),
                            )
                            .unwrap();
                            for i in 0..CONST_BLOCK_SIZE {
                                res[i] = layout.to_field::<F>(&values[i * one_cell_size..]);
                            }
                        }
                        res
                    };

                    let mut values_u32 = [0u32; DIGEST_WIDTH];
                    for i in 0..CONST_BLOCK_SIZE {
                        values_u32[i] = Self::field_to_raw_u32(leftmost_values[i]);
                    }
                    let mut merkle_records =
                        Vec::<u32>::with_capacity(MERKLE_TOUCHED_BLOCK_WIDTH);
                    merkle_records.push(ADDR_SPACE_OFFSET);
                    merkle_records.push(0);
                    merkle_records.push(0);
                    merkle_records.extend_from_slice(&values_u32);
                    let d_merkle_touched_memory = merkle_records.to_device().unwrap();

                    let unpadded_merkle_height =
                        persistent.merkle_tree.calculate_unpadded_height(&partition);
                    #[cfg(feature = "metrics")]
                    {
                        self.unpadded_merkle_height = unpadded_merkle_height;
                    }

                    self.boundary
                        .finalize_records_persistent::<DIGEST_WIDTH>(Vec::new());
                    mem.tracing_info("merkle update");
                    persistent.merkle_tree.finalize();
                    let merkle_tree_ctx = persistent.merkle_tree.update_with_touched_blocks(
                        unpadded_merkle_height,
                        &d_merkle_touched_memory,
                        true,
                    );
                    Some(merkle_tree_ctx)
                } else {
                    let in_records: Vec<MemoryInventoryRecord<4, 1>> = partition
                        .iter()
                        .map(|&((addr_space, ptr), ts_values)| MemoryInventoryRecord {
                            address_space: addr_space,
                            ptr,
                            timestamps: [ts_values.timestamp],
                            values: ts_values
                                .values
                                .map(|x| Self::field_to_raw_u32(x)),
                        })
                        .collect();
                    let in_num_records = in_records.len();
                    if in_num_records == 0 {
                        self.boundary
                            .finalize_records_persistent::<DIGEST_WIDTH>(Vec::new());
                        mem.tracing_info("merkle update");
                        persistent.merkle_tree.finalize();
                        None
                    } else {
                        let out_words = in_num_records
                            * (std::mem::size_of::<MemoryInventoryRecord<8, 2>>()
                                / std::mem::size_of::<u32>());
                        let d_in_records = in_records.to_device().unwrap().as_buffer::<u32>();
                        let d_tmp_records = DeviceBuffer::<u32>::with_capacity(out_words);
                        let d_out_records = DeviceBuffer::<u32>::with_capacity(out_words);
                        let d_out_num_records = DeviceBuffer::<usize>::with_capacity(1);
                        let d_flags = DeviceBuffer::<u32>::with_capacity(in_num_records);
                        let d_positions = DeviceBuffer::<u32>::with_capacity(in_num_records);
                        let d_initial_mem = match &self.boundary.fields {
                            BoundaryFields::Persistent(fields) => {
                                fields.initial_leaves.to_device().unwrap()
                            }
                            BoundaryFields::Volatile(_) => {
                                panic!("`merge_records` requires persistent memory")
                            }
                        };
                        let addr_space_offsets: Vec<u32> = {
                            let mut offsets = Vec::new();
                            let mut acc = 0u32;
                            for addr_sp in persistent
                                .merkle_tree
                                .mem_config()
                                .addr_spaces
                                .iter()
                                .skip(ADDR_SPACE_OFFSET as usize)
                            {
                                offsets.push(acc);
                                acc = acc.saturating_add(addr_sp.layout.size() as u32);
                            }
                            offsets.push(acc);
                            offsets
                        };
                        let d_addr_space_offsets = addr_space_offsets.to_device().unwrap();
                        let mut temp_bytes = 0usize;
                        unsafe {
                            inventory::merge_records_get_temp_bytes(
                                &d_flags,
                                in_num_records,
                                &mut temp_bytes,
                            )
                            .expect("merge_records_get_temp_bytes failed");
                        }
                        let d_temp_storage = if temp_bytes == 0 {
                            DeviceBuffer::<u8>::new()
                        } else {
                            DeviceBuffer::<u8>::with_capacity(temp_bytes)
                        };
                        unsafe {
                            inventory::merge_records(
                                &d_in_records,
                                in_num_records,
                                &d_initial_mem,
                                &d_addr_space_offsets,
                                &d_tmp_records,
                                &d_out_records,
                                &d_flags,
                                &d_positions,
                                &d_temp_storage,
                                temp_bytes,
                                &d_out_num_records,
                            )
                            .expect("merge_records failed");
                        }
                        let out_num_records = d_out_num_records.to_host().unwrap()[0] as usize;
                        self.boundary
                            .finalize_records_persistent_device::<DIGEST_WIDTH>(
                                d_out_records,
                                out_num_records,
                            );
                        let out_records = self.boundary.persistent_records().to_host().unwrap();
                        let record_words = 4 + DIGEST_WIDTH;
                        let mut merkle_records = Vec::with_capacity(
                            out_num_records * MERKLE_TOUCHED_BLOCK_WIDTH,
                        );
                        for i in 0..out_num_records {
                            let base = i * record_words;
                            let address_space = out_records[base];
                            let ptr = out_records[base + 1];
                            let ts0 = out_records[base + 2];
                            let ts1 = out_records[base + 3];
                            let timestamp = ts0.max(ts1);
                            merkle_records.push(address_space);
                            merkle_records.push(ptr);
                            merkle_records.push(timestamp);
                            merkle_records.extend_from_slice(
                                &out_records[base + 4..base + 4 + DIGEST_WIDTH],
                            );
                        }
                        persistent.merkle_records = Some(merkle_records.to_device().unwrap());

                        let unpadded_merkle_height =
                            persistent.merkle_tree.calculate_unpadded_height(&partition);
                        #[cfg(feature = "metrics")]
                        {
                            self.unpadded_merkle_height = unpadded_merkle_height;
                        }

                        mem.tracing_info("boundary finalize");
                        mem.tracing_info("merkle update");
                        persistent.merkle_tree.finalize();
                        let merkle_tree_ctx = persistent.merkle_tree.update_with_touched_blocks(
                            unpadded_merkle_height,
                            persistent
                                .merkle_records
                                .as_ref()
                                .expect("missing merkle records"),
                            false,
                        );
                        Some(merkle_tree_ctx)
                    }
                }
            }
            TouchedMemory::Volatile(partition) => {
                assert!(self.persistent.is_none(), "TouchedMemory enum mismatch");
                self.boundary.finalize_records_volatile(partition);
                None
            }
        };
        mem.tracing_info("boundary tracegen");
        let mut ret = vec![self.boundary.generate_proving_ctx(())];
        if let Some(merkle_proof_ctx) = merkle_proof_ctx {
            ret.push(merkle_proof_ctx);
            mem.tracing_info("dropping merkle tree");
            let persistent = self.persistent.as_mut().unwrap();
            persistent.merkle_tree.drop_subtrees();
            persistent.initial_memory = Vec::new();
        }
        ret.extend(
            self.access_adapters
                .generate_air_proving_ctxs(access_adapter_arena),
        );
        ret
    }
}

impl Drop for PersistentMemoryInventoryGPU {
    fn drop(&mut self) {
        // WARNING: The merkle subtree events must be completed before dropping the initial memory
        // buffers. This prevents buffers from dropping before build_async completes.
        self.merkle_tree.drop_subtrees();
        self.initial_memory.clear();
    }
}
