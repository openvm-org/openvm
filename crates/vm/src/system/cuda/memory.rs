use std::sync::Arc;

use openvm_circuit::{
    arch::{AddressSpaceHostLayout, DenseRecordArena, MemoryConfig, ADDR_SPACE_OFFSET},
    system::{
        memory::{online::LinearMemory, AddressMap, TimestampedValues},
        TouchedMemory,
    },
};
use openvm_circuit_primitives::var_range::cuda::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{prover_backend::GpuBackend, types::F};
use openvm_cuda_common::{
    copy::{cuda_memcpy, MemCopyD2D, MemCopyH2D},
    d_buffer::DeviceBuffer,
    memory_manager::MemTracker,
};
use openvm_stark_backend::{
    p3_field::FieldAlgebra, p3_util::log2_ceil_usize, prover::types::AirProvingContext, Chip,
};

use super::{
    access_adapters::AccessAdapterInventoryGPU,
    boundary::{BoundaryChipGPU, BoundaryFields},
    merkle_tree::{MemoryMerkleTree, TIMESTAMPED_BLOCK_WIDTH},
    Poseidon2PeripheryChipGPU, DIGEST_WIDTH,
};

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
}

impl MemoryInventoryGPU {
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
                    .as_ref()
                    .expect("persistent touched memory requires persistent memory interface");

                let unpadded_merkle_height =
                    persistent.merkle_tree.calculate_unpadded_height(&partition);
                #[cfg(feature = "metrics")]
                {
                    self.unpadded_merkle_height = unpadded_merkle_height;
                }

                mem.tracing_info("boundary finalize");
                let (touched_memory, empty) = if partition.is_empty() {
                    let leftmost_values = 'left: {
                        let mut res = [F::ZERO; DIGEST_WIDTH];
                        if persistent.initial_memory[ADDR_SPACE_OFFSET as usize].is_empty() {
                            break 'left res;
                        }
                        let layout = &persistent.merkle_tree.mem_config().addr_spaces
                            [ADDR_SPACE_OFFSET as usize]
                            .layout;
                        let one_cell_size = layout.size();
                        let values = vec![0u8; one_cell_size * DIGEST_WIDTH];
                        unsafe {
                            cuda_memcpy::<true, false>(
                                values.as_ptr() as *mut std::ffi::c_void,
                                persistent.initial_memory[ADDR_SPACE_OFFSET as usize].as_ptr()
                                    as *const std::ffi::c_void,
                                values.len(),
                            )
                            .unwrap();
                            for i in 0..DIGEST_WIDTH {
                                res[i] = layout.to_field::<F>(&values[i * one_cell_size..]);
                            }
                        }
                        res
                    };

                    (
                        vec![(
                            (1, 0),
                            TimestampedValues {
                                timestamp: 0,
                                values: leftmost_values,
                            },
                        )],
                        true,
                    )
                } else {
                    (partition, false)
                };
                debug_assert_eq!(
                    size_of_val(&touched_memory[0]),
                    TIMESTAMPED_BLOCK_WIDTH * size_of::<u32>()
                );
                let d_touched_memory = touched_memory.to_device().unwrap().as_buffer::<u32>();
                if empty {
                    self.boundary
                        .finalize_records_persistent::<DIGEST_WIDTH>(DeviceBuffer::new());
                } else {
                    self.boundary.finalize_records_persistent::<DIGEST_WIDTH>(
                        d_touched_memory.device_copy().unwrap().as_buffer::<u32>(),
                    ); // TODO do not copy
                }
                mem.tracing_info("merkle update");
                persistent.merkle_tree.finalize();
                Some(persistent.merkle_tree.update_with_touched_blocks(
                    unpadded_merkle_height,
                    &d_touched_memory,
                    empty,
                ))
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

#[cfg(test)]
mod tests {
    use std::array;

    use openvm_circuit::arch::MemoryConfig;
    use openvm_cuda_backend::prelude::F;
    use openvm_instructions::{
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
        NATIVE_AS,
    };
    use openvm_stark_sdk::utils::create_seeded_rng;
    use p3_field::FieldAlgebra;
    use rand::Rng;

    use crate::{
        arch::testing::{GpuChipTestBuilder, TestBuilder},
        utils::cuda_utils::{default_bitwise_lookup_bus, default_var_range_checker_bus},
    };

    #[test]
    fn test_cuda_persistent_memory() {
        let mut rng = create_seeded_rng();
        let mem_config = {
            let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
            let max_cells = 1 << 12;
            addr_spaces[RV32_REGISTER_AS as usize].num_cells = 32 * size_of::<u32>();
            addr_spaces[RV32_MEMORY_AS as usize].num_cells = max_cells;
            addr_spaces[NATIVE_AS as usize].num_cells = max_cells;
            MemoryConfig::new(2, addr_spaces, max_cells.ilog2() as usize, 29, 17, 32)
        };
        let mut tester =
            GpuChipTestBuilder::persistent(mem_config.clone(), default_var_range_checker_bus())
                .with_bitwise_op_lookup(default_bitwise_lookup_bus());
        tester.write(1, 4, [F::ONE, F::ONE, F::ONE, F::ONE]);

        let values_bounds = [256, 256, 256, (1 << 30)];
        let max_log_block_size = 4;

        let its = 20;
        for _ in 0..its {
            let addr_sp = rng.gen_range(1..=values_bounds.len());
            let align: usize = mem_config.addr_spaces[addr_sp].min_block_size;
            let value_bound: u32 = values_bounds[addr_sp - 1];
            if mem_config.addr_spaces[addr_sp].num_cells == 0 {
                continue;
            }
            let ptr = rng.gen_range(0..mem_config.addr_spaces[addr_sp].num_cells / align) * align;
            let log_len = rng.gen_range(align.trailing_zeros()..=max_log_block_size);
            if ptr + (1 << log_len) > mem_config.addr_spaces[addr_sp].num_cells {
                continue;
            }
            match log_len {
                0 => tester.write::<1>(
                    addr_sp,
                    ptr,
                    array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
                ),
                1 => tester.write::<2>(
                    addr_sp,
                    ptr,
                    array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
                ),
                2 => tester.write::<4>(
                    addr_sp,
                    ptr,
                    array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
                ),
                3 => tester.write::<8>(
                    addr_sp,
                    ptr,
                    array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
                ),
                4 => tester.write::<16>(
                    addr_sp,
                    ptr,
                    array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
                ),
                _ => unreachable!(),
            }
        }

        tester.build().finalize().simple_test().unwrap();
    }

    #[test]
    fn test_cuda_persistent_memory_do_not_touch_anything() {
        let mem_config = {
            let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
            let max_cells = 1 << 12;
            addr_spaces[RV32_REGISTER_AS as usize].num_cells = 32 * size_of::<u32>();
            addr_spaces[RV32_MEMORY_AS as usize].num_cells = max_cells;
            MemoryConfig::new(2, addr_spaces, max_cells.ilog2() as usize, 29, 17, 32)
        };
        let tester = GpuChipTestBuilder::persistent(mem_config, default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());

        tester.build().finalize().simple_test().unwrap();
    }
}
