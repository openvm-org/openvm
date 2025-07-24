use std::sync::Arc;

use openvm_circuit::{
    arch::{DenseRecordArena, MemoryConfig, ADDR_SPACE_OFFSET},
    system::{
        memory::{AddressMap, MemoryImage},
        TouchedMemory,
    },
};
use openvm_stark_backend::{p3_util::log2_ceil_usize, prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{prover_backend::GpuBackend, types::F};

use crate::{
    get_empty_air_proving_ctx,
    primitives::var_range::VariableRangeCheckerChipGPU,
    system::{
        access_adapters::AccessAdapterInventoryGPU, boundary::BoundaryChipGPU,
        poseidon2::SharedBuffer,
    },
};

pub struct MemoryInventoryGPU {
    pub boundary: BoundaryChipGPU,
    pub access_adapters: AccessAdapterInventoryGPU,
    pub persistent: Option<PersistentMemoryInventoryGPU>,
}

pub struct PersistentMemoryInventoryGPU {
    // TODO[INT-4453]: Add MerkleChipGPU
    pub hasher_buffer: SharedBuffer<F>,
    // TODO[INT-4453]: Replace MemoryImage with GPU version
    pub initial_memory: MemoryImage,
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
            access_adapters: AccessAdapterInventoryGPU::new(range_checker),
            persistent: None,
        }
    }

    pub fn persistent(
        config: MemoryConfig,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        hasher_buffer: SharedBuffer<F>,
        sbox_regs: usize,
    ) -> Self {
        Self {
            boundary: BoundaryChipGPU::persistent(hasher_buffer.clone(), sbox_regs),
            access_adapters: AccessAdapterInventoryGPU::new(range_checker),
            persistent: Some(PersistentMemoryInventoryGPU {
                hasher_buffer,
                initial_memory: MemoryImage::from_mem_config(&config),
            }),
        }
    }

    pub fn continuation_enabled(&self) -> bool {
        self.persistent.is_some()
    }

    pub fn set_initial_memory(&mut self, initial_memory: AddressMap) {
        if let Some(persistent) = &mut self.persistent {
            // TODO[INT-4453]: Convert AddressMap to GPU version, pass it to merkle and boundary chips
            persistent.initial_memory = initial_memory;
        }
    }

    pub fn generate_proving_ctxs(
        &mut self,
        access_adapter_arena: DenseRecordArena,
        touched_memory: TouchedMemory<F>,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let is_persistent = match touched_memory {
            TouchedMemory::Persistent(partition) => {
                assert!(self.persistent.is_some(), "TouchedMemory enum mismatch");
                self.boundary.finalize_records(partition);
                // TODO[INT-4453]: Finalize MerkleChipGPU
                true
            }
            TouchedMemory::Volatile(partition) => {
                assert!(self.persistent.is_none(), "TouchedMemory enum mismatch");
                self.boundary.finalize_records(partition);
                false
            }
        };
        let mut ret = vec![self.boundary.generate_proving_ctx(())];
        // TODO[INT-4453]: Push MerkleChipGPU proving ctx if persistent
        if is_persistent {
            ret.push(get_empty_air_proving_ctx());
        }
        ret.extend(
            self.access_adapters
                .generate_air_proving_ctxs(&access_adapter_arena),
        );
        ret
    }
}
