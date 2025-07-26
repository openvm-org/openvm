use std::sync::Arc;

use openvm_circuit::{
    arch::{
        Arena, ChipInventory, ChipInventoryError, DenseRecordArena, MatrixRecordArena,
        MultiRowLayout, VmChipWrapper, VmProverExtension,
    },
    system::memory::SharedMemoryHelper,
};
use openvm_sha256_circuit::{
    Sha256, Sha256VmAir, Sha256VmChip, Sha256VmFiller, Sha256VmMetadata, Sha256VmRecordMut,
    SHA256VM_WIDTH,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_baby_bear::BabyBear;
use stark_backend_gpu::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};

use crate::{
    cpu_proving_ctx_to_gpu, get_empty_air_proving_ctx,
    primitives::bitwise_op_lookup::BitwiseOperationLookupChipGPU,
    system::extensions::get_inventory_range_checker,
};

pub struct Sha2GpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Sha256> for Sha2GpuProverExt {
    fn extend_prover(
        &self,
        _: &Sha256,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let mem_helper =
            SharedMemoryHelper::new(range_checker.cpu_chip.clone().unwrap(), timestamp_max_bits);

        let bitwise_lu = {
            let existing_chip = inventory
                .find_chip::<Arc<BitwiseOperationLookupChipGPU<8>>>()
                .next();
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let chip = Arc::new(BitwiseOperationLookupChipGPU::new());
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        inventory.next_air::<Sha256VmAir>()?;
        let cpu_sha256 = VmChipWrapper::new(
            Sha256VmFiller::new(bitwise_lu.cpu_chip.clone().unwrap(), pointer_max_bits),
            mem_helper,
        );
        let sha256 = HybridSha256VmChip { cpu: cpu_sha256 };
        inventory.add_executor_chip(sha256);

        Ok(())
    }
}

pub struct HybridSha256VmChip {
    pub cpu: Sha256VmChip<BabyBear>,
}

impl Chip<DenseRecordArena, GpuBackend> for HybridSha256VmChip {
    fn generate_proving_ctx(
        &self,
        mut dense_arena: DenseRecordArena,
    ) -> AirProvingContext<GpuBackend> {
        if dense_arena.current_size() == 0 {
            return get_empty_air_proving_ctx();
        }
        // Lazy HACK: for now even a dense_arena's capacity is the matrix size
        let rows_used = dense_arena.capacity().div_ceil(SHA256VM_WIDTH * 4);
        let height = rows_used.next_power_of_two();
        let mut seeker =
            dense_arena.get_record_seeker::<Sha256VmRecordMut, MultiRowLayout<Sha256VmMetadata>>();
        let mut matrix_arena = MatrixRecordArena::<BabyBear>::with_capacity(height, SHA256VM_WIDTH);
        seeker.transfer_to_matrix_arena(&mut matrix_arena);
        let ctx = self.cpu.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_gpu(ctx)
    }
}
