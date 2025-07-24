use std::sync::Arc;

use openvm_circuit::arch::{
    AirInventory, ChipInventory, ChipInventoryError, DenseRecordArena, VmBuilder, VmChipComplex,
    VmProverExtension,
};
use openvm_keccak256_circuit::{Keccak256, Keccak256Rv32Config, KeccakVmAir};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_gpu::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};

use crate::{
    extensions::{keccak256::Keccak256ChipGpu, rv32im::Rv32ImGpuProverExt},
    primitives::bitwise_op_lookup::BitwiseOperationLookupChipGPU,
    system::{
        extensions::{get_inventory_range_checker, SystemGpuBuilder},
        SystemChipInventoryGPU,
    },
};

pub struct Keccak256GpuProverExt;

// This implementation is specific to GpuBackend because the lookup chips (VariableRangeCheckerChipGPU,
// BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Keccak256>
    for Keccak256GpuProverExt
{
    fn extend_prover(
        &self,
        _: &Keccak256,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
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

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<KeccakVmAir>()?;
        let keccak = Keccak256ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits as u32,
            timestamp_max_bits as u32,
        );
        inventory.add_executor_chip(keccak);

        Ok(())
    }
}

#[derive(Clone)]
pub struct Keccak256Rv32GpuBuilder;

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Keccak256Rv32GpuBuilder {
    type VmConfig = Keccak256Rv32Config;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Keccak256Rv32Config,
        circuit: AirInventory<BabyBearPoseidon2Config>,
    ) -> Result<
        VmChipComplex<
            BabyBearPoseidon2Config,
            Self::RecordArena,
            GpuBackend,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemGpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.rv32i, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.rv32m, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &Keccak256GpuProverExt,
            &config.keccak,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
