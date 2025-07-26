use std::sync::Arc;

use openvm_circuit::arch::{
    ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension,
};
use openvm_ecc_circuit::{WeierstrassAir, WeierstrassExtension};
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::LocalOpcode;
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_gpu::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
use strum::EnumCount;

use crate::{
    extensions::ecc::{WeierstrassAddNeChipGpu, WeierstrassDoubleChipGpu},
    primitives::bitwise_op_lookup::BitwiseOperationLookupChipGPU,
    system::extensions::get_inventory_range_checker,
};

#[derive(Clone)]
pub struct EccGpuProverExt;

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, WeierstrassExtension>
    for EccGpuProverExt
{
    fn extend_prover(
        &self,
        extension: &WeierstrassExtension,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        // Range checker should always exist in inventory
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

        for (i, curve) in extension.supported_curves.iter().enumerate() {
            let start_offset =
                Rv32WeierstrassOpcode::CLASS_OFFSET + i * Rv32WeierstrassOpcode::COUNT;
            let bytes = curve.modulus.bits().div_ceil(8);

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                inventory.next_air::<WeierstrassAir<2, 2, 32>>()?;
                let addne = WeierstrassAddNeChipGpu::<2, 32>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config.clone(),
                    start_offset,
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(addne);

                inventory.next_air::<WeierstrassAir<1, 2, 32>>()?;
                let double = WeierstrassDoubleChipGpu::<2, 32>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config,
                    start_offset,
                    curve.a.clone(),
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(double);
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };

                inventory.next_air::<WeierstrassAir<2, 6, 16>>()?;
                let addne = WeierstrassAddNeChipGpu::<6, 16>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config.clone(),
                    start_offset,
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(addne);

                inventory.next_air::<WeierstrassAir<1, 6, 16>>()?;
                let double = WeierstrassDoubleChipGpu::<6, 16>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config,
                    start_offset,
                    curve.a.clone(),
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(double);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}
