use std::sync::Arc;

use openvm_circuit::{
    arch::{
        AirInventory, ChipInventory, ChipInventoryError, DenseRecordArena, SystemConfig, VmBuilder,
        VmChipComplex, PUBLIC_VALUES_AIR_ID,
    },
    system::poseidon2::air::Poseidon2PeripheryAir,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupChip, BitwiseOperationLookupChipGPU,
    },
    var_range::{VariableRangeCheckerAir, VariableRangeCheckerChip, VariableRangeCheckerChipGPU},
};
use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_baby_bear::BabyBear;

use super::{
    phantom::PhantomChipGPU, Poseidon2PeripheryChipGPU, SystemChipInventoryGPU, DIGEST_WIDTH,
};

/// A utility method to get the `VariableRangeCheckerChipGPU` from [ChipInventory].
/// Note, `VariableRangeCheckerChipGPU` always will always exist in the inventory.
pub fn get_inventory_range_checker(
    inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
) -> Arc<VariableRangeCheckerChipGPU> {
    inventory
        .find_chip::<Arc<VariableRangeCheckerChipGPU>>()
        .next()
        .unwrap()
        .clone()
}

/// A utility method to find a **byte** [BitwiseOperationLookupChipGPU] or create one and add
/// to the inventory if it does not exist.
pub fn get_or_create_bitwise_op_lookup(
    inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
) -> Result<Arc<BitwiseOperationLookupChipGPU<8>>, ChipInventoryError> {
    let bitwise_lu = {
        let existing_chip = inventory
            .find_chip::<Arc<BitwiseOperationLookupChipGPU<8>>>()
            .next();
        if let Some(chip) = existing_chip {
            chip.clone()
        } else {
            let air: &BitwiseOperationLookupAir<8> = inventory.next_air()?;

            let chip = Arc::new(BitwiseOperationLookupChipGPU::hybrid(Arc::new(
                BitwiseOperationLookupChip::new(air.bus),
            )));
            inventory.add_periphery_chip(chip.clone());
            chip
        }
    };
    Ok(bitwise_lu)
}

/// **If** internal poseidon2 chip exists, then its insertion index is 1.
const POSEIDON2_INSERTION_IDX: usize = 1;
/// **If** public values chip exists, then its executor index is 0.
pub const PV_EXECUTOR_IDX: usize = 0;

#[derive(Clone)]
pub struct SystemGpuBuilder;

impl VmBuilder<GpuBabyBearPoseidon2Engine> for SystemGpuBuilder {
    type VmConfig = SystemConfig;
    type RecordArena = DenseRecordArena;
    type SystemChipInventory = SystemChipInventoryGPU;

    fn create_chip_complex(
        &self,
        config: &SystemConfig,
        airs: AirInventory<BabyBearPoseidon2Config>,
    ) -> Result<
        VmChipComplex<
            BabyBearPoseidon2Config,
            DenseRecordArena,
            GpuBackend,
            SystemChipInventoryGPU,
        >,
        ChipInventoryError,
    > {
        let range_bus = airs.range_checker().bus;
        let range_checker = Arc::new(VariableRangeCheckerChipGPU::hybrid(Arc::new(
            VariableRangeCheckerChip::new(range_bus),
        )));

        let mut inventory = ChipInventory::new(airs);
        // PublicValuesChip is required when num_public_values > 0 in single segment mode.
        if config.has_public_values_chip() {
            assert_eq!(
                inventory.executor_idx_to_insertion_idx.len(),
                PV_EXECUTOR_IDX
            );

            // We set insertion_idx so that air_idx = num_airs - (insertion_idx + 1) =
            // PUBLIC_VALUES_AIR_ID in `VmChipComplex::executor_idx_to_air_idx`. We need to do this
            // because this chip is special and not part of the normal inventory.
            let insertion_idx = inventory
                .airs()
                .num_airs()
                .checked_sub(1 + PUBLIC_VALUES_AIR_ID)
                .unwrap();
            inventory.executor_idx_to_insertion_idx.push(insertion_idx);
        }
        inventory.next_air::<VariableRangeCheckerAir>()?;
        inventory.add_periphery_chip(range_checker.clone());

        let hasher_chip = if config.continuation_enabled {
            let max_buffer_size = (config.segmentation_limits.max_trace_height as usize)
                .next_power_of_two() * 2 // seems like a reliable estimate
                * (DIGEST_WIDTH * 2); // size of one record
            assert_eq!(inventory.chips().len(), POSEIDON2_INSERTION_IDX);
            let sbox_registers = if config.max_constraint_degree >= 7 {
                0
            } else {
                1
            };
            // ATTENTION: The threshold 7 here must match the one in `new_poseidon2_periphery_air`
            let _direct_bus = if sbox_registers == 0 {
                inventory
                    .next_air::<Poseidon2PeripheryAir<BabyBear, 0>>()?
                    .bus
            } else {
                inventory
                    .next_air::<Poseidon2PeripheryAir<BabyBear, 1>>()?
                    .bus
            };
            let chip = Arc::new(Poseidon2PeripheryChipGPU::new(
                max_buffer_size,
                sbox_registers,
            ));
            inventory.add_periphery_chip(chip.clone());
            Some(chip)
        } else {
            None
        };
        let system = SystemChipInventoryGPU::new(
            config,
            &inventory.airs().system().memory,
            range_checker,
            hasher_chip,
        );

        let phantom_chip = PhantomChipGPU::new();
        inventory.add_executor_chip(phantom_chip);

        Ok(VmChipComplex { system, inventory })
    }
}
