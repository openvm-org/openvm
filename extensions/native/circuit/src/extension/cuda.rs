use openvm_circuit::{
    arch::{ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension},
    system::cuda::extensions::get_inventory_range_checker,
};
use openvm_cuda_backend::engine::GpuBabyBearPoseidon2Engine;
use openvm_native_compiler::BLOCK_LOAD_STORE_SIZE;
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config, p3_baby_bear::BabyBear,
};

use crate::{
    branch_eq::{NativeBranchEqAir, NativeBranchEqChipGpu},
    castf::{CastFAir, CastFChipGpu},
    field_arithmetic::{FieldArithmeticAir, FieldArithmeticChipGpu},
    field_extension::{FieldExtensionAir, FieldExtensionChipGpu},
    fri::{FriReducedOpeningAir, FriReducedOpeningChipGpu},
    jal_rangecheck::{JalRangeCheckAir, JalRangeCheckGpu},
    loadstore::{NativeLoadStoreAir, NativeLoadStoreChipGpu},
    poseidon2::{air::NativePoseidon2Air, NativePoseidon2ChipGpu},
    CastFExtension, GpuBackend, Native,
};

pub struct NativeGpuProverExt;
// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Native>
    for NativeGpuProverExt
{
    fn extend_prover(
        &self,
        _: &Native,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let range_checker = get_inventory_range_checker(inventory);

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<NativeLoadStoreAir<1>>()?;
        let load_store =
            NativeLoadStoreChipGpu::<1>::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(load_store);

        inventory.next_air::<NativeLoadStoreAir<BLOCK_LOAD_STORE_SIZE>>()?;
        let block_load_store = NativeLoadStoreChipGpu::<BLOCK_LOAD_STORE_SIZE>::new(
            range_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(block_load_store);

        inventory.next_air::<NativeBranchEqAir>()?;
        let branch_eq = NativeBranchEqChipGpu::new(range_checker.clone(), timestamp_max_bits);

        inventory.add_executor_chip(branch_eq);

        inventory.next_air::<JalRangeCheckAir>()?;
        let jal_rangecheck = JalRangeCheckGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(jal_rangecheck);

        inventory.next_air::<FieldArithmeticAir>()?;
        let field_arithmetic =
            FieldArithmeticChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(field_arithmetic);

        inventory.next_air::<FieldExtensionAir>()?;
        let field_extension = FieldExtensionChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(field_extension);

        inventory.next_air::<FriReducedOpeningAir>()?;
        let fri_reduced_opening =
            FriReducedOpeningChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(fri_reduced_opening);

        inventory.next_air::<NativePoseidon2Air<BabyBear, 1>>()?;
        let poseidon2 = NativePoseidon2ChipGpu::<1>::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(poseidon2);

        Ok(())
    }
}

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, CastFExtension>
    for NativeGpuProverExt
{
    fn extend_prover(
        &self,
        _: &CastFExtension,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let range_checker = get_inventory_range_checker(inventory);

        inventory.next_air::<CastFAir>()?;
        let castf = CastFChipGpu::new(range_checker, timestamp_max_bits);
        inventory.add_executor_chip(castf);

        Ok(())
    }
}
