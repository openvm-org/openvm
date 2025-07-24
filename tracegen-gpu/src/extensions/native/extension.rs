use std::sync::Arc;

use openvm_circuit::arch::{
    AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, DenseRecordArena,
    ExecutorInventoryBuilder, ExecutorInventoryError, VmCircuitExtension, VmExecutionExtension,
    VmProverExtension,
};
use openvm_native_circuit::{
    air::NativePoseidon2Air, CastFAir, CastFCoreAir, CastFStep, FieldArithmeticAir,
    FieldArithmeticCoreAir, FieldArithmeticStep, FieldExtensionAir, FieldExtensionCoreAir,
    FieldExtensionStep, FriReducedOpeningAir, JalRangeCheckAir, NativeBranchEqAir,
    NativeLoadStoreAir, NativeLoadStoreCoreAir,
};
use openvm_poseidon2_air::Poseidon2Config;
use openvm_stark_backend::config::StarkGenericConfig;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::PrimeField32;
use stark_backend_gpu::{prover_backend::GpuBackend, types::F};

use crate::{
    extensions::native::{
        CastFChipGpu, FieldArithmeticChipGpu, FieldExtensionChipGpu, FriReducedOpeningChipGpu,
        JalRangeCheckGpu, NativeBranchEqChipGpu, NativeLoadStoreChipGpu, NativePoseidon2ChipGpu,
    },
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
        var_range::VariableRangeCheckerChipGPU,
    },
};

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeGpu;

impl VmProverExtension<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend> for NativeGpu {
    fn extend_prover(
        &self,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = inventory
            .find_chip::<Arc<VariableRangeCheckerChipGPU>>()
            .next()
            .unwrap()
            .clone();

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

        inventory.next_air::<NativeBranchEqAir>()?;
        let branch_eq = NativeBranchEqChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(branch_eq);

        inventory.next_air::<CastFAir>()?;
        let castf = CastFChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(castf);

        inventory.next_air::<FieldArithmeticAir>()?;
        let field_arithmetic =
            FieldArithmeticChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(field_arithmetic);

        inventory.next_air::<FieldExtensionAir>()?;
        let field_extension = FieldExtensionChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(field_extension);

        inventory.next_air::<FriReducedOpeningAir>()?;
        let fri = FriReducedOpeningChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(fri);

        inventory.next_air::<JalRangeCheckAir>()?;
        let jal_rangecheck = JalRangeCheckGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(jal_rangecheck);

        inventory.next_air::<NativeLoadStoreAir<1>>()?;
        let load_store =
            NativeLoadStoreChipGpu::<1>::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(load_store);

        inventory.next_air::<NativeLoadStoreAir<4>>()?;
        let block_load_store =
            NativeLoadStoreChipGpu::<4>::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(block_load_store);

        inventory.next_air::<NativePoseidon2Air<F, 1>>()?;
        let poseidon2 = NativePoseidon2ChipGpu::<1>::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(poseidon2);

        Ok(())
    }
}
