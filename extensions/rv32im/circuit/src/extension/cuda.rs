use std::sync::Arc;

use openvm_circuit::{
    arch::{ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension},
    system::cuda::extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
};
use openvm_circuit_primitives::range_tuple::{RangeTupleCheckerAir, RangeTupleCheckerChipGPU};
use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::{
    Rv32AuipcAir, Rv32AuipcChipGpu, Rv32BaseAluAir, Rv32BaseAluChipGpu, Rv32BranchEqualAir,
    Rv32BranchEqualChipGpu, Rv32BranchLessThanAir, Rv32BranchLessThanChipGpu, Rv32DivRemAir,
    Rv32DivRemChipGpu, Rv32HintStoreAir, Rv32HintStoreChipGpu, Rv32I, Rv32Io, Rv32JalLuiAir,
    Rv32JalLuiChipGpu, Rv32JalrAir, Rv32JalrChipGpu, Rv32LessThanAir, Rv32LessThanChipGpu,
    Rv32LoadSignExtendAir, Rv32LoadSignExtendChipGpu, Rv32LoadStoreAir, Rv32LoadStoreChipGpu,
    Rv32M, Rv32MulHAir, Rv32MulHChipGpu, Rv32MultiplicationAir, Rv32MultiplicationChipGpu,
    Rv32ShiftAir, Rv32ShiftChipGpu,
};

pub struct Rv32ImGpuProverExt;

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Rv32I> for Rv32ImGpuProverExt {
    fn extend_prover(
        &self,
        _: &Rv32I,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv32BaseAluAir>()?;
        let base_alu = Rv32BaseAluChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(base_alu);

        inventory.next_air::<Rv32LessThanAir>()?;
        let lt = Rv32LessThanChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(lt);

        inventory.next_air::<Rv32ShiftAir>()?;
        let shift = Rv32ShiftChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(shift);

        inventory.next_air::<Rv32LoadStoreAir>()?;
        let load_store_chip =
            Rv32LoadStoreChipGpu::new(range_checker.clone(), pointer_max_bits, timestamp_max_bits);
        inventory.add_executor_chip(load_store_chip);

        inventory.next_air::<Rv32LoadSignExtendAir>()?;
        let load_sign_extend = Rv32LoadSignExtendChipGpu::new(
            range_checker.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_sign_extend);

        inventory.next_air::<Rv32BranchEqualAir>()?;
        let beq = Rv32BranchEqualChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(beq);

        inventory.next_air::<Rv32BranchLessThanAir>()?;
        let blt = Rv32BranchLessThanChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(blt);

        inventory.next_air::<Rv32JalLuiAir>()?;
        let jal_lui = Rv32JalLuiChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(jal_lui);

        inventory.next_air::<Rv32JalrAir>()?;
        let jalr = Rv32JalrChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(jalr);

        inventory.next_air::<Rv32AuipcAir>()?;
        let auipc = Rv32AuipcChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(auipc);

        Ok(())
    }
}

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Rv32M> for Rv32ImGpuProverExt {
    fn extend_prover(
        &self,
        extension: &Rv32M,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        let range_tuple_checker = {
            let existing_chip = inventory
                .find_chip::<Arc<RangeTupleCheckerChipGPU<2>>>()
                .find(|c| {
                    c.sizes[0] >= extension.range_tuple_checker_sizes[0]
                        && c.sizes[1] >= extension.range_tuple_checker_sizes[1]
                });
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                inventory.next_air::<RangeTupleCheckerAir<2>>()?;
                let chip = Arc::new(RangeTupleCheckerChipGPU::new(
                    extension.range_tuple_checker_sizes,
                ));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv32MultiplicationAir>()?;
        let mult = Rv32MultiplicationChipGpu::new(
            range_checker.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mult);

        inventory.next_air::<Rv32MulHAir>()?;
        let mul_h = Rv32MulHChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mul_h);

        inventory.next_air::<Rv32DivRemAir>()?;
        let div_rem = Rv32DivRemChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(div_rem);

        Ok(())
    }
}

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Rv32Io>
    for Rv32ImGpuProverExt
{
    fn extend_prover(
        &self,
        _: &Rv32Io,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        inventory.next_air::<Rv32HintStoreAir>()?;
        let hint_store = Rv32HintStoreChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(hint_store);

        Ok(())
    }
}
