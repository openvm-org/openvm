use std::sync::Arc;

use openvm_circuit::{
    arch::{ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension},
    system::cuda::extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
};
use openvm_circuit_primitives::range_tuple::{RangeTupleCheckerAir, RangeTupleCheckerChipGPU};
use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::{
    Rv64AuipcAir, Rv64AuipcChipGpu, Rv64BaseAluAir, Rv64BaseAluChipGpu, Rv64BranchEqualAir,
    Rv64BranchEqualChipGpu, Rv64BranchLessThanAir, Rv64BranchLessThanChipGpu, Rv64DivRemAir,
    Rv64DivRemChipGpu, Rv64HintStoreAir, Rv64HintStoreChipGpu, Rv64I, Rv64Io, Rv64JalLuiAir,
    Rv64JalLuiChipGpu, Rv64JalrAir, Rv64JalrChipGpu, Rv64LessThanAir, Rv64LessThanChipGpu,
    Rv64LoadSignExtendAir, Rv64LoadSignExtendChipGpu, Rv64LoadStoreAir, Rv64LoadStoreChipGpu,
    Rv64M, Rv64MulHAir, Rv64MulHChipGpu, Rv64MultiplicationAir, Rv64MultiplicationChipGpu,
    Rv64ShiftAir, Rv64ShiftChipGpu,
};

pub struct Rv64ImGpuProverExt;

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Rv64I> for Rv64ImGpuProverExt {
    fn extend_prover(
        &self,
        _: &Rv64I,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv64BaseAluAir>()?;
        let base_alu = Rv64BaseAluChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(base_alu);

        inventory.next_air::<Rv64LessThanAir>()?;
        let lt = Rv64LessThanChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(lt);

        inventory.next_air::<Rv64ShiftAir>()?;
        let shift = Rv64ShiftChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(shift);

        inventory.next_air::<Rv64LoadStoreAir>()?;
        let load_store_chip =
            Rv64LoadStoreChipGpu::new(range_checker.clone(), pointer_max_bits, timestamp_max_bits);
        inventory.add_executor_chip(load_store_chip);

        inventory.next_air::<Rv64LoadSignExtendAir>()?;
        let load_sign_extend = Rv64LoadSignExtendChipGpu::new(
            range_checker.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_sign_extend);

        inventory.next_air::<Rv64BranchEqualAir>()?;
        let beq = Rv64BranchEqualChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(beq);

        inventory.next_air::<Rv64BranchLessThanAir>()?;
        let blt = Rv64BranchLessThanChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(blt);

        inventory.next_air::<Rv64JalLuiAir>()?;
        let jal_lui = Rv64JalLuiChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(jal_lui);

        inventory.next_air::<Rv64JalrAir>()?;
        let jalr = Rv64JalrChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(jalr);

        inventory.next_air::<Rv64AuipcAir>()?;
        let auipc = Rv64AuipcChipGpu::new(
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
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Rv64M> for Rv64ImGpuProverExt {
    fn extend_prover(
        &self,
        extension: &Rv64M,
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
        inventory.next_air::<Rv64MultiplicationAir>()?;
        let mult = Rv64MultiplicationChipGpu::new(
            range_checker.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mult);

        inventory.next_air::<Rv64MulHAir>()?;
        let mul_h = Rv64MulHChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mul_h);

        inventory.next_air::<Rv64DivRemAir>()?;
        let div_rem = Rv64DivRemChipGpu::new(
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
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Rv64Io>
    for Rv64ImGpuProverExt
{
    fn extend_prover(
        &self,
        _: &Rv64Io,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        inventory.next_air::<Rv64HintStoreAir>()?;
        let hint_store = Rv64HintStoreChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(hint_store);

        Ok(())
    }
}
