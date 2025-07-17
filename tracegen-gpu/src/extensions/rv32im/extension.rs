use std::sync::Arc;

use openvm_circuit::arch::{
    AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, DenseRecordArena,
    ExecutorInventoryBuilder, ExecutorInventoryError, VmCircuitExtension, VmExecutionExtension,
    VmProverExtension,
};
use openvm_rv32im_circuit::{
    Rv32AuipcAir, Rv32BaseAluAir, Rv32BranchEqualAir, Rv32BranchLessThanAir, Rv32DivRemAir,
    Rv32HintStoreAir, Rv32I, Rv32IExecutor, Rv32Io, Rv32IoExecutor, Rv32JalLuiAir, Rv32JalrAir,
    Rv32LessThanAir, Rv32LoadSignExtendAir, Rv32LoadStoreAir, Rv32M, Rv32MExecutor, Rv32MulHAir,
    Rv32MultiplicationAir, Rv32ShiftAir,
};
use openvm_stark_backend::config::StarkGenericConfig;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::PrimeField32;
use stark_backend_gpu::prover_backend::GpuBackend;

use crate::{
    extensions::rv32im::{
        Rv32AuipcChipGpu, Rv32BaseAluChipGpu, Rv32BranchEqualChipGpu, Rv32BranchLessThanChipGpu,
        Rv32DivRemChipGpu, Rv32HintStoreChipGpu, Rv32JalLuiChipGpu, Rv32JalrChipGpu,
        Rv32LessThanChipGpu, Rv32LoadSignExtendChipGpu, Rv32LoadStoreChipGpu, Rv32MulHChipGpu,
        Rv32MultiplicationChipGpu, Rv32ShiftChipGpu,
    },
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
        var_range::VariableRangeCheckerChipGPU,
    },
};

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32IGpu(Rv32I);

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32MGpu(Rv32M);

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32IoGpu(Rv32Io);

// This implementation is specific to GpuBackend because the lookup chips (VariableRangeCheckerChipGPU,
// BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend> for Rv32IGpu {
    fn extend_prover(
        &self,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();

        // TODO[arayi]: make this a util method of ChipInventory.
        // Note, `VariableRangeCheckerChipGPU` always will always exist in the inventory.
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

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv32BaseAluAir>()?;
        let base_alu = Rv32BaseAluChipGpu::new(range_checker.clone(), bitwise_lu.clone());
        inventory.add_executor_chip(base_alu);

        inventory.next_air::<Rv32LessThanAir>()?;
        let lt = Rv32LessThanChipGpu::new(range_checker.clone(), bitwise_lu.clone());
        inventory.add_executor_chip(lt);

        inventory.next_air::<Rv32ShiftAir>()?;
        let shift = Rv32ShiftChipGpu::new(range_checker.clone(), bitwise_lu.clone());
        inventory.add_executor_chip(shift);

        inventory.next_air::<Rv32LoadStoreAir>()?;
        let load_store_chip = Rv32LoadStoreChipGpu::new(range_checker.clone(), pointer_max_bits);
        inventory.add_executor_chip(load_store_chip);

        inventory.next_air::<Rv32LoadSignExtendAir>()?;
        let load_sign_extend =
            Rv32LoadSignExtendChipGpu::new(range_checker.clone(), pointer_max_bits);
        inventory.add_executor_chip(load_sign_extend);

        inventory.next_air::<Rv32BranchEqualAir>()?;
        let beq = Rv32BranchEqualChipGpu::new(range_checker.clone());
        inventory.add_executor_chip(beq);

        inventory.next_air::<Rv32BranchLessThanAir>()?;
        let blt = Rv32BranchLessThanChipGpu::new(range_checker.clone(), bitwise_lu.clone());
        inventory.add_executor_chip(blt);

        inventory.next_air::<Rv32JalLuiAir>()?;
        let jal_lui = Rv32JalLuiChipGpu::new(range_checker.clone(), bitwise_lu.clone());
        inventory.add_executor_chip(jal_lui);

        inventory.next_air::<Rv32JalrAir>()?;
        let jalr = Rv32JalrChipGpu::new(range_checker.clone(), bitwise_lu.clone());
        inventory.add_executor_chip(jalr);

        inventory.next_air::<Rv32AuipcAir>()?;
        let auipc = Rv32AuipcChipGpu::new(range_checker.clone(), bitwise_lu.clone());
        inventory.add_executor_chip(auipc);

        Ok(())
    }
}

// This implementation is specific to GpuBackend because the lookup chips (VariableRangeCheckerChipGPU,
// BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend> for Rv32MGpu {
    fn extend_prover(
        &self,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();

        // TODO[arayi]: make this a util method of ChipInventory.
        // Note, `VariableRangeCheckerChipGPU` always will always exist in the inventory.
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

        let range_tuple_checker = {
            let existing_chip = inventory
                .find_chip::<Arc<RangeTupleCheckerChipGPU<2>>>()
                .find(|c| {
                    c.sizes[0] >= self.0.range_tuple_checker_sizes[0]
                        && c.sizes[1] >= self.0.range_tuple_checker_sizes[1]
                });
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let chip = Arc::new(RangeTupleCheckerChipGPU::new(
                    self.0.range_tuple_checker_sizes,
                ));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv32MultiplicationAir>()?;
        let mult =
            Rv32MultiplicationChipGpu::new(range_checker.clone(), range_tuple_checker.clone());
        inventory.add_executor_chip(mult);

        inventory.next_air::<Rv32MulHAir>()?;
        let mul_h = Rv32MulHChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
        );
        inventory.add_executor_chip(mul_h);

        inventory.next_air::<Rv32DivRemAir>()?;
        let div_rem = Rv32DivRemChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            pointer_max_bits,
        );
        inventory.add_executor_chip(div_rem);

        Ok(())
    }
}

// This implementation is specific to GpuBackend because the lookup chips (VariableRangeCheckerChipGPU,
// BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend> for Rv32IoGpu {
    fn extend_prover(
        &self,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();

        // TODO[arayi]: make this a util method of ChipInventory.
        // Note, `VariableRangeCheckerChipGPU` always will always exist in the inventory.
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

        inventory.next_air::<Rv32HintStoreAir>()?;
        let hint_store =
            Rv32HintStoreChipGpu::new(range_checker.clone(), bitwise_lu.clone(), pointer_max_bits);
        inventory.add_executor_chip(hint_store);

        Ok(())
    }
}

impl<F: PrimeField32> VmExecutionExtension<F> for Rv32IGpu {
    type Executor = Rv32IExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Rv32IExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        self.0.extend_execution(inventory)
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Rv32IGpu {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        self.0.extend_circuit(inventory)
    }
}

impl<F> VmExecutionExtension<F> for Rv32MGpu {
    type Executor = Rv32MExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Rv32MExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        self.0.extend_execution(inventory)
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Rv32MGpu {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        self.0.extend_circuit(inventory)
    }
}

impl<F> VmExecutionExtension<F> for Rv32IoGpu {
    type Executor = Rv32IoExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Rv32IoExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        self.0.extend_execution(inventory)
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Rv32IoGpu {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        self.0.extend_circuit(inventory)
    }
}
