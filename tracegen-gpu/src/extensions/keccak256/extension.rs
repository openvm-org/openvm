use std::sync::Arc;

use openvm_circuit::arch::{
    AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, DenseRecordArena,
    ExecutorInventoryBuilder, ExecutorInventoryError, VmCircuitExtension, VmExecutionExtension,
    VmProverExtension,
};
use openvm_keccak256_circuit::{Keccak256, Keccak256Executor, KeccakVmAir};
use openvm_stark_backend::config::{StarkGenericConfig, Val};
use p3_field::PrimeField32;
use stark_backend_gpu::prover_backend::GpuBackend;

use crate::{
    extensions::keccak256::Keccak256ChipGpu,
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
};

#[derive(Clone, Copy, Debug, Default)]
pub struct Keccak256Gpu(Keccak256);

// This implementation is specific to GpuBackend because the lookup chips (VariableRangeCheckerChipGPU,
// BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl<SC> VmProverExtension<SC, DenseRecordArena, GpuBackend> for Keccak256Gpu
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        inventory: &mut ChipInventory<SC, DenseRecordArena, GpuBackend>,
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
        inventory.next_air::<KeccakVmAir>()?;
        let keccak = Keccak256ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits as u32,
        );
        inventory.add_executor_chip(keccak);

        Ok(())
    }
}

impl<F: PrimeField32> VmExecutionExtension<F> for Keccak256Gpu {
    type Executor = Keccak256Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Keccak256Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        self.0.extend_execution(inventory)
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Keccak256Gpu {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        self.0.extend_circuit(inventory)
    }
}
