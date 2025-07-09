use std::sync::Arc;

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError,
        ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena, VmCircuitExtension,
        VmExecutionExtension, VmProverExtension,
    },
    system::memory::SharedMemoryHelper,
};
use openvm_circuit_derive::AnyEnum;
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::*;
use openvm_sha256_transpiler::Rv32Sha256Opcode;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::cpu::CpuBackend,
};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::*;

// TODO: this should be decided after e2 execution

// #[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
// pub struct Sha256Rv32Config {
//     #[system]
//     pub system: SystemConfig,
//     #[extension]
//     pub rv32i: Rv32I,
//     #[extension]
//     pub rv32m: Rv32M,
//     #[extension]
//     pub io: Rv32Io,
//     #[extension]
//     pub sha256: Sha256,
// }

// impl Default for Sha256Rv32Config {
//     fn default() -> Self {
//         Self {
//             system: SystemConfig::default().with_continuations(),
//             rv32i: Rv32I,
//             rv32m: Rv32M::default(),
//             io: Rv32Io,
//             sha256: Sha256,
//         }
//     }
// }

// Default implementation uses no init file
// impl InitFileGenerator for Sha256Rv32Config {}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Sha256;

#[derive(From, AnyEnum)]
pub enum Sha256Executor {
    Sha256(Sha256VmStep),
}

// #[derive(From, ChipUsageGetter, Chip, AnyEnum)]
// pub enum Sha256Periphery<F: PrimeField32> {
//     BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
//     Phantom(PhantomExecutor<F>),
// }

impl<F> VmExecutionExtension<F> for Sha256 {
    type Executor = Sha256Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Sha256Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        // TODO: getter for pointer_max_bits
        let pointer_max_bits = 29;
        let sha256_step = Sha256VmStep::new(Rv32Sha256Opcode::CLASS_OFFSET, pointer_max_bits);
        inventory.add_executor(
            sha256_step,
            Rv32Sha256Opcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Sha256 {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        // TODO: add a getter for pointer_max_bits
        let pointer_max_bits = inventory.config().memory_config.pointer_max_bits;

        let bitwise_lu = {
            let existing_air = inventory.find_air::<BitwiseOperationLookupAir<8>>().next();
            if let Some(air) = existing_air {
                air.bus
            } else {
                let bus = BitwiseOperationLookupBus::new(inventory.new_bus_idx());
                let air = BitwiseOperationLookupAir::<8>::new(bus);
                inventory.add_air(air);
                air.bus
            }
        };

        let sha256 = Sha256VmAir::new(
            inventory.system().port(),
            bitwise_lu,
            pointer_max_bits,
            inventory.new_bus_idx(),
        );
        inventory.add_air(sha256);

        Ok(())
    }
}

// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<SC, RA> VmProverExtension<SC, RA, CpuBackend<SC>> for Sha256
where
    SC: StarkGenericConfig,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.airs().config().memory_config.clk_max_bits;
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        // TODO: add a getter for pointer_max_bits
        let pointer_max_bits = 29;

        let bitwise_lu = {
            let existing_chip = inventory
                .find_chip::<SharedBitwiseOperationLookupChip<8>>()
                .next();
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &BitwiseOperationLookupAir<8> = inventory.next_air()?;
                let chip = Arc::new(BitwiseOperationLookupChip::new(air.bus));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        inventory.next_air::<Sha256VmAir>()?;
        let sha256 = Sha256VmChip::new(
            Sha256VmFiller::new(bitwise_lu, pointer_max_bits),
            mem_helper,
        );
        inventory.add_executor_chip(sha256);

        Ok(())
    }
}
