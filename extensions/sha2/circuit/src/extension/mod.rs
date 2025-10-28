use std::{result::Result, sync::Arc};

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError,
        ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena, VmCircuitExtension,
        VmExecutionExtension, VmProverExtension,
    },
    system::memory::SharedMemoryHelper,
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::*;
use openvm_sha2_transpiler::Rv32Sha2Opcode;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_stark_sdk::engine::StarkEngine;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::*;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use self::cuda::*;
        pub use self::cuda::Sha256GpuProverExt as Sha256ProverExt; // TODO: gpu
        pub use self::cuda::Sha256Rv32GpuBuilder as Sha256Rv32Builder;
    } else {
        pub use self::Sha2CpuProverExt as Sha2ProverExt;
        pub use self::Sha256Rv32CpuBuilder as Sha256Rv32Builder;
    }
}

// =================================== VM Extension Implementation =================================
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Sha2;

#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum Sha2Executor {
    Sha256(Sha2VmExecutor<Sha256Config>),
    Sha512(Sha2VmExecutor<Sha512Config>),
    Sha384(Sha2VmExecutor<Sha384Config>),
}

impl<F> VmExecutionExtension<F> for Sha2 {
    type Executor = Sha2Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Sha2Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();

        let sha256_step =
            Sha2VmExecutor::<Sha256Config>::new(Rv32Sha2Opcode::CLASS_OFFSET, pointer_max_bits);
        inventory.add_executor(sha256_step, [Rv32Sha2Opcode::SHA256.global_opcode()])?;

        let sha512_step =
            Sha2VmExecutor::<Sha512Config>::new(Rv32Sha2Opcode::CLASS_OFFSET, pointer_max_bits);
        inventory.add_executor(sha512_step, [Rv32Sha2Opcode::SHA512.global_opcode()])?;

        let sha384_step =
            Sha2VmExecutor::<Sha384Config>::new(Rv32Sha2Opcode::CLASS_OFFSET, pointer_max_bits);
        inventory.add_executor(sha384_step, [Rv32Sha2Opcode::SHA384.global_opcode()])?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Sha2 {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let pointer_max_bits = inventory.pointer_max_bits();

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

        let sha256_main_air = Sha2MainAir::<Sha256Config>::new(
            exec_bridge,
            memory_bridge,
            bitwise_lu,
            pointer_max_bits,
            Rv32Sha2Opcode::CLASS_OFFSET,
        );
        inventory.add_air(sha256_main_air);

        let sha256_block_hasher_air = Sha2BlockHasherAir::<Sha256Config>::new();
        inventory.add_air(sha256_block_hasher_air);

        Ok(())
    }
}

pub struct Sha2CpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Sha2> for Sha2CpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        _: &Sha2,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let pointer_max_bits = inventory.airs().pointer_max_bits();

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

        inventory.next_air::<Sha2MainAir<Sha256Config>>()?;
        let sha256 = Sha2MainChip::<Sha256Config>::new(
            Sha2MainFiller::new(bitwise_lu, pointer_max_bits),
            mem_helper,
        );
        inventory.add_executor_chip(sha256);

        Ok(())
    }
}
