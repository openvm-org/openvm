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
use openvm_sha256_transpiler::Rv32Sha256Opcode;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_stark_sdk::engine::StarkEngine;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

#[cfg(feature = "cuda")]
use {
    openvm_circuit::{
        arch::DenseRecordArena,
        system::cuda::{
            extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
            SystemChipInventoryGPU,
        },
    },
    openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend},
    openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config,
};

use crate::*;

// =================================== VM Extension Implementation =================================
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Sha256;

#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum Sha256Executor {
    Sha256(Sha256VmExecutor),
}

impl<F> VmExecutionExtension<F> for Sha256 {
    type Executor = Sha256Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Sha256Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();
        let sha256_step = Sha256VmExecutor::new(Rv32Sha256Opcode::CLASS_OFFSET, pointer_max_bits);
        inventory.add_executor(
            sha256_step,
            Rv32Sha256Opcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Sha256 {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
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

pub struct Sha2CpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Sha256> for Sha2CpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        _: &Sha256,
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

        inventory.next_air::<Sha256VmAir>()?;
        let sha256 = Sha256VmChip::new(
            Sha256VmFiller::new(bitwise_lu, pointer_max_bits),
            mem_helper,
        );
        inventory.add_executor_chip(sha256);

        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub struct Sha256GpuProverExt;

#[cfg(feature = "cuda")]
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Sha256>
    for Sha256GpuProverExt
{
    fn extend_prover(
        &self,
        _: &Sha256,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Sha256VmAir>()?;
        let sha256 = Sha256VmChipGpu::new(
            range_checker.clone(),
            bitwise_lu,
            pointer_max_bits as u32,
            timestamp_max_bits as u32,
        );
        inventory.add_executor_chip(sha256);

        Ok(())
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct Sha256Rv32GpuBuilder;

#[cfg(feature = "cuda")]
impl VmBuilder<GpuBabyBearPoseidon2Engine> for Sha256Rv32GpuBuilder {
    type VmConfig = Sha256Rv32Config;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Sha256Rv32Config,
        circuit: AirInventory<BabyBearPoseidon2Config>,
    ) -> Result<
        VmChipComplex<
            BabyBearPoseidon2Config,
            Self::RecordArena,
            GpuBackend,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        use openvm_circuit::system::cuda::extensions::SystemGpuBuilder;

        let mut chip_complex = VmBuilder::<GpuBabyBearPoseidon2Engine>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
        )?;
        let inventory = &mut chip_complex.inventory;
        // VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
        //     &Rv32ImGpuProverExt,
        //     &config.rv32i,
        //     inventory,
        // )?;
        // VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
        //     &Rv32ImGpuProverExt,
        //     &config.rv32m,
        //     inventory,
        // )?;
        // VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
        //     &Rv32ImGpuProverExt,
        //     &config.io,
        //     inventory,
        // )?;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &Sha256GpuProverExt,
            &config.sha256,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
