use std::result::Result;

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, InitFileGenerator, MatrixRecordArena,
        RowMajorMatrixArena, SystemConfig, VmBuilder, VmChipComplex, VmCircuitExtension,
        VmExecutionExtension, VmProverExtension,
    },
    system::{SystemChipInventory, SystemCpuBuilder, SystemExecutor, SystemPort, memory::SharedMemoryHelper},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor, VmConfig};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::LocalOpcode;
use openvm_new_keccak256_transpiler::{XorinOpcode, KeccakfOpcode};
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32ImCpuProverExt, Rv32Io, Rv32IoExecutor, Rv32M, Rv32MExecutor,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_stark_sdk::engine::StarkEngine;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{keccakf::{KeccakfVmChip, KeccakfVmExecutor, KeccakfVmFiller, wrapper::air::KeccakfWrapperAir}, xorin::{XorinVmExecutor, XorinVmFiller, air::XorinVmAir}};
use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupChip;
use std::sync::Arc;
use crate::xorin::XorinVmChip;
use crate::keccakf::air::KeccakfVmAir;
use openvm_stark_backend::interaction::PermutationCheckBus;
use crate::keccakf::wrapper::KeccakfWrapperChip;
use crate::keccakf::wrapper::KeccakfWrapperFiller;

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Keccak256Rv32Config {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub keccak: Keccak256,
}

impl Default for Keccak256Rv32Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            rv32i: Rv32I,
            rv32m: Rv32M::default(),
            io: Rv32Io,
            keccak: Keccak256,
        }
    }
}

// Default implementation uses no init file
impl InitFileGenerator for Keccak256Rv32Config {}

#[derive(Clone)]
pub struct Keccak256Rv32CpuBuilder;

impl<E, SC> VmBuilder<E> for Keccak256Rv32CpuBuilder
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: PrimeField32,
{
    type VmConfig = Keccak256Rv32Config;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &Keccak256Rv32Config,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemCpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.rv32i, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.rv32m, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &Keccak256CpuProverExt,
            &config.keccak,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

// =================================== VM Extension Implementation =================================
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Keccak256;



#[derive(Clone, Copy, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]

pub enum Keccak256Executor {
    Keccakf(KeccakfVmExecutor),
    Xorin(XorinVmExecutor),
}

impl<F> VmExecutionExtension<F> for Keccak256 {
    type Executor = Keccak256Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Keccak256Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();
        
        let xorin_executor =
            XorinVmExecutor::new(XorinOpcode::CLASS_OFFSET, pointer_max_bits);
        inventory.add_executor(
            xorin_executor,
            XorinOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let keccak_executor = KeccakfVmExecutor::new(KeccakfOpcode::CLASS_OFFSET, pointer_max_bits);
        inventory.add_executor(
            keccak_executor,
            KeccakfOpcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Keccak256 {
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

        let xorin_air = XorinVmAir::new(
            exec_bridge,
            memory_bridge,
            bitwise_lu,
            pointer_max_bits,
            XorinOpcode::CLASS_OFFSET,
        );
        inventory.add_air(xorin_air);

        let keccak_bus = PermutationCheckBus::new(inventory.new_bus_idx());

        let keccak_air = KeccakfVmAir::new(
            exec_bridge,
            memory_bridge,
            bitwise_lu,
            pointer_max_bits,
            KeccakfOpcode::CLASS_OFFSET,
            keccak_bus,
        );
        inventory.add_air(keccak_air);

        let keccak_wrapper_air = KeccakfWrapperAir::new(
            keccak_bus,
        );
        inventory.add_air(keccak_wrapper_air);

        Ok(())
    }
}

pub struct Keccak256CpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Keccak256> for Keccak256CpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        _: &Keccak256,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let pointer_max_bits = inventory.airs().pointer_max_bits();

        let bitwise_lu = {
            let existing_chip = inventory.find_chip::<SharedBitwiseOperationLookupChip<8>>()
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

        inventory.next_air::<XorinVmAir>()?;
        let xorin_chip = XorinVmChip::new(
            XorinVmFiller::new(bitwise_lu.clone(), pointer_max_bits),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(xorin_chip);

        inventory.next_air::<KeccakfVmAir>()?;
        let keccak_chip = KeccakfVmChip::new(
            KeccakfVmFiller::new(bitwise_lu, pointer_max_bits),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(keccak_chip);

        inventory.next_air::<KeccakfWrapperAir>()?;
        let keccak_wrapper_chip = KeccakfWrapperChip::new(KeccakfWrapperFiller::new(), mem_helper);
        inventory.add_executor_chip(keccak_wrapper_chip);

        Ok(())
    }
}
