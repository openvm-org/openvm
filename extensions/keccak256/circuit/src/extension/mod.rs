use std::{
    result::Result,
    sync::{Arc, Mutex},
};

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        to_byte_ptr_bits, AirInventory, AirInventoryError, ChipInventory, ChipInventoryError,
        ExecutionBridge, ExecutorInventoryBuilder, ExecutorInventoryError, InitFileGenerator,
        SystemConfig, VmBuilder, VmChipComplex, VmCircuitExtension, VmExecutionExtension, VmField,
        VmProverExtension,
    },
    system::{
        memory::SharedMemoryHelper, SystemChipInventory, SystemCpuBuilder, SystemExecutor,
        SystemPort,
    },
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, VmConfig};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    Chip,
};
use openvm_cpu_backend::{CpuBackend, CpuDevice};
use openvm_instructions::*;
use openvm_keccak256_transpiler::{KeccakfOpcode, XorinOpcode};
use openvm_riscv_circuit::{
    RiscvI, RiscvIExecutor, RiscvImCpuProverExt, RiscvIo, RiscvIoExecutor, RiscvM, RiscvMExecutor,
};
use openvm_stark_backend::{
    interaction::PermutationCheckBus, p3_field::PrimeField32, prover::AirProvingContext,
    StarkEngine, StarkProtocolConfig, Val,
};
#[cfg(feature = "rvr")]
use rvr_openvm_ext_keccak::KeccakExtension;
#[cfg(feature = "rvr")]
use rvr_openvm_lift::{RvrExtensionCtx, RvrExtensions, VmRvrExtension};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{
    keccakf_op::{KeccakfExecutor, KeccakfOpAir, KeccakfOpChip},
    keccakf_perm::{KeccakfPermAir, KeccakfPermChip},
    xorin::{air::XorinVmAir, XorinVmChip, XorinVmExecutor, XorinVmFiller},
};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(all(test, feature = "rvr"))]
mod rvr_tests;

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Keccak256Config {
    #[config(executor = "SystemExecutor")]
    pub system: SystemConfig,
    #[extension]
    pub riscv_i: RiscvI,
    #[extension]
    pub riscv_m: RiscvM,
    #[extension]
    pub io: RiscvIo,
    #[extension]
    pub keccak: Keccak256,
}

impl Default for Keccak256Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            riscv_i: RiscvI,
            riscv_m: RiscvM::default(),
            io: RiscvIo,
            keccak: Keccak256,
        }
    }
}

// Default implementation uses no init file
impl InitFileGenerator for Keccak256Config {}

#[derive(Clone)]
pub struct Keccak256CpuBuilder;

impl<SC, E> VmBuilder<E> for Keccak256CpuBuilder
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    type VmConfig = Keccak256Config;
    type SystemChipInventory = SystemChipInventory<SC>;

    fn create_chip_complex(
        &self,
        config: &Keccak256Config,
        circuit: AirInventory<SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<VmChipComplex<SC, E::PB, Self::SystemChipInventory>, ChipInventoryError> {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &SystemCpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _>::extend_prover(&RiscvImCpuProverExt, &config.riscv_i, inventory)?;
        VmProverExtension::<E, _>::extend_prover(&RiscvImCpuProverExt, &config.riscv_m, inventory)?;
        VmProverExtension::<E, _>::extend_prover(&RiscvImCpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _>::extend_prover(
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

#[cfg(feature = "rvr")]
impl<F: PrimeField32> VmRvrExtension<F> for Keccak256 {
    fn extend_rvr(&self, extensions: &mut RvrExtensions, ctx: Option<&RvrExtensionCtx>) {
        let ext = KeccakExtension::new(ctx).expect("failed to construct rvr KeccakExtension");
        extensions.register_lifter(ext);
    }
}

#[derive(Clone, Copy, From, AnyEnum, Executor, MeteredExecutor)]
pub enum Keccak256Executor {
    Keccakf(KeccakfExecutor),
    Xorin(XorinVmExecutor),
}

impl VmExecutionExtension for Keccak256 {
    type Executor = Keccak256Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<Keccak256Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.pointer_max_bits());

        let xorin_executor = XorinVmExecutor::new(XorinOpcode::CLASS_OFFSET, byte_ptr_max_bits);
        inventory.add_executor(
            xorin_executor,
            XorinOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let keccak_executor = KeccakfExecutor::new(KeccakfOpcode::CLASS_OFFSET, byte_ptr_max_bits);
        inventory.add_executor(
            keccak_executor,
            KeccakfOpcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(())
    }
}

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for Keccak256 {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.pointer_max_bits());
        let range_checker = inventory.range_checker().bus;

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
            range_checker,
            byte_ptr_max_bits,
            XorinOpcode::CLASS_OFFSET,
        );
        inventory.add_air(xorin_air);

        let keccakf_state_bus = PermutationCheckBus::new(inventory.new_bus_idx());
        let periphery_air = KeccakfPermAir::new(keccakf_state_bus);
        inventory.add_air(periphery_air);

        let op_air = KeccakfOpAir::new(
            exec_bridge,
            memory_bridge,
            keccakf_state_bus,
            range_checker,
            byte_ptr_max_bits,
            KeccakfOpcode::CLASS_OFFSET,
        );
        inventory.add_air(op_air);

        Ok(())
    }
}

pub struct Keccak256CpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<SC, E> VmProverExtension<E, Keccak256> for Keccak256CpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: PrimeField32,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        _: &Keccak256,
        inventory: &mut ChipInventory<SC, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());

        let bitwise_lu = {
            let existing_chip = inventory
                .find_chip::<SharedBitwiseOperationLookupChip<8>>()
                .next();

            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &BitwiseOperationLookupAir<8> = inventory.next_air()?;
                let chip = Arc::new(BitwiseOperationLookupChip::new(air.bus));
                inventory.add_periphery_chip_with_tracegen(chip.clone(), |chip, _| {
                    Ok(chip.generate_proving_ctx())
                });
                chip
            }
        };

        inventory.next_air::<XorinVmAir>()?;
        let xorin_chip = XorinVmChip::new(
            XorinVmFiller::new(bitwise_lu.clone(), range_checker.clone(), byte_ptr_max_bits),
            mem_helper.clone(),
        );
        inventory.add_executor_chip_with_tracegen(xorin_chip, |chip, postflight| {
            crate::xorin::trace::generate_trace_from_postflight(chip, postflight)
                .map(AirProvingContext::simple_no_pis)
        });

        inventory.next_air::<KeccakfPermAir>()?;
        let shared_preimages = Arc::new(Mutex::new(Vec::new()));
        let periphery_chip = KeccakfPermChip::new(shared_preimages.clone());
        // Trace generators run in reverse insertion order. Register the permutation first so the
        // operation generator publishes its preimages before they are consumed here.
        inventory.add_periphery_chip_with_tracegen(periphery_chip, |chip, postflight| {
            crate::keccakf_perm::generate_trace_from_postflight(chip, postflight)
                .map(AirProvingContext::simple_no_pis)
        });

        inventory.next_air::<KeccakfOpAir>()?;
        let op_chip = KeccakfOpChip::new(
            range_checker.clone(),
            byte_ptr_max_bits,
            mem_helper.clone(),
            shared_preimages,
        );
        inventory.add_executor_chip_with_tracegen(op_chip, |chip, postflight| {
            crate::keccakf_op::generate_trace_from_postflight(chip, postflight)
                .map(AirProvingContext::simple_no_pis)
        });

        Ok(())
    }
}
