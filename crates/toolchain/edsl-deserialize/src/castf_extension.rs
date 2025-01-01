use derive_more::From;
use openvm_circuit::{
    arch::{SystemPort, VmExtension, VmInventory, VmInventoryBuilder, VmInventoryError},
    circuit_derive::{Chip, ChipUsageGetter},
};
use openvm_circuit_derive::{AnyEnum, InstructionExecutor};
use openvm_instructions::{UsizeOpcode, VmOpcode};
use openvm_native_circuit::{
    adapters::convert_adapter::ConvertAdapterChip, CastFChip, CastFCoreChip,
};
use openvm_native_compiler::CastfOpcode;
use p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct CastF;

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum CastFExecutor<F: PrimeField32> {
    CastF(CastFChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum CastFPeriphery<F: PrimeField32> {
    Placeholder(CastFChip<F>),
}

impl<F: PrimeField32> VmExtension<F> for CastF {
    type Executor = CastFExecutor<F>;
    type Periphery = CastFPeriphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let SystemPort {
            execution_bus,
            program_bus,
            memory_controller,
        } = builder.system_port();
        let range_checker = builder.system_base().range_checker_chip.clone();

        let castf_chip = CastFChip::new(
            ConvertAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
            CastFCoreChip::new(range_checker.clone(), CastfOpcode::default_offset()),
            memory_controller.clone(),
        );
        inventory.add_executor(
            castf_chip,
            [VmOpcode::with_default_offset(CastfOpcode::CASTF)],
        )?;

        Ok(inventory)
    }
}
