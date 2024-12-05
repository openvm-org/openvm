use std::sync::Arc;

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, BitwiseOperationLookupChip,
};
use ax_mod_circuit_builder::ExprBuilderConfig;
use axvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
use axvm_circuit::{
    self,
    arch::{VmExtension, VmInventory, VmInventoryBuilder, VmInventoryError},
    system::phantom::PhantomChip,
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor};
use axvm_instructions::{AxVmOpcode, UsizeOpcode};
use axvm_rv32_adapters::{Rv32IsEqualModAdapterChip, Rv32VecHeapAdapterChip};
use derive_more::derive::From;
use num_bigint_dig::BigUint;
use p3_field::PrimeField32;
use strum::EnumCount;

use crate::modular_chip::{
    ModularAddSubChip, ModularAddSubCoreChip, ModularIsEqualChip, ModularIsEqualCoreChip,
    ModularMulDivChip, ModularMulDivCoreChip,
};

#[derive(Clone, Debug, derive_new::new)]
pub struct ModularExtension {
    pub supported_modulus: Vec<BigUint>,
}

#[derive(ChipUsageGetter, Chip, InstructionExecutor, AnyEnum, From)]
pub enum ModularExtensionExecutor<F: PrimeField32> {
    // 32 limbs prime
    ModularAddSubRv32_32(ModularAddSubChip<F, 1, 32>),
    ModularMulDivRv32_32(ModularMulDivChip<F, 1, 32>),
    ModularIsEqualRv32_32(ModularIsEqualChip<F, 1, 32, 32>),
    // 48 limbs prime
    ModularAddSubRv32_48(ModularAddSubChip<F, 3, 16>),
    ModularMulDivRv32_48(ModularMulDivChip<F, 3, 16>),
    ModularIsEqualRv32_48(ModularIsEqualChip<F, 3, 16, 48>),
}

#[derive(ChipUsageGetter, Chip, AnyEnum, From)]
pub enum ModularExtensionPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(Arc<BitwiseOperationLookupChip<8>>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

impl<F: PrimeField32> VmExtension<F> for ModularExtension {
    type Executor = ModularExtensionExecutor<F>;
    type Periphery = ModularExtensionPeriphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let execution_bus = builder.system_base().execution_bus();
        let program_bus = builder.system_base().program_bus();
        let memory_controller = builder.memory_controller().clone();
        let range_checker = builder.system_base().range_checker_chip.clone();
        let bitwise_lu_chip = if let Some(chip) = builder
            .find_chip::<Arc<BitwiseOperationLookupChip<8>>>()
            .first()
        {
            Arc::clone(chip)
        } else {
            let bitwise_lu_bus = BitwiseOperationLookupBus::new(builder.new_bus_idx());
            let chip = Arc::new(BitwiseOperationLookupChip::new(bitwise_lu_bus));
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let addsub_opcodes = (Rv32ModularArithmeticOpcode::ADD as usize)
            ..=(Rv32ModularArithmeticOpcode::SETUP_ADDSUB as usize);
        let muldiv_opcodes = (Rv32ModularArithmeticOpcode::MUL as usize)
            ..=(Rv32ModularArithmeticOpcode::SETUP_MULDIV as usize);
        let iseq_opcodes = (Rv32ModularArithmeticOpcode::IS_EQ as usize)
            ..=(Rv32ModularArithmeticOpcode::SETUP_ISEQ as usize);

        for (i, modulus) in self.supported_modulus.iter().enumerate() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8);
            let class_offset = Rv32ModularArithmeticOpcode::default_offset()
                + i * Rv32ModularArithmeticOpcode::COUNT;

            let config32 = ExprBuilderConfig {
                modulus: modulus.clone(),
                num_limbs: 32,
                limb_bits: 8,
            };
            let config48 = ExprBuilderConfig {
                modulus: modulus.clone(),
                num_limbs: 48,
                limb_bits: 8,
            };
            let adapter_chip_32 = Rv32VecHeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
                bitwise_lu_chip.clone(),
            );
            let adapter_chip_48 = Rv32VecHeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
                bitwise_lu_chip.clone(),
            );

            if bytes <= 32 {
                let addsub_chip = ModularAddSubChip::new(
                    adapter_chip_32.clone(),
                    ModularAddSubCoreChip::new(
                        config32.clone(),
                        range_checker.clone(),
                        class_offset,
                    ),
                    memory_controller.clone(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularAddSubRv32_32(addsub_chip),
                    addsub_opcodes
                        .clone()
                        .map(|x| AxVmOpcode::from_usize(x + class_offset)),
                )?;
                let muldiv_chip = ModularMulDivChip::new(
                    adapter_chip_32.clone(),
                    ModularMulDivCoreChip::new(
                        config32.clone(),
                        range_checker.clone(),
                        class_offset,
                    ),
                    memory_controller.clone(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularMulDivRv32_32(muldiv_chip),
                    muldiv_opcodes
                        .clone()
                        .map(|x| AxVmOpcode::from_usize(x + class_offset)),
                )?;
                let isequal_chip = ModularIsEqualChip::new(
                    Rv32IsEqualModAdapterChip::new(
                        execution_bus,
                        program_bus,
                        memory_controller.clone(),
                        bitwise_lu_chip.clone(),
                    ),
                    ModularIsEqualCoreChip::new(
                        modulus.clone(),
                        bitwise_lu_chip.clone(),
                        class_offset,
                    ),
                    memory_controller.clone(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularIsEqualRv32_32(isequal_chip),
                    iseq_opcodes
                        .clone()
                        .map(|x| AxVmOpcode::from_usize(x + class_offset)),
                )?;
            } else if bytes <= 48 {
                let addsub_chip = ModularAddSubChip::new(
                    adapter_chip_48.clone(),
                    ModularAddSubCoreChip::new(
                        config48.clone(),
                        range_checker.clone(),
                        class_offset,
                    ),
                    memory_controller.clone(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularAddSubRv32_48(addsub_chip),
                    addsub_opcodes
                        .clone()
                        .map(|x| AxVmOpcode::from_usize(x + class_offset)),
                )?;
                let muldiv_chip = ModularMulDivChip::new(
                    adapter_chip_48.clone(),
                    ModularMulDivCoreChip::new(
                        config48.clone(),
                        range_checker.clone(),
                        class_offset,
                    ),
                    memory_controller.clone(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularMulDivRv32_48(muldiv_chip),
                    muldiv_opcodes
                        .clone()
                        .map(|x| AxVmOpcode::from_usize(x + class_offset)),
                )?;
                let isequal_chip = ModularIsEqualChip::new(
                    Rv32IsEqualModAdapterChip::new(
                        execution_bus,
                        program_bus,
                        memory_controller.clone(),
                        bitwise_lu_chip.clone(),
                    ),
                    ModularIsEqualCoreChip::new(
                        modulus.clone(),
                        bitwise_lu_chip.clone(),
                        class_offset,
                    ),
                    memory_controller.clone(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularIsEqualRv32_48(isequal_chip),
                    iseq_opcodes
                        .clone()
                        .map(|x| AxVmOpcode::from_usize(x + class_offset)),
                )?;
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(inventory)
    }
}
