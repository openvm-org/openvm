use derive_more::derive::From;
use num_bigint::BigUint;
use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
use openvm_circuit::{
    self,
    arch::{SystemPort, VmExtension, VmInventory, VmInventoryBuilder, VmInventoryError},
    system::phantom::PhantomChip,
};
use openvm_circuit_derive::{AnyEnum, InstructionExecutor};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::{LocalOpcode, VmOpcode};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_rv32_adapters::{Rv32IsEqualModAdapterChip, Rv32VecHeapAdapterChip};
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use strum::EnumCount;

use crate::modular_chip::{
    ModularAddSubChip, ModularIsEqualChip, ModularIsEqualCoreChip, ModularMulDivChip,
};

#[serde_as]
#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct ModularExtension {
    #[serde_as(as = "Vec<DisplayFromStr>")]
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
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
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
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = builder.system_port();
        let range_checker = builder.system_base().range_checker_chip.clone();
        let bitwise_lu_chip = if let Some(&chip) = builder
            .find_chip::<SharedBitwiseOperationLookupChip<8>>()
            .first()
        {
            chip.clone()
        } else {
            let bitwise_lu_bus = BitwiseOperationLookupBus::new(builder.new_bus_idx());
            let chip = SharedBitwiseOperationLookupChip::new(bitwise_lu_bus);
            inventory.add_periphery_chip(chip.clone());
            chip
        };
        let offline_memory = builder.system_base().offline_memory();
        let address_bits = builder.system_config().memory_config.pointer_max_bits;

        let addsub_opcodes = (Rv32ModularArithmeticOpcode::ADD as usize)
            ..=(Rv32ModularArithmeticOpcode::SETUP_ADDSUB as usize);
        let muldiv_opcodes = (Rv32ModularArithmeticOpcode::MUL as usize)
            ..=(Rv32ModularArithmeticOpcode::SETUP_MULDIV as usize);
        let iseq_opcodes = (Rv32ModularArithmeticOpcode::IS_EQ as usize)
            ..=(Rv32ModularArithmeticOpcode::SETUP_ISEQ as usize);

        for (i, modulus) in self.supported_modulus.iter().enumerate() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8);
            let start_offset =
                Rv32ModularArithmeticOpcode::CLASS_OFFSET + i * Rv32ModularArithmeticOpcode::COUNT;

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
                memory_bridge,
                address_bits,
                bitwise_lu_chip.clone(),
            );
            let adapter_chip_48 = Rv32VecHeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_bridge,
                address_bits,
                bitwise_lu_chip.clone(),
            );

            if bytes <= 32 {
                let addsub_chip = ModularAddSubChip::new(
                    adapter_chip_32.clone(),
                    config32.clone(),
                    start_offset,
                    range_checker.clone(),
                    offline_memory.clone(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularAddSubRv32_32(addsub_chip),
                    addsub_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
                let muldiv_chip = ModularMulDivChip::new(
                    adapter_chip_32.clone(),
                    config32.clone(),
                    start_offset,
                    range_checker.clone(),
                    offline_memory.clone(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularMulDivRv32_32(muldiv_chip),
                    muldiv_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
                let isequal_chip = ModularIsEqualChip::new(
                    Rv32IsEqualModAdapterChip::new(
                        execution_bus,
                        program_bus,
                        memory_bridge,
                        address_bits,
                        bitwise_lu_chip.clone(),
                    ),
                    ModularIsEqualCoreChip::new(
                        modulus.clone(),
                        bitwise_lu_chip.clone(),
                        start_offset,
                    ),
                    offline_memory.clone(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularIsEqualRv32_32(isequal_chip),
                    iseq_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else if bytes <= 48 {
                let addsub_chip = ModularAddSubChip::new(
                    adapter_chip_48.clone(),
                    config48.clone(),
                    start_offset,
                    range_checker.clone(),
                    offline_memory.clone(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularAddSubRv32_48(addsub_chip),
                    addsub_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
                let muldiv_chip = ModularMulDivChip::new(
                    adapter_chip_48.clone(),
                    config48.clone(),
                    start_offset,
                    range_checker.clone(),
                    offline_memory.clone(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularMulDivRv32_48(muldiv_chip),
                    muldiv_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
                let isequal_chip = ModularIsEqualChip::new(
                    Rv32IsEqualModAdapterChip::new(
                        execution_bus,
                        program_bus,
                        memory_bridge,
                        address_bits,
                        bitwise_lu_chip.clone(),
                    ),
                    ModularIsEqualCoreChip::new(
                        modulus.clone(),
                        bitwise_lu_chip.clone(),
                        start_offset,
                    ),
                    offline_memory.clone(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularIsEqualRv32_48(isequal_chip),
                    iseq_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(inventory)
    }
}
