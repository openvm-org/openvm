use std::array;

use derive_more::derive::From;
use num_bigint::BigUint;
use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
use openvm_circuit::{
    self,
    arch::{
        ExecutionBridge, SystemPort, VmExtension, VmInventory, VmInventoryBuilder, VmInventoryError,
    },
    system::phantom::PhantomChip,
};
use openvm_circuit_derive::{AnyEnum, InsExecutorE1, InstructionExecutor};
use openvm_circuit_primitives::{
    bigint::utils::big_uint_to_limbs,
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::{LocalOpcode, VmOpcode};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_rv32_adapters::{Rv32IsEqualModAdapterAir, Rv32IsEqualModeAdapterStep};
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use strum::EnumCount;

use crate::modular_chip::{
    ModularAddSubChip, ModularIsEqualAir, ModularIsEqualChip, ModularIsEqualCoreAir,
    ModularIsEqualStep, ModularMulDivChip,
};

// TODO: this should be decided after e2 execution
const MAX_INS_CAPACITY: usize = 1 << 22;

#[serde_as]
#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct ModularExtension {
    #[serde_as(as = "Vec<DisplayFromStr>")]
    pub supported_modulus: Vec<BigUint>,
}

#[derive(ChipUsageGetter, Chip, InstructionExecutor, AnyEnum, From, InsExecutorE1)]
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

        let execution_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let range_checker = builder.system_base().range_checker_chip.clone();
        let pointer_max_bits = builder.system_config().memory_config.pointer_max_bits;

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

            let modulus_limbs = big_uint_to_limbs(&modulus, 8);

            if bytes <= 32 {
                let addsub_chip = ModularAddSubChip::new(
                    execution_bridge.clone(),
                    memory_bridge.clone(),
                    builder.system_base().memory_controller.helper(),
                    pointer_max_bits,
                    config32.clone(),
                    start_offset,
                    bitwise_lu_chip.clone(),
                    range_checker.clone(),
                    MAX_INS_CAPACITY,
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularAddSubRv32_32(addsub_chip),
                    addsub_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
                let muldiv_chip = ModularMulDivChip::new(
                    execution_bridge.clone(),
                    memory_bridge.clone(),
                    builder.system_base().memory_controller.helper(),
                    pointer_max_bits,
                    config32.clone(),
                    start_offset,
                    bitwise_lu_chip.clone(),
                    range_checker.clone(),
                    MAX_INS_CAPACITY,
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularMulDivRv32_32(muldiv_chip),
                    muldiv_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let modulus_limbs = array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u8
                    } else {
                        0
                    }
                });
                let isequal_chip = ModularIsEqualChip::new(
                    ModularIsEqualAir::new(
                        Rv32IsEqualModAdapterAir::new(
                            execution_bridge.clone(),
                            memory_bridge.clone(),
                            bitwise_lu_chip.bus(),
                            pointer_max_bits,
                        ),
                        ModularIsEqualCoreAir::new(
                            modulus.clone(),
                            bitwise_lu_chip.bus(),
                            start_offset,
                        ),
                    ),
                    ModularIsEqualStep::new(
                        Rv32IsEqualModeAdapterStep::new(pointer_max_bits, bitwise_lu_chip.clone()),
                        modulus_limbs,
                        start_offset,
                        bitwise_lu_chip.clone(),
                    ),
                    MAX_INS_CAPACITY,
                    builder.system_base().memory_controller.helper(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularIsEqualRv32_32(isequal_chip),
                    iseq_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else if bytes <= 48 {
                let addsub_chip = ModularAddSubChip::new(
                    execution_bridge.clone(),
                    memory_bridge.clone(),
                    builder.system_base().memory_controller.helper(),
                    pointer_max_bits,
                    config48.clone(),
                    start_offset,
                    bitwise_lu_chip.clone(),
                    range_checker.clone(),
                    MAX_INS_CAPACITY,
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularAddSubRv32_48(addsub_chip),
                    addsub_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
                let muldiv_chip = ModularMulDivChip::new(
                    execution_bridge.clone(),
                    memory_bridge.clone(),
                    builder.system_base().memory_controller.helper(),
                    pointer_max_bits,
                    config48.clone(),
                    start_offset,
                    bitwise_lu_chip.clone(),
                    range_checker.clone(),
                    MAX_INS_CAPACITY,
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularMulDivRv32_48(muldiv_chip),
                    muldiv_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
                let modulus_limbs = array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u8
                    } else {
                        0
                    }
                });
                let isequal_chip = ModularIsEqualChip::new(
                    ModularIsEqualAir::new(
                        Rv32IsEqualModAdapterAir::new(
                            execution_bridge.clone(),
                            memory_bridge.clone(),
                            bitwise_lu_chip.bus(),
                            pointer_max_bits,
                        ),
                        ModularIsEqualCoreAir::new(
                            modulus.clone(),
                            bitwise_lu_chip.bus(),
                            start_offset,
                        ),
                    ),
                    ModularIsEqualStep::new(
                        Rv32IsEqualModeAdapterStep::new(pointer_max_bits, bitwise_lu_chip.clone()),
                        modulus_limbs,
                        start_offset,
                        bitwise_lu_chip.clone(),
                    ),
                    MAX_INS_CAPACITY,
                    builder.system_base().memory_controller.helper(),
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
