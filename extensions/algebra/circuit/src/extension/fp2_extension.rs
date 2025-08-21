use std::sync::Arc;

use num_bigint::BigUint;
use openvm_algebra_transpiler::Fp2Opcode;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena, VmCircuitExtension,
        VmExecutionExtension, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    var_range::VariableRangeCheckerBus,
};
use openvm_instructions::{LocalOpcode, VmOpcode};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_stark_sdk::engine::StarkEngine;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use strum::EnumCount;

use crate::{
    fp2_chip::{
        get_fp2_addsub_air, get_fp2_addsub_chip, get_fp2_addsub_step, get_fp2_muldiv_air,
        get_fp2_muldiv_chip, get_fp2_muldiv_step, Fp2Air, Fp2Executor,
    },
    AlgebraCpuProverExt, ModularExtension,
};

#[serde_as]
#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct Fp2Extension {
    // (name, modulus)
    // name must match the struct name defined by complex_declare
    #[serde_as(as = "Vec<(_, DisplayFromStr)>")]
    pub supported_moduli: Vec<(String, BigUint)>,
}

impl Fp2Extension {
    pub fn generate_complex_init(&self, modular_config: &ModularExtension) -> String {
        fn get_index_of_modulus(modulus: &BigUint, modular_config: &ModularExtension) -> usize {
            modular_config
                .supported_moduli
                .iter()
                .position(|m| m == modulus)
                .expect("Modulus used in Fp2Extension not found in ModularExtension")
        }

        let supported_moduli = self
            .supported_moduli
            .iter()
            .map(|(name, modulus)| {
                format!(
                    "\"{}\" {{ mod_idx = {} }}",
                    name,
                    get_index_of_modulus(modulus, modular_config)
                )
            })
            .collect::<Vec<String>>()
            .join(", ");

        format!("openvm_algebra_guest::complex_macros::complex_init! {{ {supported_moduli} }}")
    }
}

#[derive(Clone, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum Fp2ExtensionExecutor {
    // 32 limbs prime
    Fp2AddSubRv32_32(Fp2Executor<2, 32>), // Fp2AddSub
    Fp2MulDivRv32_32(Fp2Executor<2, 32>), // Fp2MulDiv
    // 48 limbs prime
    Fp2AddSubRv32_48(Fp2Executor<6, 16>), // Fp2AddSub
    Fp2MulDivRv32_48(Fp2Executor<6, 16>), // Fp2MulDiv
}

impl<F: PrimeField32> VmExecutionExtension<F> for Fp2Extension {
    type Executor = Fp2ExtensionExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Fp2ExtensionExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();
        // TODO: somehow get the range checker bus from `ExecutorInventory`
        let dummy_range_checker_bus = VariableRangeCheckerBus::new(u16::MAX, 16);
        for (i, (_, modulus)) in self.supported_moduli.iter().enumerate() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8);
            let start_offset = Fp2Opcode::CLASS_OFFSET + i * Fp2Opcode::COUNT;

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };
                let addsub = get_fp2_addsub_step(
                    config.clone(),
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    Fp2ExtensionExecutor::Fp2AddSubRv32_32(addsub),
                    ((Fp2Opcode::ADD as usize)..=(Fp2Opcode::SETUP_ADDSUB as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let muldiv = get_fp2_muldiv_step(
                    config,
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    Fp2ExtensionExecutor::Fp2MulDivRv32_32(muldiv),
                    ((Fp2Opcode::MUL as usize)..=(Fp2Opcode::SETUP_MULDIV as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };
                let addsub = get_fp2_addsub_step(
                    config.clone(),
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    Fp2ExtensionExecutor::Fp2AddSubRv32_48(addsub),
                    ((Fp2Opcode::ADD as usize)..=(Fp2Opcode::SETUP_ADDSUB as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let muldiv = get_fp2_muldiv_step(
                    config,
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    Fp2ExtensionExecutor::Fp2MulDivRv32_48(muldiv),
                    ((Fp2Opcode::MUL as usize)..=(Fp2Opcode::SETUP_MULDIV as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else {
                panic!("Modulus too large");
            }
        }
        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Fp2Extension {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let range_checker_bus = inventory.range_checker().bus;
        let pointer_max_bits = inventory.pointer_max_bits();

        let bitwise_lu = {
            // A trick to get around Rust's borrow rules
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
        for (i, (_, modulus)) in self.supported_moduli.iter().enumerate() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8);
            let start_offset = Fp2Opcode::CLASS_OFFSET + i * Fp2Opcode::COUNT;

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                let addsub = get_fp2_addsub_air::<2, 32>(
                    exec_bridge,
                    memory_bridge,
                    config.clone(),
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                );
                inventory.add_air(addsub);

                let muldiv = get_fp2_muldiv_air::<2, 32>(
                    exec_bridge,
                    memory_bridge,
                    config,
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                );
                inventory.add_air(muldiv);
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };

                let addsub = get_fp2_addsub_air::<6, 16>(
                    exec_bridge,
                    memory_bridge,
                    config.clone(),
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                );
                inventory.add_air(addsub);

                let muldiv = get_fp2_muldiv_air::<6, 16>(
                    exec_bridge,
                    memory_bridge,
                    config,
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                );
                inventory.add_air(muldiv);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Fp2Extension> for AlgebraCpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        extension: &Fp2Extension,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
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
        for (_, modulus) in extension.supported_moduli.iter() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8);

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                inventory.next_air::<Fp2Air<2, 32>>()?;
                let addsub = get_fp2_addsub_chip::<Val<SC>, 2, 32>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(addsub);

                inventory.next_air::<Fp2Air<2, 32>>()?;
                let muldiv = get_fp2_muldiv_chip::<Val<SC>, 2, 32>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(muldiv);
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };

                inventory.next_air::<Fp2Air<6, 16>>()?;
                let addsub = get_fp2_addsub_chip::<Val<SC>, 6, 16>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(addsub);

                inventory.next_air::<Fp2Air<6, 16>>()?;
                let muldiv = get_fp2_muldiv_chip::<Val<SC>, 6, 16>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(muldiv);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}
