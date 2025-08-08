use std::sync::Arc;

use hex_literal::hex;
use lazy_static::lazy_static;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, Zero};
use once_cell::sync::Lazy;
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
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::{LocalOpcode, VmOpcode};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use strum::EnumCount;

use crate::{
    get_ec_addne_air, get_ec_addne_chip, get_ec_addne_step, get_ec_double_air, get_ec_double_chip,
    get_ec_double_step, EcAddNeExecutor, EcDoubleExecutor, EccCpuProverExt, WeierstrassAir,
};

#[serde_as]
#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct CurveConfig {
    /// The name of the curve struct as defined by moduli_declare.
    pub struct_name: String,
    /// The coordinate modulus of the curve.
    #[serde_as(as = "DisplayFromStr")]
    pub modulus: BigUint,
    /// The scalar field modulus of the curve.
    #[serde_as(as = "DisplayFromStr")]
    pub scalar: BigUint,
    /// The coefficient a of y^2 = x^3 + ax + b.
    #[serde_as(as = "DisplayFromStr")]
    pub a: BigUint,
    /// The coefficient b of y^2 = x^3 + ax + b.
    #[serde_as(as = "DisplayFromStr")]
    pub b: BigUint,
}

pub static SECP256K1_CONFIG: Lazy<CurveConfig> = Lazy::new(|| CurveConfig {
    struct_name: SECP256K1_ECC_STRUCT_NAME.to_string(),
    modulus: SECP256K1_MODULUS.clone(),
    scalar: SECP256K1_ORDER.clone(),
    a: BigUint::zero(),
    b: BigUint::from_u8(7u8).unwrap(),
});

pub static P256_CONFIG: Lazy<CurveConfig> = Lazy::new(|| CurveConfig {
    struct_name: P256_ECC_STRUCT_NAME.to_string(),
    modulus: P256_MODULUS.clone(),
    scalar: P256_ORDER.clone(),
    a: BigUint::from_bytes_le(&P256_A),
    b: BigUint::from_bytes_le(&P256_B),
});

#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct WeierstrassExtension {
    pub supported_curves: Vec<CurveConfig>,
}

impl WeierstrassExtension {
    pub fn generate_sw_init(&self) -> String {
        let supported_curves = self
            .supported_curves
            .iter()
            .map(|curve_config| format!("\"{}\"", curve_config.struct_name))
            .collect::<Vec<String>>()
            .join(", ");

        format!("openvm_ecc_guest::sw_macros::sw_init! {{ {supported_curves} }}")
    }
}

#[derive(Clone, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum WeierstrassExtensionExecutor {
    // 32 limbs prime
    EcAddNeRv32_32(EcAddNeExecutor<2, 32>),
    EcDoubleRv32_32(EcDoubleExecutor<2, 32>),
    // 48 limbs prime
    EcAddNeRv32_48(EcAddNeExecutor<6, 16>),
    EcDoubleRv32_48(EcDoubleExecutor<6, 16>),
}

impl<F: PrimeField32> VmExecutionExtension<F> for WeierstrassExtension {
    type Executor = WeierstrassExtensionExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, WeierstrassExtensionExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();
        // TODO: somehow get the range checker bus from `ExecutorInventory`
        let dummy_range_checker_bus = VariableRangeCheckerBus::new(u16::MAX, 16);
        for (i, curve) in self.supported_curves.iter().enumerate() {
            let start_offset =
                Rv32WeierstrassOpcode::CLASS_OFFSET + i * Rv32WeierstrassOpcode::COUNT;
            let bytes = curve.modulus.bits().div_ceil(8);

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };
                let addne = get_ec_addne_step(
                    config.clone(),
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    WeierstrassExtensionExecutor::EcAddNeRv32_32(addne),
                    ((Rv32WeierstrassOpcode::EC_ADD_NE as usize)
                        ..=(Rv32WeierstrassOpcode::SETUP_EC_ADD_NE as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let double = get_ec_double_step(
                    config,
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                    curve.a.clone(),
                );

                inventory.add_executor(
                    WeierstrassExtensionExecutor::EcDoubleRv32_32(double),
                    ((Rv32WeierstrassOpcode::EC_DOUBLE as usize)
                        ..=(Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };
                let addne = get_ec_addne_step(
                    config.clone(),
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    WeierstrassExtensionExecutor::EcAddNeRv32_48(addne),
                    ((Rv32WeierstrassOpcode::EC_ADD_NE as usize)
                        ..=(Rv32WeierstrassOpcode::SETUP_EC_ADD_NE as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let double = get_ec_double_step(
                    config,
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                    curve.a.clone(),
                );

                inventory.add_executor(
                    WeierstrassExtensionExecutor::EcDoubleRv32_48(double),
                    ((Rv32WeierstrassOpcode::EC_DOUBLE as usize)
                        ..=(Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for WeierstrassExtension {
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
        for (i, curve) in self.supported_curves.iter().enumerate() {
            let start_offset =
                Rv32WeierstrassOpcode::CLASS_OFFSET + i * Rv32WeierstrassOpcode::COUNT;
            let bytes = curve.modulus.bits().div_ceil(8);

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                let addne = get_ec_addne_air::<2, 32>(
                    exec_bridge,
                    memory_bridge,
                    config.clone(),
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                );
                inventory.add_air(addne);

                let double = get_ec_double_air::<2, 32>(
                    exec_bridge,
                    memory_bridge,
                    config,
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                    curve.a.clone(),
                );
                inventory.add_air(double);
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };

                let addne = get_ec_addne_air::<6, 16>(
                    exec_bridge,
                    memory_bridge,
                    config.clone(),
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                );
                inventory.add_air(addne);

                let double = get_ec_double_air::<6, 16>(
                    exec_bridge,
                    memory_bridge,
                    config,
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                    curve.a.clone(),
                );
                inventory.add_air(double);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, WeierstrassExtension> for EccCpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        extension: &WeierstrassExtension,
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
        for curve in extension.supported_curves.iter() {
            let bytes = curve.modulus.bits().div_ceil(8);

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                inventory.next_air::<WeierstrassAir<2, 2, 32>>()?;
                let addne = get_ec_addne_chip::<Val<SC>, 2, 32>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(addne);

                inventory.next_air::<WeierstrassAir<1, 2, 32>>()?;
                let double = get_ec_double_chip::<Val<SC>, 2, 32>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                    curve.a.clone(),
                );
                inventory.add_executor_chip(double);
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };

                inventory.next_air::<WeierstrassAir<2, 6, 16>>()?;
                let addne = get_ec_addne_chip::<Val<SC>, 6, 16>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(addne);

                inventory.next_air::<WeierstrassAir<1, 6, 16>>()?;
                let double = get_ec_double_chip::<Val<SC>, 6, 16>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                    curve.a.clone(),
                );
                inventory.add_executor_chip(double);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

// Convenience constants for constructors
lazy_static! {
    // The constants are taken from: https://en.bitcoin.it/wiki/Secp256k1
    pub static ref SECP256K1_MODULUS: BigUint = BigUint::from_bytes_be(&hex!(
        "FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F"
    ));
    pub static ref SECP256K1_ORDER: BigUint = BigUint::from_bytes_be(&hex!(
        "FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
    ));
}

lazy_static! {
    // The constants are taken from: https://neuromancer.sk/std/secg/secp256r1
    pub static ref P256_MODULUS: BigUint = BigUint::from_bytes_be(&hex!(
        "ffffffff00000001000000000000000000000000ffffffffffffffffffffffff"
    ));
    pub static ref P256_ORDER: BigUint = BigUint::from_bytes_be(&hex!(
        "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551"
    ));
}
// little-endian
const P256_A: [u8; 32] = hex!("fcffffffffffffffffffffff00000000000000000000000001000000ffffffff");
// little-endian
const P256_B: [u8; 32] = hex!("4b60d2273e3cce3bf6b053ccb0061d65bc86987655bdebb3e7933aaad835c65a");

pub const SECP256K1_ECC_STRUCT_NAME: &str = "Secp256k1Point";
pub const P256_ECC_STRUCT_NAME: &str = "P256Point";
