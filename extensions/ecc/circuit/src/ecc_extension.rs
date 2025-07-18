use derive_more::derive::From;
use hex_literal::hex;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, Zero};
use once_cell::sync::Lazy;
use openvm_circuit::{
    arch::{
        ExecutionBridge, SystemPort, VmExtension, VmInventory, VmInventoryBuilder, VmInventoryError,
    },
    system::phantom::PhantomChip,
};
use openvm_circuit_derive::{AnyEnum, InsExecutorE1, InsExecutorE2, InstructionExecutor};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_ecc_guest::{
    algebra::IntMod,
    ed25519::{CURVE_A as ED25519_A, CURVE_D as ED25519_D, ED25519_MODULUS, ED25519_ORDER},
};
use openvm_ecc_transpiler::{Rv32EdwardsOpcode, Rv32WeierstrassOpcode};
use openvm_instructions::{LocalOpcode, VmOpcode};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use strum::EnumCount;

use super::{SwAddNeChip, SwDoubleChip, TeAddChip};

#[serde_as]
#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct CurveConfig<T> {
    /// The name of the curve struct as defined by moduli_declare.
    pub struct_name: String,
    /// The coordinate modulus of the curve.
    #[serde_as(as = "DisplayFromStr")]
    pub modulus: BigUint,
    /// The scalar field modulus of the curve.
    #[serde_as(as = "DisplayFromStr")]
    pub scalar: BigUint,
    // curve-specific coefficients
    pub coeffs: T,
}

#[serde_as]
#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct SwCurveCoeffs {
    /// The coefficient a of y^2 = x^3 + ax + b.
    #[serde_as(as = "DisplayFromStr")]
    pub a: BigUint,
    /// The coefficient b of y^2 = x^3 + ax + b.
    #[serde_as(as = "DisplayFromStr")]
    pub b: BigUint,
}

#[serde_as]
#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct TeCurveCoeffs {
    /// The coefficient a of ax^2 + y^2 = 1 + dx^2y^2
    #[serde_as(as = "DisplayFromStr")]
    pub a: BigUint,
    /// The coefficient d of ax^2 + y^2 = 1 + dx^2y^2
    #[serde_as(as = "DisplayFromStr")]
    pub d: BigUint,
}

pub static SECP256K1_CONFIG: Lazy<CurveConfig<SwCurveCoeffs>> = Lazy::new(|| CurveConfig {
    struct_name: "Secp256k1Point".to_string(),
    modulus: BigUint::from_bytes_be(&hex!(
        "FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F"
    )),
    scalar: BigUint::from_bytes_be(&hex!(
        "FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
    )),
    coeffs: SwCurveCoeffs {
        a: BigUint::zero(),
        b: BigUint::from_u8(7u8).unwrap(),
    },
});

pub static P256_CONFIG: Lazy<CurveConfig<SwCurveCoeffs>> = Lazy::new(|| CurveConfig {
    struct_name: "P256Point".to_string(),
    modulus: BigUint::from_bytes_be(&hex!(
        "ffffffff00000001000000000000000000000000ffffffffffffffffffffffff"
    )),
    scalar: BigUint::from_bytes_be(&hex!(
        "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551"
    )),
    coeffs: SwCurveCoeffs {
        a: BigUint::from_bytes_le(&hex!(
            "fcffffffffffffffffffffff00000000000000000000000001000000ffffffff"
        )),
        b: BigUint::from_bytes_le(&hex!(
            "4b60d2273e3cce3bf6b053ccb0061d65bc86987655bdebb3e7933aaad835c65a"
        )),
    },
});

pub static ED25519_CONFIG: Lazy<CurveConfig<TeCurveCoeffs>> = Lazy::new(|| CurveConfig {
    struct_name: "Ed25519Point".to_string(),
    modulus: ED25519_MODULUS.clone(),
    scalar: ED25519_ORDER.clone(),
    coeffs: TeCurveCoeffs {
        a: BigUint::from_bytes_le(ED25519_A.as_le_bytes()),
        d: BigUint::from_bytes_le(ED25519_D.as_le_bytes()),
    },
});

#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct EccExtension {
    #[serde(default)]
    pub supported_sw_curves: Vec<CurveConfig<SwCurveCoeffs>>,
    #[serde(default)]
    pub supported_te_curves: Vec<CurveConfig<TeCurveCoeffs>>,
}

impl EccExtension {
    pub fn generate_ecc_init(&self) -> String {
        let supported_sw_curves = self
            .supported_sw_curves
            .iter()
            .map(|curve_config| curve_config.struct_name.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        let supported_te_curves = self
            .supported_te_curves
            .iter()
            .map(|curve_config| curve_config.struct_name.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        format!(
            "openvm_ecc_guest::sw_macros::sw_init! {{ {supported_sw_curves} }}\nopenvm_ecc_guest::te_macros::te_init! {{ {supported_te_curves} }}"
        )
    }
}

#[derive(Chip, ChipUsageGetter, InstructionExecutor, AnyEnum, InsExecutorE1, InsExecutorE2)]
pub enum EccExtensionExecutor<F: PrimeField32> {
    // 32 limbs prime
    SwEcAddNeRv32_32(SwAddNeChip<F, 2, 32>),
    SwEcDoubleRv32_32(SwDoubleChip<F, 2, 32>),
    // 48 limbs prime
    SwEcAddNeRv32_48(SwAddNeChip<F, 6, 16>),
    SwEcDoubleRv32_48(SwDoubleChip<F, 6, 16>),
    // 32 limbs prime
    TeEcAddRv32_32(TeAddChip<F, 2, 32>),
    // 48 limbs prime
    TeEcAddRv32_48(TeAddChip<F, 6, 16>),
}

#[derive(ChipUsageGetter, Chip, AnyEnum, From)]
pub enum EccExtensionPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

impl<F: PrimeField32> VmExtension<F> for EccExtension {
    type Executor = EccExtensionExecutor<F>;
    type Periphery = EccExtensionPeriphery<F>;

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

        let sw_add_ne_opcodes = (Rv32WeierstrassOpcode::SW_ADD_NE as usize)
            ..=(Rv32WeierstrassOpcode::SETUP_SW_ADD_NE as usize);
        let sw_double_opcodes = (Rv32WeierstrassOpcode::SW_DOUBLE as usize)
            ..=(Rv32WeierstrassOpcode::SETUP_SW_DOUBLE as usize);

        let te_add_opcodes =
            (Rv32EdwardsOpcode::TE_ADD as usize)..=(Rv32EdwardsOpcode::SETUP_TE_ADD as usize);

        for (sw_idx, curve) in self.supported_sw_curves.iter().enumerate() {
            // TODO: Better support for different limb sizes. Currently only 32 or 48 limbs are
            // supported.
            let sw_start_offset =
                Rv32WeierstrassOpcode::CLASS_OFFSET + sw_idx * Rv32WeierstrassOpcode::COUNT;
            let bytes = curve.modulus.bits().div_ceil(8);
            let config32 = ExprBuilderConfig {
                modulus: curve.modulus.clone(),
                num_limbs: 32,
                limb_bits: 8,
            };
            let config48 = ExprBuilderConfig {
                modulus: curve.modulus.clone(),
                num_limbs: 48,
                limb_bits: 8,
            };
            if bytes <= 32 {
                let sw_add_ne_chip = SwAddNeChip::new(
                    execution_bridge,
                    memory_bridge,
                    builder.system_base().memory_controller.helper(),
                    pointer_max_bits,
                    config32.clone(),
                    sw_start_offset,
                    bitwise_lu_chip.clone(),
                    range_checker.clone(),
                );
                inventory.add_executor(
                    EccExtensionExecutor::SwEcAddNeRv32_32(sw_add_ne_chip),
                    sw_add_ne_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + sw_start_offset)),
                )?;
                let sw_double_chip = SwDoubleChip::new(
                    execution_bridge,
                    memory_bridge,
                    builder.system_base().memory_controller.helper(),
                    pointer_max_bits,
                    config32.clone(),
                    sw_start_offset,
                    bitwise_lu_chip.clone(),
                    range_checker.clone(),
                    curve.coeffs.a.clone(),
                );
                inventory.add_executor(
                    EccExtensionExecutor::SwEcDoubleRv32_32(sw_double_chip),
                    sw_double_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + sw_start_offset)),
                )?;
            } else if bytes <= 48 {
                let sw_add_ne_chip = SwAddNeChip::new(
                    execution_bridge,
                    memory_bridge,
                    builder.system_base().memory_controller.helper(),
                    pointer_max_bits,
                    config48.clone(),
                    sw_start_offset,
                    bitwise_lu_chip.clone(),
                    range_checker.clone(),
                );
                inventory.add_executor(
                    EccExtensionExecutor::SwEcAddNeRv32_48(sw_add_ne_chip),
                    sw_add_ne_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + sw_start_offset)),
                )?;
                let sw_double_chip = SwDoubleChip::new(
                    execution_bridge,
                    memory_bridge,
                    builder.system_base().memory_controller.helper(),
                    pointer_max_bits,
                    config48.clone(),
                    sw_start_offset,
                    bitwise_lu_chip.clone(),
                    range_checker.clone(),
                    curve.coeffs.a.clone(),
                );
                inventory.add_executor(
                    EccExtensionExecutor::SwEcDoubleRv32_48(sw_double_chip),
                    sw_double_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + sw_start_offset)),
                )?;
            } else {
                panic!("Modulus too large");
            }
        }

        for (te_idx, curve) in self.supported_te_curves.iter().enumerate() {
            let bytes = curve.modulus.bits().div_ceil(8);
            let config32 = ExprBuilderConfig {
                modulus: curve.modulus.clone(),
                num_limbs: 32,
                limb_bits: 8,
            };
            let config48 = ExprBuilderConfig {
                modulus: curve.modulus.clone(),
                num_limbs: 48,
                limb_bits: 8,
            };
            let te_start_offset =
                Rv32EdwardsOpcode::CLASS_OFFSET + te_idx * Rv32EdwardsOpcode::COUNT;
            if bytes <= 32 {
                let te_add_chip = TeAddChip::new(
                    execution_bridge,
                    memory_bridge,
                    builder.system_base().memory_controller.helper(),
                    pointer_max_bits,
                    config32.clone(),
                    te_start_offset,
                    bitwise_lu_chip.clone(),
                    range_checker.clone(),
                    curve.coeffs.a.clone(),
                    curve.coeffs.d.clone(),
                );
                inventory.add_executor(
                    EccExtensionExecutor::TeEcAddRv32_32(te_add_chip),
                    te_add_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + te_start_offset)),
                )?;
            } else if bytes <= 48 {
                let te_add_chip = TeAddChip::new(
                    execution_bridge,
                    memory_bridge,
                    builder.system_base().memory_controller.helper(),
                    pointer_max_bits,
                    config48.clone(),
                    te_start_offset,
                    bitwise_lu_chip.clone(),
                    range_checker.clone(),
                    curve.coeffs.a.clone(),
                    curve.coeffs.d.clone(),
                );
                inventory.add_executor(
                    EccExtensionExecutor::TeEcAddRv32_48(te_add_chip),
                    te_add_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + te_start_offset)),
                )?;
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(inventory)
    }
}
