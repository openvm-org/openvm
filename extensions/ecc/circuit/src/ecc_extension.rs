use derive_more::derive::From;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, Zero};
use once_cell::sync::Lazy;
use openvm_algebra_guest::IntMod;
use openvm_circuit::{
    arch::{SystemPort, VmExtension, VmInventory, VmInventoryBuilder, VmInventoryError},
    system::phantom::PhantomChip,
};
use openvm_circuit_derive::{AnyEnum, InstructionExecutor};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_circuit_primitives_derive::{BytesStateful, Chip, ChipUsageGetter};
use openvm_ecc_guest::{
    k256::{SECP256K1_MODULUS, SECP256K1_ORDER},
    p256::{CURVE_A as P256_A, CURVE_B as P256_B, P256_MODULUS, P256_ORDER},
};
use openvm_ecc_transpiler::{EccPhantom, Rv32EdwardsOpcode, Rv32WeierstrassOpcode};
use openvm_instructions::{LocalOpcode, PhantomDiscriminant, VmOpcode};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_rv32_adapters::Rv32VecHeapAdapterChip;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use strum::EnumCount;

use super::{SwAddNeChip, SwDoubleChip, TeAddChip};

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
    // curve-specific coefficients
    #[serde_as(as = "_")]
    pub coeffs: CurveCoeffs,
}

#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CurveCoeffs {
    SwCurve(SwCurveConfig),
    TeCurve(TeCurveConfig),
}

#[serde_as]
#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct SwCurveConfig {
    /// The coefficient a of y^2 = x^3 + ax + b.
    #[serde_as(as = "DisplayFromStr")]
    pub a: BigUint,
    /// The coefficient b of y^2 = x^3 + ax + b.
    #[serde_as(as = "DisplayFromStr")]
    pub b: BigUint,
}

#[serde_as]
#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct TeCurveConfig {
    /// The coefficient a of ax^2 + y^2 = 1 + dx^2y^2
    #[serde_as(as = "DisplayFromStr")]
    pub a: BigUint,
    /// The coefficient d of ax^2 + y^2 = 1 + dx^2y^2
    #[serde_as(as = "DisplayFromStr")]
    pub d: BigUint,
}

pub static SECP256K1_CONFIG: Lazy<CurveConfig> = Lazy::new(|| CurveConfig {
    modulus: SECP256K1_MODULUS.clone(),
    scalar: SECP256K1_ORDER.clone(),
    coeffs: CurveCoeffs::SwCurve(SwCurveConfig {
        a: BigUint::zero(),
        b: BigUint::from_u8(7u8).unwrap(),
    }),
});

pub static P256_CONFIG: Lazy<CurveConfig> = Lazy::new(|| CurveConfig {
    modulus: P256_MODULUS.clone(),
    scalar: P256_ORDER.clone(),
    coeffs: CurveCoeffs::SwCurve(SwCurveConfig {
        a: BigUint::from_bytes_le(P256_A.as_le_bytes()),
        b: BigUint::from_bytes_le(P256_B.as_le_bytes()),
    }),
});

#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct EccExtension {
    pub supported_curves: Vec<CurveConfig>,
}

#[derive(Chip, ChipUsageGetter, InstructionExecutor, AnyEnum, BytesStateful)]
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

#[derive(ChipUsageGetter, Chip, AnyEnum, From, BytesStateful)]
pub enum EccExtensionPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
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
        let range_checker = builder.system_base().range_checker_chip.clone();
        let pointer_bits = builder.system_config().memory_config.pointer_max_bits;

        let sw_add_ne_opcodes = (Rv32WeierstrassOpcode::EC_ADD_NE as usize)
            ..=(Rv32WeierstrassOpcode::SETUP_EC_ADD_NE as usize);
        let sw_double_opcodes = (Rv32WeierstrassOpcode::EC_DOUBLE as usize)
            ..=(Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize);

        let te_add_opcodes =
            (Rv32EdwardsOpcode::EC_ADD as usize)..=(Rv32EdwardsOpcode::SETUP_EC_ADD as usize);

        for (i, curve) in self.supported_curves.iter().enumerate() {
            let sw_start_offset =
                Rv32WeierstrassOpcode::CLASS_OFFSET + i * Rv32WeierstrassOpcode::COUNT;
            // right now this is the same as sw_class_offset
            let te_start_offset = Rv32EdwardsOpcode::CLASS_OFFSET + i * Rv32EdwardsOpcode::COUNT;

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
                match curve.coeffs.clone() {
                    CurveCoeffs::SwCurve(SwCurveConfig { a, b: _ }) => {
                        let sw_add_ne_chip = SwAddNeChip::new(
                            Rv32VecHeapAdapterChip::<F, 2, 2, 2, 32, 32>::new(
                                execution_bus,
                                program_bus,
                                memory_bridge,
                                pointer_bits,
                                bitwise_lu_chip.clone(),
                            ),
                            config32.clone(),
                            sw_start_offset,
                            range_checker.clone(),
                            offline_memory.clone(),
                        );
                        inventory.add_executor(
                            EccExtensionExecutor::SwEcAddNeRv32_32(sw_add_ne_chip),
                            sw_add_ne_opcodes
                                .clone()
                                .map(|x| VmOpcode::from_usize(x + sw_start_offset)),
                        )?;
                        let sw_double_chip = SwDoubleChip::new(
                            Rv32VecHeapAdapterChip::<F, 1, 2, 2, 32, 32>::new(
                                execution_bus,
                                program_bus,
                                memory_bridge,
                                pointer_bits,
                                bitwise_lu_chip.clone(),
                            ),
                            range_checker.clone(),
                            config32.clone(),
                            sw_start_offset,
                            a.clone(),
                            offline_memory.clone(),
                        );
                        inventory.add_executor(
                            EccExtensionExecutor::SwEcDoubleRv32_32(sw_double_chip),
                            sw_double_opcodes
                                .clone()
                                .map(|x| VmOpcode::from_usize(x + sw_start_offset)),
                        )?;
                    }

                    CurveCoeffs::TeCurve(TeCurveConfig { a, d }) => {
                        let te_add_chip = TeAddChip::new(
                            Rv32VecHeapAdapterChip::<F, 2, 2, 2, 32, 32>::new(
                                execution_bus,
                                program_bus,
                                memory_bridge,
                                pointer_bits,
                                bitwise_lu_chip.clone(),
                            ),
                            config32.clone(),
                            te_start_offset,
                            a.clone(),
                            d.clone(),
                            range_checker.clone(),
                            offline_memory.clone(),
                        );
                        inventory.add_executor(
                            EccExtensionExecutor::TeEcAddRv32_32(te_add_chip),
                            sw_add_ne_opcodes
                                .clone()
                                .map(|x| VmOpcode::from_usize(x + te_start_offset)),
                        )?;
                    }
                }
            } else if bytes <= 48 {
                match curve.coeffs.clone() {
                    CurveCoeffs::SwCurve(SwCurveConfig { a, b: _ }) => {
                        let sw_add_ne_chip = SwAddNeChip::new(
                            Rv32VecHeapAdapterChip::<F, 2, 6, 6, 16, 16>::new(
                                execution_bus,
                                program_bus,
                                memory_bridge,
                                pointer_bits,
                                bitwise_lu_chip.clone(),
                            ),
                            config48.clone(),
                            sw_start_offset,
                            range_checker.clone(),
                            offline_memory.clone(),
                        );
                        inventory.add_executor(
                            EccExtensionExecutor::SwEcAddNeRv32_48(sw_add_ne_chip),
                            sw_add_ne_opcodes
                                .clone()
                                .map(|x| VmOpcode::from_usize(x + sw_start_offset)),
                        )?;
                        let sw_double_chip = SwDoubleChip::new(
                            Rv32VecHeapAdapterChip::<F, 1, 6, 6, 16, 16>::new(
                                execution_bus,
                                program_bus,
                                memory_bridge,
                                pointer_bits,
                                bitwise_lu_chip.clone(),
                            ),
                            range_checker.clone(),
                            config48.clone(),
                            sw_start_offset,
                            a.clone(),
                            offline_memory.clone(),
                        );
                        inventory.add_executor(
                            EccExtensionExecutor::SwEcDoubleRv32_48(sw_double_chip),
                            sw_double_opcodes
                                .clone()
                                .map(|x| VmOpcode::from_usize(x + sw_start_offset)),
                        )?;
                    }

                    CurveCoeffs::TeCurve(TeCurveConfig { a, d }) => {
                        let te_add_chip = TeAddChip::new(
                            Rv32VecHeapAdapterChip::<F, 2, 6, 6, 16, 16>::new(
                                execution_bus,
                                program_bus,
                                memory_bridge,
                                pointer_bits,
                                bitwise_lu_chip.clone(),
                            ),
                            config48.clone(),
                            te_start_offset,
                            a.clone(),
                            d.clone(),
                            range_checker.clone(),
                            offline_memory.clone(),
                        );
                        inventory.add_executor(
                            EccExtensionExecutor::TeEcAddRv32_48(te_add_chip),
                            te_add_opcodes
                                .clone()
                                .map(|x| VmOpcode::from_usize(x + te_start_offset)),
                        )?;
                    }
                }
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(inventory)
    }
}
