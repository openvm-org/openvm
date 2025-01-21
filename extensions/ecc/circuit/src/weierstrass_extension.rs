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

use super::{EcAddNeChip, EcDoubleChip, TeEcAddChip};

#[serde_as]
#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct CurveConfig {
    /// The coordinate modulus of the curve.
    #[serde_as(as = "DisplayFromStr")]
    pub modulus: BigUint,
    /// The scalar field modulus of the curve.
    #[serde_as(as = "DisplayFromStr")]
    pub scalar: BigUint,
    // curve-specific coefficients
    pub coeffs: CurveCoeffs,
}

#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
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
pub struct WeierstrassExtension {
    pub supported_curves: Vec<CurveConfig>,
}

#[derive(Chip, ChipUsageGetter, InstructionExecutor, AnyEnum, BytesStateful)]
pub enum WeierstrassExtensionExecutor<F: PrimeField32> {
    // 32 limbs prime
    SwEcAddNeRv32_32(EcAddNeChip<F, 2, 32>),
    SwEcDoubleRv32_32(EcDoubleChip<F, 2, 32>),
    // 48 limbs prime
    SwEcAddNeRv32_48(EcAddNeChip<F, 6, 16>),
    SwEcDoubleRv32_48(EcDoubleChip<F, 6, 16>),
    // 32 limbs prime
    TeEcAddRv32_32(TeEcAddChip<F, 2, 32>),
    // 48 limbs prime
    TeEcAddRv32_48(TeEcAddChip<F, 6, 16>),
}

#[derive(ChipUsageGetter, Chip, AnyEnum, From, BytesStateful)]
pub enum WeierstrassExtensionPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
    Phantom(PhantomChip<F>),
}

impl<F: PrimeField32> VmExtension<F> for WeierstrassExtension {
    type Executor = WeierstrassExtensionExecutor<F>;
    type Periphery = WeierstrassExtensionPeriphery<F>;

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
            // TODO: Better support for different limb sizes. Currently only 32 or 48 limbs are supported.
            if bytes <= 32 {
                match curve.coeffs.clone() {
                    CurveCoeffs::SwCurve(SwCurveConfig { a, b: _ }) => {
                        let sw_add_ne_chip = EcAddNeChip::new(
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
                            WeierstrassExtensionExecutor::SwEcAddNeRv32_32(sw_add_ne_chip),
                            sw_add_ne_opcodes
                                .clone()
                                .map(|x| VmOpcode::from_usize(x + sw_start_offset)),
                        )?;
                        let sw_double_chip = EcDoubleChip::new(
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
                            WeierstrassExtensionExecutor::SwEcDoubleRv32_32(sw_double_chip),
                            sw_double_opcodes
                                .clone()
                                .map(|x| VmOpcode::from_usize(x + sw_start_offset)),
                        )?;
                    }

                    CurveCoeffs::TeCurve(TeCurveConfig { a, d }) => {
                        let te_add_chip = TeEcAddChip::new(
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
                            WeierstrassExtensionExecutor::TeEcAddRv32_32(te_add_chip),
                            sw_add_ne_opcodes
                                .clone()
                                .map(|x| VmOpcode::from_usize(x + te_start_offset)),
                        )?;
                    }
                }
            } else if bytes <= 48 {
                match curve.coeffs.clone() {
                    CurveCoeffs::SwCurve(SwCurveConfig { a, b: _ }) => {
                        let sw_add_ne_chip = EcAddNeChip::new(
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
                            WeierstrassExtensionExecutor::SwEcAddNeRv32_48(sw_add_ne_chip),
                            sw_add_ne_opcodes
                                .clone()
                                .map(|x| VmOpcode::from_usize(x + sw_start_offset)),
                        )?;
                        let sw_double_chip = EcDoubleChip::new(
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
                            WeierstrassExtensionExecutor::SwEcDoubleRv32_48(sw_double_chip),
                            sw_double_opcodes
                                .clone()
                                .map(|x| VmOpcode::from_usize(x + sw_start_offset)),
                        )?;
                    }

                    CurveCoeffs::TeCurve(TeCurveConfig { a, d }) => {
                        let te_add_chip = TeEcAddChip::new(
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
                            WeierstrassExtensionExecutor::TeEcAddRv32_48(te_add_chip),
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
        builder.add_phantom_sub_executor(
            phantom::DecompressHintSubEx::new(self.supported_curves.clone()),
            PhantomDiscriminant(EccPhantom::HintDecompress as u16),
        )?;

        Ok(inventory)
    }
}

pub(crate) mod phantom {
    use std::iter::repeat;

    use eyre::bail;
    use num_bigint::BigUint;
    use num_integer::Integer;
    use num_traits::One;
    use openvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::MemoryController,
    };
    use openvm_instructions::{riscv::RV32_MEMORY_AS, PhantomDiscriminant};
    use openvm_rv32im_circuit::adapters::unsafe_read_rv32_register;
    use openvm_stark_backend::p3_field::PrimeField32;

    use super::{CurveCoeffs, CurveConfig, SwCurveConfig, TeCurveConfig};

    #[derive(derive_new::new)]
    pub struct DecompressHintSubEx {
        pub supported_curves: Vec<CurveConfig>,
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for DecompressHintSubEx {
        fn phantom_execute(
            &mut self,
            memory: &MemoryController<F>,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            a: F,
            b: F,
            c_upper: u16,
        ) -> eyre::Result<()> {
            let c_idx = c_upper as usize;
            if c_idx >= self.supported_curves.len() {
                bail!(
                    "Curve index {c_idx} out of range: {} supported curves",
                    self.supported_curves.len()
                );
            }
            let curve = &self.supported_curves[c_idx];
            let modulus_mod_4 = BigUint::from(3u8) & curve.modulus.clone();
            if modulus_mod_4 != BigUint::from(3u8) {
                bail!("Currently only supporting curves with modulus congruent to 3 mod 4.");
                // TODO: Tonelli-Shanks algorithm
            }
            let rs1 = unsafe_read_rv32_register(memory, a);
            // TODO: Better support for different limb sizes
            let num_limbs: usize = if curve.modulus.bits().div_ceil(8) <= 32 {
                32
            } else if curve.modulus.bits().div_ceil(8) <= 48 {
                48
            } else {
                bail!("Modulus too large")
            };
            let mut x_limbs: Vec<u8> = Vec::with_capacity(num_limbs);
            for i in 0..num_limbs {
                let limb = memory.unsafe_read_cell(
                    F::from_canonical_u32(RV32_MEMORY_AS),
                    F::from_canonical_u32(rs1 + i as u32),
                );
                x_limbs.push(limb.as_canonical_u32() as u8);
            }
            let x = BigUint::from_bytes_le(&x_limbs);
            let rs2 = unsafe_read_rv32_register(memory, b);
            let rec_id = memory.unsafe_read_cell(
                F::from_canonical_u32(RV32_MEMORY_AS),
                F::from_canonical_u32(rs2),
            );
            let y = decompress_point(x, rec_id.as_canonical_u32() & 1 == 1, curve);
            let y_bytes = y
                .to_bytes_le()
                .into_iter()
                .map(F::from_canonical_u8)
                .chain(repeat(F::ZERO))
                .take(num_limbs)
                .collect();
            streams.hint_stream = y_bytes;
            Ok(())
        }
    }

    fn decompress_point(x: BigUint, is_y_odd: bool, curve: &CurveConfig) -> BigUint {
        match curve.coeffs.clone() {
            CurveCoeffs::SwCurve(SwCurveConfig { a, b }) => {
                let alpha = ((&x * &x * &x) + (&x * &a) + &b) % &curve.modulus;
                let beta = mod_sqrt(alpha, &curve.modulus);
                if is_y_odd == beta.is_odd() {
                    beta
                } else {
                    &curve.modulus - &beta
                }
            }
            CurveCoeffs::TeCurve(TeCurveConfig { a: _, d: _ }) => {
                unimplemented!("should not call decompress_point for Twisted Edwards curves");
            }
        }
    }

    fn mod_sqrt(x: BigUint, modulus: &BigUint) -> BigUint {
        let exponent = (modulus + BigUint::one()) >> 2;
        x.modpow(&exponent, modulus)
    }
}
