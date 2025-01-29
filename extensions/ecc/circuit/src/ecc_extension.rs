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
    ed25519::{CURVE_A as ED25519_A, CURVE_D as ED25519_D, ED25519_MODULUS, ED25519_ORDER},
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
pub struct CurveConfig<T> {
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
    modulus: SECP256K1_MODULUS.clone(),
    scalar: SECP256K1_ORDER.clone(),
    coeffs: SwCurveCoeffs {
        a: BigUint::zero(),
        b: BigUint::from_u8(7u8).unwrap(),
    },
});

pub static P256_CONFIG: Lazy<CurveConfig<SwCurveCoeffs>> = Lazy::new(|| CurveConfig {
    modulus: P256_MODULUS.clone(),
    scalar: P256_ORDER.clone(),
    coeffs: SwCurveCoeffs {
        a: BigUint::from_bytes_le(P256_A.as_le_bytes()),
        b: BigUint::from_bytes_le(P256_B.as_le_bytes()),
    },
});

pub static ED25519_CONFIG: Lazy<CurveConfig<TeCurveCoeffs>> = Lazy::new(|| CurveConfig {
    modulus: ED25519_MODULUS.clone(),
    scalar: ED25519_ORDER.clone(),
    coeffs: TeCurveCoeffs {
        a: BigUint::from_bytes_le(ED25519_A.as_le_bytes()),
        d: BigUint::from_bytes_le(ED25519_D.as_le_bytes()),
    },
});

#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct EccExtension {
    pub supported_sw_curves: Vec<CurveConfig<SwCurveCoeffs>>,
    pub supported_te_curves: Vec<CurveConfig<TeCurveCoeffs>>,
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

        let sw_add_ne_opcodes = (Rv32WeierstrassOpcode::SW_ADD_NE as usize)
            ..=(Rv32WeierstrassOpcode::SETUP_SW_ADD_NE as usize);
        let sw_double_opcodes = (Rv32WeierstrassOpcode::SW_DOUBLE as usize)
            ..=(Rv32WeierstrassOpcode::SETUP_SW_DOUBLE as usize);

        let te_add_opcodes =
            (Rv32EdwardsOpcode::TE_ADD as usize)..=(Rv32EdwardsOpcode::SETUP_TE_ADD as usize);

        for (sw_idx, curve) in self.supported_sw_curves.iter().enumerate() {
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
            let sw_start_offset =
                Rv32WeierstrassOpcode::CLASS_OFFSET + sw_idx * Rv32WeierstrassOpcode::COUNT;
            if bytes <= 32 {
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
                    curve.coeffs.a.clone(),
                    offline_memory.clone(),
                );
                inventory.add_executor(
                    EccExtensionExecutor::SwEcDoubleRv32_32(sw_double_chip),
                    sw_double_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + sw_start_offset)),
                )?;
            } else if bytes <= 48 {
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
                    curve.coeffs.a.clone(),
                    offline_memory.clone(),
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
                    Rv32VecHeapAdapterChip::<F, 2, 2, 2, 32, 32>::new(
                        execution_bus,
                        program_bus,
                        memory_bridge,
                        pointer_bits,
                        bitwise_lu_chip.clone(),
                    ),
                    config32.clone(),
                    te_start_offset,
                    curve.coeffs.a.clone(),
                    curve.coeffs.d.clone(),
                    range_checker.clone(),
                    offline_memory.clone(),
                );
                inventory.add_executor(
                    EccExtensionExecutor::TeEcAddRv32_32(te_add_chip),
                    te_add_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + te_start_offset)),
                )?;
            } else if bytes <= 48 {
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
                    curve.coeffs.a.clone(),
                    curve.coeffs.d.clone(),
                    range_checker.clone(),
                    offline_memory.clone(),
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
        builder.add_phantom_sub_executor(
            phantom::DecompressHintSubEx::new(self.supported_sw_curves.clone()),
            PhantomDiscriminant(EccPhantom::SwHintDecompress as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::DecompressHintSubEx::new(self.supported_te_curves.clone()),
            PhantomDiscriminant(EccPhantom::TeHintDecompress as u16),
        )?;

        Ok(inventory)
    }
}

pub(crate) mod phantom {
    use std::iter::repeat;

    use eyre::bail;
    use num_bigint::BigUint;
    use num_integer::Integer;
    use num_traits::{FromPrimitive, One};
    use openvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::MemoryController,
    };
    use openvm_instructions::{riscv::RV32_MEMORY_AS, PhantomDiscriminant};
    use openvm_rv32im_circuit::adapters::unsafe_read_rv32_register;
    use openvm_stark_backend::p3_field::PrimeField32;

    use super::{CurveConfig, SwCurveCoeffs, TeCurveCoeffs};

    #[derive(derive_new::new)]
    pub struct DecompressHintSubEx<T> {
        pub supported_curves: Vec<CurveConfig<T>>,
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for DecompressHintSubEx<SwCurveCoeffs> {
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
                bail!(
                        "Currently only supporting short Weierstrass curves with modulus congruent to 3 mod 4."
                    );
                // TODO: Tonelli-Shanks algorithm
            }

            let rs1 = unsafe_read_rv32_register(memory, a);
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

            let y = decompress_sw_point(x, rec_id.as_canonical_u32() & 1 == 1, curve);

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

    fn decompress_sw_point(
        x: BigUint,
        is_y_odd: bool,
        curve: &CurveConfig<SwCurveCoeffs>,
    ) -> BigUint {
        let alpha = ((&x * &x * &x) + (&x * &curve.coeffs.a) + &curve.coeffs.b) % &curve.modulus;
        let beta = mod_sqrt(alpha, &curve.modulus);
        if is_y_odd == beta.is_odd() {
            beta
        } else {
            &curve.modulus - &beta
        }
    }

    fn mod_sqrt(x: BigUint, modulus: &BigUint) -> BigUint {
        let exponent = (modulus + BigUint::one()) >> 2;
        x.modpow(&exponent, modulus)
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for DecompressHintSubEx<TeCurveCoeffs> {
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

            let modulus_mod_8 = BigUint::from(7u8) & curve.modulus.clone();
            if modulus_mod_8 != BigUint::from(5u8) {
                bail!(
                            "Currently only supporting twisted Edwards curves with modulus congruent to 5 mod 8."
                        );
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

            let mut y_limbs: Vec<u8> = Vec::with_capacity(num_limbs);

            for i in 0..num_limbs {
                let limb = memory.unsafe_read_cell(
                    F::from_canonical_u32(RV32_MEMORY_AS),
                    F::from_canonical_u32(rs1 + i as u32),
                );
                y_limbs.push(limb.as_canonical_u32() as u8);
            }
            let y = BigUint::from_bytes_le(&y_limbs);
            let rs2 = unsafe_read_rv32_register(memory, b);
            let rec_id = memory.unsafe_read_cell(
                F::from_canonical_u32(RV32_MEMORY_AS),
                F::from_canonical_u32(rs2),
            );

            let x = decompress_te_point(y, rec_id.as_canonical_u32() & 1 == 1, curve)?;

            let x_bytes = x
                .to_bytes_le()
                .into_iter()
                .map(F::from_canonical_u8)
                .chain(repeat(F::ZERO))
                .take(num_limbs)
                .collect();

            streams.hint_stream = x_bytes;

            Ok(())
        }
    }

    fn decompress_te_point(
        y: BigUint,
        is_x_odd: bool,
        curve: &CurveConfig<TeCurveCoeffs>,
    ) -> eyre::Result<BigUint> {
        let u: BigUint = (&y * &y - 1u8) % &curve.modulus;
        let v: BigUint = (&curve.coeffs.d * &y * &y - &curve.coeffs.a) % &curve.modulus;
        let exponent = (&curve.modulus - 5u8) >> 3;
        let mut x = (&u * &v * &v * &v) % &curve.modulus
            * (&u * &v.modpow(&7u8.into(), &curve.modulus) % &curve.modulus)
                .modpow(&exponent, &curve.modulus)
            % &curve.modulus;
        if (&v * &x * &x + &u).is_multiple_of(&curve.modulus) {
            x = &x
                * BigUint::from_u32(2)
                    .unwrap()
                    .modpow(&((&curve.modulus - BigUint::one()) >> 2), &curve.modulus)
                % &curve.modulus;
        }
        if x == BigUint::one() && is_x_odd {
            bail!("decoding twisted Edwards point failed");
        } else if is_x_odd != x.is_odd() {
            x = &curve.modulus - &x;
        }
        Ok(x)
    }
}
