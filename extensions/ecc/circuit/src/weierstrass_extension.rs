use derive_more::derive::From;
use hex_literal::hex;
use lazy_static::lazy_static;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, Zero};
use once_cell::sync::Lazy;
use openvm_circuit::{
    arch::{SystemPort, VmExtension, VmInventory, VmInventoryBuilder, VmInventoryError},
    system::phantom::PhantomChip,
};
use openvm_circuit_derive::{AnyEnum, InstructionExecutor};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_ecc_transpiler::{EccPhantom, Rv32WeierstrassOpcode};
use openvm_instructions::{LocalOpcode, PhantomDiscriminant, VmOpcode};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_rv32_adapters::Rv32VecHeapAdapterChip;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use strum::EnumCount;

use super::{EcAddNeChip, EcDoubleChip};

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
            .map(|curve_config| curve_config.struct_name.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        format!("openvm_ecc_guest::sw_macros::sw_init! {{ {supported_curves} }}")
    }
}

#[derive(Chip, ChipUsageGetter, InstructionExecutor, AnyEnum)]
pub enum WeierstrassExtensionExecutor<F: PrimeField32> {
    // 32 limbs prime
    EcAddNeRv32_32(EcAddNeChip<F, 2, 32>),
    EcDoubleRv32_32(EcDoubleChip<F, 2, 32>),
    // 48 limbs prime
    EcAddNeRv32_48(EcAddNeChip<F, 6, 16>),
    EcDoubleRv32_48(EcDoubleChip<F, 6, 16>),
}

#[derive(ChipUsageGetter, Chip, AnyEnum, From)]
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
        let ec_add_ne_opcodes = (Rv32WeierstrassOpcode::EC_ADD_NE as usize)
            ..=(Rv32WeierstrassOpcode::SETUP_EC_ADD_NE as usize);
        let ec_double_opcodes = (Rv32WeierstrassOpcode::EC_DOUBLE as usize)
            ..=(Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize);

        for (i, curve) in self.supported_curves.iter().enumerate() {
            let start_offset =
                Rv32WeierstrassOpcode::CLASS_OFFSET + i * Rv32WeierstrassOpcode::COUNT;
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
                let add_ne_chip = EcAddNeChip::new(
                    Rv32VecHeapAdapterChip::<F, 2, 2, 2, 32, 32>::new(
                        execution_bus,
                        program_bus,
                        memory_bridge,
                        pointer_bits,
                        bitwise_lu_chip.clone(),
                    ),
                    config32.clone(),
                    start_offset,
                    range_checker.clone(),
                    offline_memory.clone(),
                );
                inventory.add_executor(
                    WeierstrassExtensionExecutor::EcAddNeRv32_32(add_ne_chip),
                    ec_add_ne_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
                let double_chip = EcDoubleChip::new(
                    Rv32VecHeapAdapterChip::<F, 1, 2, 2, 32, 32>::new(
                        execution_bus,
                        program_bus,
                        memory_bridge,
                        pointer_bits,
                        bitwise_lu_chip.clone(),
                    ),
                    range_checker.clone(),
                    config32.clone(),
                    start_offset,
                    curve.a.clone(),
                    offline_memory.clone(),
                );
                inventory.add_executor(
                    WeierstrassExtensionExecutor::EcDoubleRv32_32(double_chip),
                    ec_double_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else if bytes <= 48 {
                let add_ne_chip = EcAddNeChip::new(
                    Rv32VecHeapAdapterChip::<F, 2, 6, 6, 16, 16>::new(
                        execution_bus,
                        program_bus,
                        memory_bridge,
                        pointer_bits,
                        bitwise_lu_chip.clone(),
                    ),
                    config48.clone(),
                    start_offset,
                    range_checker.clone(),
                    offline_memory.clone(),
                );
                inventory.add_executor(
                    WeierstrassExtensionExecutor::EcAddNeRv32_48(add_ne_chip),
                    ec_add_ne_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
                let double_chip = EcDoubleChip::new(
                    Rv32VecHeapAdapterChip::<F, 1, 6, 6, 16, 16>::new(
                        execution_bus,
                        program_bus,
                        memory_bridge,
                        pointer_bits,
                        bitwise_lu_chip.clone(),
                    ),
                    range_checker.clone(),
                    config48.clone(),
                    start_offset,
                    curve.a.clone(),
                    offline_memory.clone(),
                );
                inventory.add_executor(
                    WeierstrassExtensionExecutor::EcDoubleRv32_48(double_chip),
                    ec_double_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else {
                panic!("Modulus too large");
            }
        }
        let non_qr_hint_sub_ex = phantom::NonQrHintSubEx::new(self.supported_curves.clone());
        builder.add_phantom_sub_executor(
            non_qr_hint_sub_ex.clone(),
            PhantomDiscriminant(EccPhantom::HintNonQr as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::DecompressHintSubEx::new(non_qr_hint_sub_ex),
            PhantomDiscriminant(EccPhantom::HintDecompress as u16),
        )?;

        Ok(inventory)
    }
}

pub(crate) mod phantom {
    use std::{
        iter::{once, repeat},
        ops::Deref,
    };

    use eyre::bail;
    use num_bigint::BigUint;
    use num_integer::Integer;
    use num_traits::One;
    use openvm_algebra_circuit::{find_non_qr, mod_sqrt};
    use openvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::MemoryController,
    };
    use openvm_instructions::{riscv::RV32_MEMORY_AS, PhantomDiscriminant};
    use openvm_rv32im_circuit::adapters::unsafe_read_rv32_register;
    use openvm_stark_backend::p3_field::PrimeField32;

    use super::CurveConfig;

    // Hint for a decompression
    // if possible is true, then `sqrt` is the decompressed y-coordinate
    // if possible is false, then `sqrt` is such that
    // `sqrt^2 = rhs * non_qr` where `rhs` is the rhs of the curve equation
    pub struct DecompressionHint<T> {
        pub possible: bool,
        pub sqrt: T,
    }

    #[derive(derive_new::new)]
    pub struct DecompressHintSubEx(NonQrHintSubEx);

    impl Deref for DecompressHintSubEx {
        type Target = NonQrHintSubEx;

        fn deref(&self) -> &NonQrHintSubEx {
            &self.0
        }
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
            let hint = self.decompress_point(x, rec_id.as_canonical_u32() & 1 == 1, c_idx);
            let hint_bytes = once(F::from_bool(hint.possible))
                .chain(repeat(F::ZERO))
                .take(4)
                .chain(
                    hint.sqrt
                        .to_bytes_le()
                        .into_iter()
                        .map(F::from_canonical_u8)
                        .chain(repeat(F::ZERO))
                        .take(num_limbs),
                )
                .collect();
            streams.hint_stream = hint_bytes;
            Ok(())
        }
    }

    impl DecompressHintSubEx {
        /// Given `x` in the coordinate field of the curve, and the recovery id,
        /// return the unique `y` such that `(x, y)` is a point on the curve and
        /// `y` has the same parity as the recovery id.
        ///
        /// If no such `y` exists, return the square root of `(x^3 + ax + b) * non_qr`
        /// where `non_qr` is a quadratic nonresidue of the field.
        fn decompress_point(
            &self,
            x: BigUint,
            is_y_odd: bool,
            curve_idx: usize,
        ) -> DecompressionHint<BigUint> {
            let curve = &self.supported_curves[curve_idx];
            let alpha = ((&x * &x * &x) + (&x * &curve.a) + &curve.b) % &curve.modulus;
            match mod_sqrt(&alpha, &curve.modulus, &self.non_qrs[curve_idx]) {
                Some(beta) => {
                    if is_y_odd == beta.is_odd() {
                        DecompressionHint {
                            possible: true,
                            sqrt: beta,
                        }
                    } else {
                        DecompressionHint {
                            possible: true,
                            sqrt: &curve.modulus - &beta,
                        }
                    }
                }
                None => {
                    debug_assert_eq!(
                        self.non_qrs[curve_idx]
                            .modpow(&((&curve.modulus - BigUint::one()) >> 1), &curve.modulus),
                        &curve.modulus - BigUint::one()
                    );
                    let sqrt = mod_sqrt(
                        &(&alpha * &self.non_qrs[curve_idx]),
                        &curve.modulus,
                        &self.non_qrs[curve_idx],
                    )
                    .unwrap();
                    DecompressionHint {
                        possible: false,
                        sqrt,
                    }
                }
            }
        }
    }

    #[derive(Clone)]
    pub struct NonQrHintSubEx {
        pub supported_curves: Vec<CurveConfig>,
        pub non_qrs: Vec<BigUint>,
    }

    impl NonQrHintSubEx {
        pub fn new(supported_curves: Vec<CurveConfig>) -> Self {
            let non_qrs = supported_curves
                .iter()
                .map(|curve| find_non_qr(&curve.modulus))
                .collect();
            Self {
                supported_curves,
                non_qrs,
            }
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for NonQrHintSubEx {
        fn phantom_execute(
            &mut self,
            _: &MemoryController<F>,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            _: F,
            _: F,
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

            let num_limbs: usize = if curve.modulus.bits().div_ceil(8) <= 32 {
                32
            } else if curve.modulus.bits().div_ceil(8) <= 48 {
                48
            } else {
                bail!("Modulus too large")
            };

            let hint_bytes = self.non_qrs[c_idx]
                .to_bytes_le()
                .into_iter()
                .map(F::from_canonical_u8)
                .chain(repeat(F::ZERO))
                .take(num_limbs)
                .collect();
            streams.hint_stream = hint_bytes;
            Ok(())
        }
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
