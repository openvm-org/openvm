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
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_ecc_guest::{
    k256::{SECP256K1_MODULUS, SECP256K1_ORDER},
    p256::{CURVE_A as P256_A, CURVE_B as P256_B, P256_MODULUS, P256_ORDER},
};
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
    modulus: SECP256K1_MODULUS.clone(),
    scalar: SECP256K1_ORDER.clone(),
    a: BigUint::zero(),
    b: BigUint::from_u8(7u8).unwrap(),
});

pub static P256_CONFIG: Lazy<CurveConfig> = Lazy::new(|| CurveConfig {
    modulus: P256_MODULUS.clone(),
    scalar: P256_ORDER.clone(),
    a: BigUint::from_bytes_le(P256_A.as_le_bytes()),
    b: BigUint::from_bytes_le(P256_B.as_le_bytes()),
});

#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct WeierstrassExtension {
    pub supported_curves: Vec<CurveConfig>,
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
    use num_bigint::{BigUint, RandBigInt};
    use num_integer::Integer;
    use num_traits::{FromPrimitive, One};
    use openvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::MemoryController,
    };
    use openvm_ecc_guest::weierstrass::DecompressionHint;
    use openvm_instructions::{riscv::RV32_MEMORY_AS, PhantomDiscriminant};
    use openvm_rv32im_circuit::adapters::unsafe_read_rv32_register;
    use openvm_stark_backend::p3_field::PrimeField32;
    use rand::{rngs::StdRng, SeedableRng};

    use super::CurveConfig;

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

    /// Find the square root of `x` modulo `modulus` with `non_qr` a
    /// quadratic nonresidue of the field.
    pub fn mod_sqrt(x: &BigUint, modulus: &BigUint, non_qr: &BigUint) -> Option<BigUint> {
        if modulus % 4u32 == BigUint::from_u8(3).unwrap() {
            // x^(1/2) = x^((p+1)/4) when p = 3 mod 4
            let exponent = (modulus + BigUint::one()) >> 2;
            let ret = x.modpow(&exponent, modulus);
            if &ret * &ret % modulus == x % modulus {
                Some(ret)
            } else {
                None
            }
        } else {
            // Tonelli-Shanks algorithm
            // https://en.wikipedia.org/wiki/Tonelli%E2%80%93Shanks_algorithm#The_algorithm
            let mut q = modulus - BigUint::one();
            let mut s = 0;
            while &q % 2u32 == BigUint::ZERO {
                s += 1;
                q /= 2u32;
            }
            let z = non_qr;
            let mut m = s;
            let mut c = z.modpow(&q, modulus);
            let mut t = x.modpow(&q, modulus);
            let mut r = x.modpow(&((q + BigUint::one()) >> 1), modulus);
            loop {
                if t == BigUint::ZERO {
                    return Some(BigUint::ZERO);
                }
                if t == BigUint::one() {
                    return Some(r);
                }
                let mut i = 0;
                let mut tmp = t.clone();
                while tmp != BigUint::one() && i < m {
                    tmp = &tmp * &tmp % modulus;
                    i += 1;
                }
                if i == m {
                    // self is not a quadratic residue
                    return None;
                }
                for _ in 0..m - i - 1 {
                    c = &c * &c % modulus;
                }
                let b = c;
                m = i;
                c = &b * &b % modulus;
                t = ((t * &b % modulus) * &b) % modulus;
                r = (r * b) % modulus;
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

    // Returns a non-quadratic residue in the field
    fn find_non_qr(modulus: &BigUint) -> BigUint {
        if modulus % 4u32 == BigUint::from(3u8) {
            // p = 3 mod 4 then -1 is a quadratic residue
            modulus - BigUint::one()
        } else if modulus % 8u32 == BigUint::from(5u8) {
            // p = 5 mod 8 then 2 is a non-quadratic residue
            // since 2^((p-1)/2) = (-1)^((p^2-1)/8)
            BigUint::from_u8(2u8).unwrap()
        } else {
            let mut rng = StdRng::from_entropy();
            let mut non_qr = rng.gen_biguint_range(
                &BigUint::from_u8(2).unwrap(),
                &(modulus - BigUint::from_u8(1).unwrap()),
            );
            // To check if non_qr is a quadratic nonresidue, we compute non_qr^((p-1)/2)
            // If the result is p-1, then non_qr is a quadratic nonresidue
            // Otherwise, non_qr is a quadratic residue
            let exponent = (modulus - BigUint::one()) >> 1;
            while non_qr.modpow(&exponent, modulus) != modulus - BigUint::one() {
                non_qr = rng.gen_biguint_range(
                    &BigUint::from_u8(2).unwrap(),
                    &(modulus - BigUint::from_u8(1).unwrap()),
                );
            }
            non_qr
        }
    }
}
