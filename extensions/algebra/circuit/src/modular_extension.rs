use std::array;

use derive_more::derive::From;
use num_bigint::{BigUint, RandBigInt};
use num_traits::{FromPrimitive, One};
use openvm_algebra_transpiler::{ModularPhantom, Rv32ModularArithmeticOpcode};
use openvm_circuit::{
    self,
    arch::{
        ExecutionBridge, SystemPort, VmExtension, VmInventory, VmInventoryBuilder, VmInventoryError,
    },
    system::phantom::PhantomChip,
};
use openvm_circuit_derive::{AnyEnum, InsExecutorE1, InsExecutorE2, InstructionExecutor};
use openvm_circuit_primitives::{
    bigint::utils::big_uint_to_limbs,
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::{LocalOpcode, PhantomDiscriminant, VmOpcode};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_rv32_adapters::{Rv32IsEqualModAdapterAir, Rv32IsEqualModeAdapterStep};
use openvm_stark_backend::p3_field::PrimeField32;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use strum::EnumCount;

use crate::modular_chip::{
    ModularAddSubChip, ModularIsEqualAir, ModularIsEqualChip, ModularIsEqualCoreAir,
    ModularMulDivChip, VmModularIsEqualStep,
};

// TODO: this should be decided after e2 execution

#[serde_as]
#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct ModularExtension {
    #[serde_as(as = "Vec<DisplayFromStr>")]
    pub supported_moduli: Vec<BigUint>,
}

impl ModularExtension {
    // Generates a call to the moduli_init! macro with moduli in the correct order
    pub fn generate_moduli_init(&self) -> String {
        let supported_moduli = self
            .supported_moduli
            .iter()
            .map(|modulus| format!("\"{}\"", modulus))
            .collect::<Vec<String>>()
            .join(", ");

        format!("openvm_algebra_guest::moduli_macros::moduli_init! {{ {supported_moduli} }}",)
    }
}

#[derive(
    ChipUsageGetter, Chip, InstructionExecutor, AnyEnum, From, InsExecutorE1, InsExecutorE2,
)]
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

        for (i, modulus) in self.supported_moduli.iter().enumerate() {
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

            let modulus_limbs = big_uint_to_limbs(modulus, 8);

            if bytes <= 32 {
                let addsub_chip = ModularAddSubChip::new(
                    execution_bridge,
                    memory_bridge,
                    builder.system_base().memory_controller.helper(),
                    pointer_max_bits,
                    config32.clone(),
                    start_offset,
                    bitwise_lu_chip.clone(),
                    range_checker.clone(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularAddSubRv32_32(addsub_chip),
                    addsub_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
                let muldiv_chip = ModularMulDivChip::new(
                    execution_bridge,
                    memory_bridge,
                    builder.system_base().memory_controller.helper(),
                    pointer_max_bits,
                    config32.clone(),
                    start_offset,
                    bitwise_lu_chip.clone(),
                    range_checker.clone(),
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
                            execution_bridge,
                            memory_bridge,
                            bitwise_lu_chip.bus(),
                            pointer_max_bits,
                        ),
                        ModularIsEqualCoreAir::new(
                            modulus.clone(),
                            bitwise_lu_chip.bus(),
                            start_offset,
                        ),
                    ),
                    VmModularIsEqualStep::new(
                        Rv32IsEqualModeAdapterStep::new(pointer_max_bits, bitwise_lu_chip.clone()),
                        modulus_limbs,
                        start_offset,
                        bitwise_lu_chip.clone(),
                    ),
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
                    execution_bridge,
                    memory_bridge,
                    builder.system_base().memory_controller.helper(),
                    pointer_max_bits,
                    config48.clone(),
                    start_offset,
                    bitwise_lu_chip.clone(),
                    range_checker.clone(),
                );
                inventory.add_executor(
                    ModularExtensionExecutor::ModularAddSubRv32_48(addsub_chip),
                    addsub_opcodes
                        .clone()
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
                let muldiv_chip = ModularMulDivChip::new(
                    execution_bridge,
                    memory_bridge,
                    builder.system_base().memory_controller.helper(),
                    pointer_max_bits,
                    config48.clone(),
                    start_offset,
                    bitwise_lu_chip.clone(),
                    range_checker.clone(),
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
                            execution_bridge,
                            memory_bridge,
                            bitwise_lu_chip.bus(),
                            pointer_max_bits,
                        ),
                        ModularIsEqualCoreAir::new(
                            modulus.clone(),
                            bitwise_lu_chip.bus(),
                            start_offset,
                        ),
                    ),
                    VmModularIsEqualStep::new(
                        Rv32IsEqualModeAdapterStep::new(pointer_max_bits, bitwise_lu_chip.clone()),
                        modulus_limbs,
                        start_offset,
                        bitwise_lu_chip.clone(),
                    ),
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
        let non_qr_hint_sub_ex = phantom::NonQrHintSubEx::new(self.supported_moduli.clone());
        builder.add_phantom_sub_executor(
            non_qr_hint_sub_ex.clone(),
            PhantomDiscriminant(ModularPhantom::HintNonQr as u16),
        )?;

        let sqrt_hint_sub_ex = phantom::SqrtHintSubEx::new(non_qr_hint_sub_ex);
        builder.add_phantom_sub_executor(
            sqrt_hint_sub_ex,
            PhantomDiscriminant(ModularPhantom::HintSqrt as u16),
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
    use openvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::online::GuestMemory,
    };
    use openvm_instructions::{riscv::RV32_MEMORY_AS, PhantomDiscriminant};
    use openvm_rv32im_circuit::adapters::read_rv32_register;
    use openvm_stark_backend::p3_field::PrimeField32;
    use rand::{rngs::StdRng, SeedableRng};

    use super::{find_non_qr, mod_sqrt};

    #[derive(derive_new::new)]
    pub struct SqrtHintSubEx(NonQrHintSubEx);

    impl Deref for SqrtHintSubEx {
        type Target = NonQrHintSubEx;

        fn deref(&self) -> &NonQrHintSubEx {
            &self.0
        }
    }

    // Given x returns either a sqrt of x or a sqrt of x * non_qr, whichever exists.
    // Note that non_qr is fixed for each modulus.
    impl<F: PrimeField32> PhantomSubExecutor<F> for SqrtHintSubEx {
        fn phantom_execute(
            &self,
            memory: &GuestMemory,
            streams: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            a: u32,
            _: u32,
            c_upper: u16,
        ) -> eyre::Result<()> {
            let mod_idx = c_upper as usize;
            if mod_idx >= self.supported_moduli.len() {
                bail!(
                    "Modulus index {mod_idx} out of range: {} supported moduli",
                    self.supported_moduli.len()
                );
            }
            let modulus = &self.supported_moduli[mod_idx];
            let num_limbs: usize = if modulus.bits().div_ceil(8) <= 32 {
                32
            } else if modulus.bits().div_ceil(8) <= 48 {
                48
            } else {
                bail!("Modulus too large")
            };

            let rs1 = read_rv32_register(memory, a);
            // SAFETY:
            // - MEMORY_AS consists of `u8`s
            // - MEMORY_AS is in bounds
            let x_limbs: Vec<u8> =
                unsafe { memory.memory.get_slice((RV32_MEMORY_AS, rs1), num_limbs) }.to_vec();
            let x = BigUint::from_bytes_le(&x_limbs);

            let (success, sqrt) = match mod_sqrt(&x, modulus, &self.non_qrs[mod_idx]) {
                Some(sqrt) => (true, sqrt),
                None => {
                    let sqrt = mod_sqrt(
                        &(&x * &self.non_qrs[mod_idx]),
                        modulus,
                        &self.non_qrs[mod_idx],
                    )
                    .expect("Either x or x * non_qr should be a square");
                    (false, sqrt)
                }
            };

            let hint_bytes = once(F::from_bool(success))
                .chain(repeat(F::ZERO))
                .take(4)
                .chain(
                    sqrt.to_bytes_le()
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

    #[derive(Clone)]
    pub struct NonQrHintSubEx {
        pub supported_moduli: Vec<BigUint>,
        pub non_qrs: Vec<BigUint>,
    }

    impl NonQrHintSubEx {
        pub fn new(supported_moduli: Vec<BigUint>) -> Self {
            // Use deterministic seed so that the non-QR are deterministic between different
            // instances of the VM. The seed determines the runtime of Tonelli-Shanks, if the
            // algorithm is necessary, which affects the time it takes to construct and initialize
            // the VM but does not affect the runtime.
            let mut rng = StdRng::from_seed([0u8; 32]);
            let non_qrs = supported_moduli
                .iter()
                .map(|modulus| find_non_qr(modulus, &mut rng))
                .collect();
            Self {
                supported_moduli,
                non_qrs,
            }
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for NonQrHintSubEx {
        fn phantom_execute(
            &self,
            _: &GuestMemory,
            streams: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            _: u32,
            _: u32,
            c_upper: u16,
        ) -> eyre::Result<()> {
            let mod_idx = c_upper as usize;
            if mod_idx >= self.supported_moduli.len() {
                bail!(
                    "Modulus index {mod_idx} out of range: {} supported moduli",
                    self.supported_moduli.len()
                );
            }
            let modulus = &self.supported_moduli[mod_idx];

            let num_limbs: usize = if modulus.bits().div_ceil(8) <= 32 {
                32
            } else if modulus.bits().div_ceil(8) <= 48 {
                48
            } else {
                bail!("Modulus too large")
            };

            let hint_bytes = self.non_qrs[mod_idx]
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

// Returns a non-quadratic residue in the field
pub fn find_non_qr(modulus: &BigUint, rng: &mut impl Rng) -> BigUint {
    if modulus % 4u32 == BigUint::from(3u8) {
        // p = 3 mod 4 then -1 is a quadratic residue
        modulus - BigUint::one()
    } else if modulus % 8u32 == BigUint::from(5u8) {
        // p = 5 mod 8 then 2 is a non-quadratic residue
        // since 2^((p-1)/2) = (-1)^((p^2-1)/8)
        BigUint::from_u8(2u8).unwrap()
    } else {
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
