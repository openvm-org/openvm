use std::{array, sync::Arc};

use num_bigint::{BigUint, RandBigInt};
use num_traits::{FromPrimitive, One};
use openvm_algebra_transpiler::{ModularPhantom, Rv32ModularArithmeticOpcode};
use openvm_circuit::{
    self,
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena, VmCircuitExtension,
        VmExecutionExtension, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::{
    bigint::utils::big_uint_to_limbs,
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    var_range::VariableRangeCheckerBus,
};
use openvm_instructions::{LocalOpcode, PhantomDiscriminant, VmOpcode};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_rv32_adapters::{
    Rv32IsEqualModAdapterAir, Rv32IsEqualModAdapterExecutor, Rv32IsEqualModAdapterFiller,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_stark_sdk::engine::StarkEngine;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use strum::EnumCount;

use crate::{
    modular_chip::{
        get_modular_addsub_air, get_modular_addsub_chip, get_modular_addsub_step,
        get_modular_muldiv_air, get_modular_muldiv_chip, get_modular_muldiv_step, ModularAir,
        ModularExecutor, ModularIsEqualAir, ModularIsEqualChip, ModularIsEqualCoreAir,
        ModularIsEqualFiller, VmModularIsEqualExecutor,
    },
    AlgebraCpuProverExt,
};

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

#[derive(Clone, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum ModularExtensionExecutor {
    // 32 limbs prime
    ModularAddSubRv32_32(ModularExecutor<1, 32>), // ModularAddSub
    ModularMulDivRv32_32(ModularExecutor<1, 32>), // ModularMulDiv
    ModularIsEqualRv32_32(VmModularIsEqualExecutor<1, 32, 32>), // ModularIsEqual
    // 48 limbs prime
    ModularAddSubRv32_48(ModularExecutor<3, 16>), // ModularAddSub
    ModularMulDivRv32_48(ModularExecutor<3, 16>), // ModularMulDiv
    ModularIsEqualRv32_48(VmModularIsEqualExecutor<3, 16, 48>), // ModularIsEqual
}

impl<F: PrimeField32> VmExecutionExtension<F> for ModularExtension {
    type Executor = ModularExtensionExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, ModularExtensionExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();
        // TODO: somehow get the range checker bus from `ExecutorInventory`
        let dummy_range_checker_bus = VariableRangeCheckerBus::new(u16::MAX, 16);
        for (i, modulus) in self.supported_moduli.iter().enumerate() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8);
            let start_offset =
                Rv32ModularArithmeticOpcode::CLASS_OFFSET + i * Rv32ModularArithmeticOpcode::COUNT;
            let modulus_limbs = big_uint_to_limbs(modulus, 8);
            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };
                let addsub = get_modular_addsub_step(
                    config.clone(),
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    ModularExtensionExecutor::ModularAddSubRv32_32(addsub),
                    ((Rv32ModularArithmeticOpcode::ADD as usize)
                        ..=(Rv32ModularArithmeticOpcode::SETUP_ADDSUB as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let muldiv = get_modular_muldiv_step(
                    config,
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    ModularExtensionExecutor::ModularMulDivRv32_32(muldiv),
                    ((Rv32ModularArithmeticOpcode::MUL as usize)
                        ..=(Rv32ModularArithmeticOpcode::SETUP_MULDIV as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let modulus_limbs = array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u8
                    } else {
                        0
                    }
                });

                let is_eq = VmModularIsEqualExecutor::new(
                    Rv32IsEqualModAdapterExecutor::new(pointer_max_bits),
                    start_offset,
                    modulus_limbs,
                );

                inventory.add_executor(
                    ModularExtensionExecutor::ModularIsEqualRv32_32(is_eq),
                    ((Rv32ModularArithmeticOpcode::IS_EQ as usize)
                        ..=(Rv32ModularArithmeticOpcode::SETUP_ISEQ as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };
                let addsub = get_modular_addsub_step(
                    config.clone(),
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    ModularExtensionExecutor::ModularAddSubRv32_48(addsub),
                    ((Rv32ModularArithmeticOpcode::ADD as usize)
                        ..=(Rv32ModularArithmeticOpcode::SETUP_ADDSUB as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let muldiv = get_modular_muldiv_step(
                    config,
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    ModularExtensionExecutor::ModularMulDivRv32_48(muldiv),
                    ((Rv32ModularArithmeticOpcode::MUL as usize)
                        ..=(Rv32ModularArithmeticOpcode::SETUP_MULDIV as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let modulus_limbs = array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u8
                    } else {
                        0
                    }
                });

                let is_eq = VmModularIsEqualExecutor::new(
                    Rv32IsEqualModAdapterExecutor::new(pointer_max_bits),
                    start_offset,
                    modulus_limbs,
                );

                inventory.add_executor(
                    ModularExtensionExecutor::ModularIsEqualRv32_48(is_eq),
                    ((Rv32ModularArithmeticOpcode::IS_EQ as usize)
                        ..=(Rv32ModularArithmeticOpcode::SETUP_ISEQ as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else {
                panic!("Modulus too large");
            }
        }

        let non_qr_hint_sub_ex = phantom::NonQrHintSubEx::new(self.supported_moduli.clone());
        inventory.add_phantom_sub_executor(
            non_qr_hint_sub_ex.clone(),
            PhantomDiscriminant(ModularPhantom::HintNonQr as u16),
        )?;

        let sqrt_hint_sub_ex = phantom::SqrtHintSubEx::new(non_qr_hint_sub_ex);
        inventory.add_phantom_sub_executor(
            sqrt_hint_sub_ex,
            PhantomDiscriminant(ModularPhantom::HintSqrt as u16),
        )?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for ModularExtension {
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
        for (i, modulus) in self.supported_moduli.iter().enumerate() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8);
            let start_offset =
                Rv32ModularArithmeticOpcode::CLASS_OFFSET + i * Rv32ModularArithmeticOpcode::COUNT;

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                let addsub = get_modular_addsub_air::<1, 32>(
                    exec_bridge,
                    memory_bridge,
                    config.clone(),
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                );
                inventory.add_air(addsub);

                let muldiv = get_modular_muldiv_air::<1, 32>(
                    exec_bridge,
                    memory_bridge,
                    config,
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                );
                inventory.add_air(muldiv);

                let is_eq = ModularIsEqualAir::<1, 32, 32>::new(
                    Rv32IsEqualModAdapterAir::new(
                        exec_bridge,
                        memory_bridge,
                        bitwise_lu,
                        pointer_max_bits,
                    ),
                    ModularIsEqualCoreAir::new(modulus.clone(), bitwise_lu, start_offset),
                );
                inventory.add_air(is_eq);
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };

                let addsub = get_modular_addsub_air::<3, 16>(
                    exec_bridge,
                    memory_bridge,
                    config.clone(),
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                );
                inventory.add_air(addsub);

                let muldiv = get_modular_muldiv_air::<3, 16>(
                    exec_bridge,
                    memory_bridge,
                    config,
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                );
                inventory.add_air(muldiv);

                let is_eq = ModularIsEqualAir::<3, 16, 48>::new(
                    Rv32IsEqualModAdapterAir::new(
                        exec_bridge,
                        memory_bridge,
                        bitwise_lu,
                        pointer_max_bits,
                    ),
                    ModularIsEqualCoreAir::new(modulus.clone(), bitwise_lu, start_offset),
                );
                inventory.add_air(is_eq);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, ModularExtension> for AlgebraCpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        extension: &ModularExtension,
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
        for (i, modulus) in extension.supported_moduli.iter().enumerate() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8);
            let start_offset =
                Rv32ModularArithmeticOpcode::CLASS_OFFSET + i * Rv32ModularArithmeticOpcode::COUNT;

            let modulus_limbs = big_uint_to_limbs(modulus, 8);

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                inventory.next_air::<ModularAir<1, 32>>()?;
                let addsub = get_modular_addsub_chip::<Val<SC>, 1, 32>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(addsub);

                inventory.next_air::<ModularAir<1, 32>>()?;
                let muldiv = get_modular_muldiv_chip::<Val<SC>, 1, 32>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(muldiv);

                let modulus_limbs = array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u8
                    } else {
                        0
                    }
                });
                inventory.next_air::<ModularIsEqualAir<1, 32, 32>>()?;
                let is_eq = ModularIsEqualChip::<Val<SC>, 1, 32, 32>::new(
                    ModularIsEqualFiller::new(
                        Rv32IsEqualModAdapterFiller::new(pointer_max_bits, bitwise_lu.clone()),
                        start_offset,
                        modulus_limbs,
                        bitwise_lu.clone(),
                    ),
                    mem_helper.clone(),
                );
                inventory.add_executor_chip(is_eq);
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };

                inventory.next_air::<ModularAir<3, 16>>()?;
                let addsub = get_modular_addsub_chip::<Val<SC>, 3, 16>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(addsub);

                inventory.next_air::<ModularAir<3, 16>>()?;
                let muldiv = get_modular_muldiv_chip::<Val<SC>, 3, 16>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(muldiv);

                let modulus_limbs = array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u8
                    } else {
                        0
                    }
                });
                inventory.next_air::<ModularIsEqualAir<3, 16, 48>>()?;
                let is_eq = ModularIsEqualChip::<Val<SC>, 3, 16, 48>::new(
                    ModularIsEqualFiller::new(
                        Rv32IsEqualModAdapterFiller::new(pointer_max_bits, bitwise_lu.clone()),
                        start_offset,
                        modulus_limbs,
                        bitwise_lu.clone(),
                    ),
                    mem_helper.clone(),
                );
                inventory.add_executor_chip(is_eq);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
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
