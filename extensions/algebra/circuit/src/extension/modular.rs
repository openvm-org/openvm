use std::array;

use num_bigint::BigUint;
use num_traits::{FromPrimitive, One};
use openvm_algebra_transpiler::{ModularPhantom, Rv64ModularArithmeticOpcode};
use openvm_algebra_utils::{find_non_qr, NQR_RNG_SEED};
use openvm_circuit::{
    self,
    arch::{
        to_byte_ptr_bits, AirInventory, AirInventoryError, ChipInventory, ChipInventoryError,
        ExecutionBridge, ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena,
        VmCircuitExtension, VmExecutionExtension, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::bigint::utils::big_uint_to_limbs;
use openvm_cpu_backend::{CpuBackend, CpuDevice};
use openvm_instructions::{LocalOpcode, PhantomDiscriminant, VmOpcode};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_riscv_adapters::{
    Rv64IsEqualModU16AdapterAir, Rv64IsEqualModU16AdapterExecutor, Rv64IsEqualModU16AdapterFiller,
};
use openvm_riscv_circuit::adapters::U16_BITS;
use openvm_stark_backend::{p3_field::PrimeField32, StarkEngine, StarkProtocolConfig, Val};
#[cfg(feature = "rvr")]
use rvr_openvm_ext_algebra::ModularRvrExtension;
#[cfg(feature = "rvr")]
use rvr_openvm_lift::{RvrExtensionCtx, RvrExtensions, VmRvrExtension};
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use strum::EnumCount;

use crate::{
    modular_chip::{
        get_modular_addsub_air, get_modular_addsub_chip, get_modular_addsub_executor,
        get_modular_muldiv_air, get_modular_muldiv_chip, get_modular_muldiv_executor, ModularAir,
        ModularExecutor, ModularIsEqualCoreAir, ModularIsEqualFiller, ModularIsEqualU16Air,
        ModularIsEqualU16Chip, VmModularIsEqualU16Executor,
    },
    AlgebraCpuProverExt, MODULAR_BLOCKS_32, MODULAR_BLOCKS_48, NUM_LIMBS_32, NUM_LIMBS_32_U16,
    NUM_LIMBS_48, NUM_LIMBS_48_U16,
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
            .map(|modulus| format!("\"{modulus}\""))
            .collect::<Vec<String>>()
            .join(", ");

        format!("openvm_algebra_guest::moduli_macros::moduli_init! {{ {supported_moduli} }}",)
    }
}

#[cfg(feature = "rvr")]
impl<F: PrimeField32> VmRvrExtension<F> for ModularExtension {
    fn extend_rvr(&self, extensions: &mut RvrExtensions, _ctx: Option<&RvrExtensionCtx>) {
        extensions.register_lifter(ModularRvrExtension::new(self.supported_moduli.clone()));
    }
}

#[derive(Clone, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum ModularExtensionExecutor {
    // 32 limbs prime
    ModularAddSubRv64_32(ModularExecutor<MODULAR_BLOCKS_32>), // ModularAddSub
    ModularMulDivRv64_32(ModularExecutor<MODULAR_BLOCKS_32>), // ModularMulDiv
    ModularIsEqualRv64_32(VmModularIsEqualU16Executor<MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>), /* ModularIsEqual */
    // 48 limbs prime
    ModularAddSubRv64_48(ModularExecutor<MODULAR_BLOCKS_48>), // ModularAddSub
    ModularMulDivRv64_48(ModularExecutor<MODULAR_BLOCKS_48>), // ModularMulDiv
    ModularIsEqualRv64_48(VmModularIsEqualU16Executor<MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>), /* ModularIsEqual */
}

impl VmExecutionExtension for ModularExtension {
    type Executor = ModularExtensionExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<ModularExtensionExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.pointer_max_bits());
        for (i, modulus) in self.supported_moduli.iter().enumerate() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8) as usize;
            let start_offset =
                Rv64ModularArithmeticOpcode::CLASS_OFFSET + i * Rv64ModularArithmeticOpcode::COUNT;
            let modulus_limbs_u16 = big_uint_to_limbs(modulus, U16_BITS);
            if bytes <= NUM_LIMBS_32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_32,
                    limb_bits: 8,
                };
                let addsub = get_modular_addsub_executor::<MODULAR_BLOCKS_32>(
                    config.clone(),
                    U16_BITS,
                    byte_ptr_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    ModularExtensionExecutor::ModularAddSubRv64_32(addsub),
                    ((Rv64ModularArithmeticOpcode::ADD as usize)
                        ..=(Rv64ModularArithmeticOpcode::SETUP_ADDSUB as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let muldiv = get_modular_muldiv_executor::<MODULAR_BLOCKS_32>(
                    config,
                    U16_BITS,
                    byte_ptr_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    ModularExtensionExecutor::ModularMulDivRv64_32(muldiv),
                    ((Rv64ModularArithmeticOpcode::MUL as usize)
                        ..=(Rv64ModularArithmeticOpcode::SETUP_MULDIV as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let modulus_limbs = array::from_fn(|i| {
                    if i < modulus_limbs_u16.len() {
                        modulus_limbs_u16[i] as u16
                    } else {
                        0
                    }
                });

                let is_eq = VmModularIsEqualU16Executor::new(
                    Rv64IsEqualModU16AdapterExecutor::new(byte_ptr_max_bits),
                    start_offset,
                    modulus_limbs,
                );

                inventory.add_executor(
                    ModularExtensionExecutor::ModularIsEqualRv64_32(is_eq),
                    ((Rv64ModularArithmeticOpcode::IS_EQ as usize)
                        ..=(Rv64ModularArithmeticOpcode::SETUP_ISEQ as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else if bytes <= NUM_LIMBS_48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_48,
                    limb_bits: 8,
                };
                let addsub = get_modular_addsub_executor::<MODULAR_BLOCKS_48>(
                    config.clone(),
                    U16_BITS,
                    byte_ptr_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    ModularExtensionExecutor::ModularAddSubRv64_48(addsub),
                    ((Rv64ModularArithmeticOpcode::ADD as usize)
                        ..=(Rv64ModularArithmeticOpcode::SETUP_ADDSUB as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let muldiv = get_modular_muldiv_executor::<MODULAR_BLOCKS_48>(
                    config,
                    U16_BITS,
                    byte_ptr_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    ModularExtensionExecutor::ModularMulDivRv64_48(muldiv),
                    ((Rv64ModularArithmeticOpcode::MUL as usize)
                        ..=(Rv64ModularArithmeticOpcode::SETUP_MULDIV as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let modulus_limbs = array::from_fn(|i| {
                    if i < modulus_limbs_u16.len() {
                        modulus_limbs_u16[i] as u16
                    } else {
                        0
                    }
                });

                let is_eq = VmModularIsEqualU16Executor::new(
                    Rv64IsEqualModU16AdapterExecutor::new(byte_ptr_max_bits),
                    start_offset,
                    modulus_limbs,
                );

                inventory.add_executor(
                    ModularExtensionExecutor::ModularIsEqualRv64_48(is_eq),
                    ((Rv64ModularArithmeticOpcode::IS_EQ as usize)
                        ..=(Rv64ModularArithmeticOpcode::SETUP_ISEQ as usize))
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

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for ModularExtension {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let range_checker_bus = inventory.range_checker().bus;
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.pointer_max_bits());
        for (i, modulus) in self.supported_moduli.iter().enumerate() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8) as usize;
            let start_offset =
                Rv64ModularArithmeticOpcode::CLASS_OFFSET + i * Rv64ModularArithmeticOpcode::COUNT;

            if bytes <= NUM_LIMBS_32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_32,
                    limb_bits: 8,
                };

                let addsub = get_modular_addsub_air::<MODULAR_BLOCKS_32>(
                    exec_bridge,
                    memory_bridge,
                    config.clone(),
                    range_checker_bus,
                    byte_ptr_max_bits,
                    start_offset,
                );
                inventory.add_air(addsub);

                let muldiv = get_modular_muldiv_air::<MODULAR_BLOCKS_32>(
                    exec_bridge,
                    memory_bridge,
                    config,
                    range_checker_bus,
                    byte_ptr_max_bits,
                    start_offset,
                );
                inventory.add_air(muldiv);

                let is_eq = ModularIsEqualU16Air::<MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>::new(
                    Rv64IsEqualModU16AdapterAir::new(
                        exec_bridge,
                        memory_bridge,
                        range_checker_bus,
                        byte_ptr_max_bits,
                    ),
                    ModularIsEqualCoreAir::new(modulus.clone(), range_checker_bus, start_offset),
                );
                inventory.add_air(is_eq);
            } else if bytes <= NUM_LIMBS_48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_48,
                    limb_bits: 8,
                };

                let addsub = get_modular_addsub_air::<MODULAR_BLOCKS_48>(
                    exec_bridge,
                    memory_bridge,
                    config.clone(),
                    range_checker_bus,
                    byte_ptr_max_bits,
                    start_offset,
                );
                inventory.add_air(addsub);

                let muldiv = get_modular_muldiv_air::<MODULAR_BLOCKS_48>(
                    exec_bridge,
                    memory_bridge,
                    config,
                    range_checker_bus,
                    byte_ptr_max_bits,
                    start_offset,
                );
                inventory.add_air(muldiv);

                let is_eq = ModularIsEqualU16Air::<MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>::new(
                    Rv64IsEqualModU16AdapterAir::new(
                        exec_bridge,
                        memory_bridge,
                        range_checker_bus,
                        byte_ptr_max_bits,
                    ),
                    ModularIsEqualCoreAir::new(modulus.clone(), range_checker_bus, start_offset),
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
impl<SC, E, RA> VmProverExtension<E, RA, ModularExtension> for AlgebraCpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        extension: &ModularExtension,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        for (i, modulus) in extension.supported_moduli.iter().enumerate() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8) as usize;
            let start_offset =
                Rv64ModularArithmeticOpcode::CLASS_OFFSET + i * Rv64ModularArithmeticOpcode::COUNT;

            let modulus_limbs_u16 = big_uint_to_limbs(modulus, U16_BITS);

            if bytes <= NUM_LIMBS_32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_32,
                    limb_bits: 8,
                };

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_32>>()?;
                let addsub = get_modular_addsub_chip::<Val<SC>, MODULAR_BLOCKS_32>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                inventory.add_executor_chip(addsub);

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_32>>()?;
                let muldiv = get_modular_muldiv_chip::<Val<SC>, MODULAR_BLOCKS_32>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                inventory.add_executor_chip(muldiv);

                let modulus_limbs = array::from_fn(|i| {
                    if i < modulus_limbs_u16.len() {
                        modulus_limbs_u16[i] as u16
                    } else {
                        0
                    }
                });
                inventory
                    .next_air::<ModularIsEqualU16Air<MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>>()?;
                let is_eq =
                    ModularIsEqualU16Chip::<Val<SC>, MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>::new(
                        ModularIsEqualFiller::new(
                            Rv64IsEqualModU16AdapterFiller::new(
                                byte_ptr_max_bits,
                                range_checker.clone(),
                            ),
                            start_offset,
                            modulus_limbs,
                            range_checker.clone(),
                        ),
                        mem_helper.clone(),
                    );
                inventory.add_executor_chip(is_eq);
            } else if bytes <= NUM_LIMBS_48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_48,
                    limb_bits: 8,
                };

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_48>>()?;
                let addsub = get_modular_addsub_chip::<Val<SC>, MODULAR_BLOCKS_48>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                inventory.add_executor_chip(addsub);

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_48>>()?;
                let muldiv = get_modular_muldiv_chip::<Val<SC>, MODULAR_BLOCKS_48>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                inventory.add_executor_chip(muldiv);

                let modulus_limbs = array::from_fn(|i| {
                    if i < modulus_limbs_u16.len() {
                        modulus_limbs_u16[i] as u16
                    } else {
                        0
                    }
                });
                inventory
                    .next_air::<ModularIsEqualU16Air<MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>>()?;
                let is_eq =
                    ModularIsEqualU16Chip::<Val<SC>, MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>::new(
                        ModularIsEqualFiller::new(
                            Rv64IsEqualModU16AdapterFiller::new(
                                byte_ptr_max_bits,
                                range_checker.clone(),
                            ),
                            start_offset,
                            modulus_limbs,
                            range_checker.clone(),
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
    use openvm_instructions::{
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_NUM_LIMBS},
        PhantomDiscriminant,
    };
    use openvm_riscv_circuit::adapters::read_rv64_register_as_u32;
    use rand::{rngs::StdRng, SeedableRng};

    use super::{find_non_qr, mod_sqrt, NQR_RNG_SEED};
    use crate::{NUM_LIMBS_32, NUM_LIMBS_48};

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
    impl PhantomSubExecutor for SqrtHintSubEx {
        fn phantom_execute(
            &self,
            memory: &GuestMemory,
            streams: &mut Streams,
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
            let bytes = modulus.bits().div_ceil(8) as usize;
            let num_limbs: usize = if bytes <= NUM_LIMBS_32 {
                NUM_LIMBS_32
            } else if bytes <= NUM_LIMBS_48 {
                NUM_LIMBS_48
            } else {
                bail!("Modulus too large")
            };

            let rs1: u32 = read_rv64_register_as_u32(memory, a);
            // SAFETY:
            // - MEMORY_AS consists of `u8`s
            // - MEMORY_AS is in bounds
            let x_limbs: Vec<u8> = unsafe {
                memory
                    .memory
                    .get_u8_slice(RV64_MEMORY_AS, rs1 as usize, num_limbs)
            }
            .to_vec();
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

            let hint_bytes = once(u8::from(success))
                .chain(repeat(0u8))
                .take(RV64_REGISTER_NUM_LIMBS)
                .chain(
                    sqrt.to_bytes_le()
                        .into_iter()
                        .chain(repeat(0u8))
                        .take(num_limbs),
                )
                .collect();
            streams.hint_stream.set_hint(hint_bytes);
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
            let mut rng = StdRng::from_seed(NQR_RNG_SEED);
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

    impl PhantomSubExecutor for NonQrHintSubEx {
        fn phantom_execute(
            &self,
            _: &GuestMemory,
            streams: &mut Streams,
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

            let bytes = modulus.bits().div_ceil(8) as usize;
            let num_limbs: usize = if bytes <= NUM_LIMBS_32 {
                NUM_LIMBS_32
            } else if bytes <= NUM_LIMBS_48 {
                NUM_LIMBS_48
            } else {
                bail!("Modulus too large")
            };

            let hint_bytes = self.non_qrs[mod_idx]
                .to_bytes_le()
                .into_iter()
                .chain(repeat(0u8))
                .take(num_limbs)
                .collect();
            streams.hint_stream.set_hint(hint_bytes);
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
