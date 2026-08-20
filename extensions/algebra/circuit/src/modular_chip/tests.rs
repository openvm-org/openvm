#[cfg(all(feature = "cuda", feature = "rvr"))]
use std::sync::Arc;
use std::{borrow::BorrowMut, str::FromStr};

use num_bigint::BigUint;
use num_traits::Zero;
use openvm_algebra_transpiler::ModularArithmeticOpcode;
use openvm_circuit::arch::{
    instructions::LocalOpcode,
    testing::{
        memory::gen_pointer, TestBuilder, TestChipHarness, TestPreflight, VmChipTestBuilder,
    },
    BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES,
};
use openvm_circuit_primitives::bigint::utils::{secp256k1_coord_prime, secp256k1_scalar_prime};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_cuda_common::copy::MemCopyD2H;
use openvm_instructions::{
    instruction::Instruction,
    riscv::{MEMORY_AS, REGISTER_AS},
    VmOpcode,
};
use openvm_mod_circuit_builder::{
    test_utils::{generate_field_element, generate_random_biguint},
    utils::biguint_to_limbs_vec,
    ExprBuilderConfig,
};
use openvm_pairing_guest::{bls12_381::BLS12_381_MODULUS, bn254::BN254_MODULUS};
use openvm_riscv_adapters::{write_ptr_reg, write_u16_heap_default};
use openvm_riscv_circuit::adapters::REGISTER_NUM_LIMBS;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::extension::{HybridModularChip, HybridModularIsEqualChip},
    openvm_circuit::arch::testing::{
        default_var_range_checker_bus, GpuChipTestBuilder, GpuTestChipHarness,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
};

use crate::{
    modular_chip::{
        get_modular_addsub_air, get_modular_addsub_chip, get_modular_addsub_executor,
        get_modular_muldiv_air, get_modular_muldiv_chip, get_modular_muldiv_executor, ModularAir,
        ModularChip, ModularExecutor, ModularIsEqualCoreAir, ModularIsEqualCoreCols,
        ModularIsEqualFiller, ModularIsEqualU16Air, ModularIsEqualU16Chip,
        VmModularIsEqualU16Executor,
    },
    MODULAR_BLOCKS_32, MODULAR_BLOCKS_48, NUM_LIMBS_32, NUM_LIMBS_32_U16, NUM_LIMBS_48,
    NUM_LIMBS_48_U16,
};

const LIMB_BITS: usize = 8;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

#[cfg(test)]
mod addsub_tests {
    use openvm_circuit::arch::{MemoryConfig, Postflight};
    use openvm_instructions::program::Program;

    use super::*;
    use crate::trace::generate_field_expression_trace_from_postflight;

    const ADD_LOCAL: usize = ModularArithmeticOpcode::ADD as usize;

    type Harness<const BLOCKS: usize> =
        TestChipHarness<F, ModularExecutor<BLOCKS>, ModularAir<BLOCKS>, ModularChip<F, BLOCKS>>;

    fn create_harness<const BLOCKS: usize>(
        tester: &VmChipTestBuilder<F>,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> Harness<BLOCKS> {
        let air = get_modular_addsub_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            tester.range_checker().bus(),
            tester.address_bits(),
            offset,
        );
        let executor = get_modular_addsub_executor(
            config.clone(),
            tester.range_checker().bus().range_max_bits,
            offset,
        );
        let chip = get_modular_addsub_chip(
            config,
            tester.memory_helper(),
            tester.range_checker(),
            tester.address_bits(),
        );
        let address_bits = tester.address_bits();
        Harness::with_capacity(
            executor,
            air,
            chip,
            MAX_INS_CAPACITY,
            move |chip, postflight| {
                generate_field_expression_trace_from_postflight::<_, BLOCKS>(
                    chip,
                    postflight,
                    offset,
                    address_bits,
                )
            },
        )
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    type GpuHarness<const BLOCKS: usize> = GpuTestChipHarness<
        F,
        ModularExecutor<BLOCKS>,
        ModularAir<BLOCKS>,
        HybridModularChip<F, BLOCKS>,
        ModularChip<F, BLOCKS>,
    >;

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn create_cuda_harness<const BLOCKS: usize>(
        tester: &GpuChipTestBuilder,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> GpuHarness<BLOCKS> {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let air = get_modular_addsub_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            range_bus,
            tester.address_bits(),
            offset,
        );
        let executor =
            get_modular_addsub_executor(config.clone(), range_bus.range_max_bits, offset);

        let cpu_chip = get_modular_addsub_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            tester.address_bits(),
        );

        // Use hybrid chip wrapping the CPU chip
        let replay_modulus = config.modulus.clone();
        let replay_cpu_chip = get_modular_addsub_chip(
            config,
            tester.cpu_memory_helper(),
            tester.cpu_range_checker(),
            tester.address_bits(),
        );
        let hybrid_chip = HybridModularChip::new_addsub_with_replay(
            replay_cpu_chip,
            tester.range_checker().device_ctx.clone(),
            &replay_modulus,
            offset,
            tester.address_bits(),
            tester.timestamp_max_bits(),
            tester.range_checker(),
        )
        .unwrap();
        let address_bits = tester.address_bits();
        GpuHarness::with_capacity(executor, air, hybrid_chip, cpu_chip, MAX_INS_CAPACITY)
            .with_trace_generators(
                move |chip, postflight| {
                    generate_field_expression_trace_from_postflight::<_, BLOCKS>(
                        chip,
                        postflight,
                        offset,
                        address_bits,
                    )
                },
                |chip, program, transcript, plan| {
                    chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                },
            )
    }

    fn set_and_execute_addsub<const BLOCKS: usize, const NUM_LIMBS: usize>(
        tester: &mut impl TestBuilder<F>,
        executor: &mut ModularExecutor<BLOCKS>,
        preflight: &mut TestPreflight,
        rng: &mut StdRng,
        modulus: &BigUint,
        is_setup: bool,
        offset: usize,
    ) {
        let (a, b, op) = if is_setup {
            (modulus.clone(), BigUint::zero(), ADD_LOCAL + 2)
        } else {
            let a = generate_random_biguint(modulus);
            let b = generate_random_biguint(modulus);

            let op = rng.random_range(0..2) + ADD_LOCAL; // 0 for add, 1 for sub
            (a, b, op)
        };

        let expected_answer = match op - ADD_LOCAL {
            0 => (&a + &b) % modulus,
            1 => (&a + modulus - &b) % modulus,
            2 => a.clone() % modulus,
            _ => panic!(),
        };

        // Write to memories
        // For each biguint (a, b, r), there are 2 writes:
        // 1. address_ptr which stores the actual address
        // 2. actual address which stores the biguint limbs
        // The write of result r is done in the chip.
        let ptr_as = REGISTER_AS as usize;
        let addr_ptr1 = 0;
        let addr_ptr2 = 3 * REGISTER_NUM_LIMBS;
        let addr_ptr3 = 6 * REGISTER_NUM_LIMBS;

        let data_as = MEMORY_AS as usize;
        let address1 = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u32;
        let address2 = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u32;
        let address3 = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u32;

        write_ptr_reg(tester, ptr_as, addr_ptr1, address1.into());
        write_ptr_reg(tester, ptr_as, addr_ptr2, address2.into());
        write_ptr_reg(tester, ptr_as, addr_ptr3, address3.into());

        let a_limbs: Vec<F> = biguint_to_limbs_vec(&a, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();
        let b_limbs: Vec<F> = biguint_to_limbs_vec(&b, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();

        for i in (0..NUM_LIMBS).step_by(MEMORY_BLOCK_BYTES) {
            tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
                data_as,
                address1 as usize + i,
                a_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
            );
            tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
                data_as,
                address2 as usize + i,
                b_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
            );
        }

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(offset + op),
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );
        tester.execute(executor, preflight, &instruction);

        let expected_limbs: Vec<F> = biguint_to_limbs_vec(&expected_answer, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();

        for i in (0..NUM_LIMBS).step_by(MEMORY_BLOCK_BYTES) {
            let read_vals =
                tester.read_bytes::<{ MEMORY_BLOCK_BYTES }>(data_as, address3 as usize + i);
            let expected_limbs: [F; MEMORY_BLOCK_BYTES] = expected_limbs[i..i + MEMORY_BLOCK_BYTES]
                .try_into()
                .unwrap();
            assert_eq!(read_vals, expected_limbs);
        }
    }

    fn run_addsub_test<const BLOCKS: usize, const NUM_LIMBS: usize>(
        opcode_offset: usize,
        modulus: BigUint,
        num_ops: usize,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let offset = ModularArithmeticOpcode::CLASS_OFFSET + opcode_offset;
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness = create_harness::<BLOCKS>(&tester, config, offset);

        for i in 0..num_ops {
            set_and_execute_addsub::<BLOCKS, NUM_LIMBS>(
                &mut tester,
                &mut harness.executor,
                &mut harness.preflight,
                &mut rng,
                &modulus,
                i == 0,
                offset,
            );
        }

        let tester = tester.build().load(harness).finalize();
        tester.simple_test().expect("Verification failed");
    }

    #[test]
    fn test_modular_addsub_32limb_small() {
        run_addsub_test::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(
            0,
            BigUint::from_str("357686312646216567629137").unwrap(),
            50,
        );
    }

    #[test]
    fn test_modular_addsub_32limb_secp256k1() {
        run_addsub_test::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(0, secp256k1_coord_prime(), 50);
        run_addsub_test::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(4, secp256k1_scalar_prime(), 50);
    }

    #[test]
    fn test_modular_addsub_32limb_bn254() {
        run_addsub_test::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(0, BN254_MODULUS.clone(), 50);
    }

    #[test]
    fn test_modular_addsub_48limb_bls12_381() {
        run_addsub_test::<MODULAR_BLOCKS_48, NUM_LIMBS_48>(0, BLS12_381_MODULUS.clone(), 50);
    }

    #[test]
    fn postflight_modular_addsub_generates_trace_and_rejects_bad_output() {
        const BLOCKS: usize = MODULAR_BLOCKS_32;
        let mut tester = VmChipTestBuilder::default();
        let modulus = secp256k1_coord_prime();
        let offset = ModularArithmeticOpcode::CLASS_OFFSET;
        let config = ExprBuilderConfig {
            modulus,
            num_limbs: NUM_LIMBS_32,
            limb_bits: LIMB_BITS,
        };
        let mut harness = create_harness::<BLOCKS>(&tester, config, offset);

        let rd_register = 24usize;
        let lhs_register = 8usize;
        let rhs_register = 16usize;
        let rd_pointer = 0x300u32;
        let lhs_pointer = 0x100u32;
        let rhs_pointer = 0x200u32;
        for (register, pointer) in [
            (rd_register, rd_pointer),
            (lhs_register, lhs_pointer),
            (rhs_register, rhs_pointer),
        ] {
            unsafe {
                tester.memory.memory.data.write_bytes(
                    REGISTER_AS,
                    register as u32,
                    u64::from(pointer).to_le_bytes(),
                );
            }
        }
        for (pointer, value) in [
            (lhs_pointer, BigUint::from(9u32)),
            (rhs_pointer, BigUint::from(13u32)),
        ] {
            let limbs = biguint_to_limbs_vec(&value, NUM_LIMBS_32);
            for byte_offset in (0..NUM_LIMBS_32).step_by(MEMORY_BLOCK_BYTES) {
                unsafe {
                    tester.memory.memory.data.write_bytes::<MEMORY_BLOCK_BYTES>(
                        MEMORY_AS,
                        pointer + byte_offset as u32,
                        std::array::from_fn(|index| limbs[byte_offset + index]),
                    );
                }
            }
        }
        let instruction = Instruction::from_usize(
            VmOpcode::from_usize(offset + ModularArithmeticOpcode::ADD as usize),
            [
                rd_register,
                lhs_register,
                rhs_register,
                REGISTER_AS as usize,
                MEMORY_AS as usize,
            ],
        );
        let sentinel = Instruction::from_usize(
            VmOpcode::from_usize(offset + ModularArithmeticOpcode::SUB as usize),
            [
                rd_register,
                lhs_register,
                rhs_register,
                REGISTER_AS as usize,
                MEMORY_AS as usize,
            ],
        );
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.preflight,
            &instruction,
            0,
        );
        let history = &mut harness.preflight.executions[0].history;
        let program = Program::new_without_debug_infos(&[instruction, sentinel], 0);
        let memory_config = MemoryConfig::default();
        let postflight = Postflight::new(&program, history, &memory_config, None).unwrap();
        let actual = generate_field_expression_trace_from_postflight::<_, BLOCKS>(
            &harness.chip,
            &postflight,
            offset,
            tester.address_bits(),
        )
        .unwrap();
        assert!(!actual.values.is_empty());

        history.memory.accesses.last_mut().unwrap().value[0] ^= 1;
        let corrupt = Postflight::new(&program, history, &memory_config, None).unwrap();
        assert!(
            generate_field_expression_trace_from_postflight::<_, BLOCKS>(
                &harness.chip,
                &corrupt,
                offset,
                tester.address_bits(),
            )
            .is_err()
        );
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn run_cuda_addsub_test_with_config<const BLOCKS: usize, const NUM_LIMBS: usize>(
        opcode_offset: usize,
        modulus: BigUint,
        num_ops: usize,
    ) {
        let mut rng = create_seeded_rng();

        let mut tester = GpuChipTestBuilder::default();

        let offset = ModularArithmeticOpcode::CLASS_OFFSET + opcode_offset;
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness = create_cuda_harness::<BLOCKS>(&tester, config, offset);

        for i in 0..num_ops {
            set_and_execute_addsub::<BLOCKS, NUM_LIMBS>(
                &mut tester,
                &mut harness.executor,
                &mut harness.preflight,
                &mut rng,
                &modulus,
                i == 0,
                offset,
            );
        }

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn cuda_test_modular_addsub() {
        run_cuda_addsub_test_with_config::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(
            0,
            BigUint::from_str("357686312646216567629137").unwrap(),
            50,
        );
        run_cuda_addsub_test_with_config::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(
            0,
            secp256k1_coord_prime(),
            50,
        );
        run_cuda_addsub_test_with_config::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(
            4,
            secp256k1_scalar_prime(),
            50,
        );
        run_cuda_addsub_test_with_config::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(
            0,
            BN254_MODULUS.clone(),
            50,
        );
        run_cuda_addsub_test_with_config::<MODULAR_BLOCKS_48, NUM_LIMBS_48>(
            0,
            BLS12_381_MODULUS.clone(),
            50,
        );
    }
}

#[cfg(test)]
mod muldiv_tests {
    use super::*;
    use crate::trace::generate_field_expression_trace_from_postflight;

    const MUL_LOCAL: usize = ModularArithmeticOpcode::MUL as usize;
    type Harness<const BLOCKS: usize> =
        TestChipHarness<F, ModularExecutor<BLOCKS>, ModularAir<BLOCKS>, ModularChip<F, BLOCKS>>;

    fn create_harness<const BLOCKS: usize>(
        tester: &VmChipTestBuilder<F>,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> Harness<BLOCKS> {
        let air = get_modular_muldiv_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            tester.range_checker().bus(),
            tester.address_bits(),
            offset,
        );

        let executor = get_modular_muldiv_executor(
            config.clone(),
            tester.range_checker().bus().range_max_bits,
            offset,
        );

        let chip = get_modular_muldiv_chip(
            config,
            tester.memory_helper(),
            tester.range_checker(),
            tester.address_bits(),
        );
        let address_bits = tester.address_bits();
        Harness::with_capacity(
            executor,
            air,
            chip,
            MAX_INS_CAPACITY,
            move |chip, postflight| {
                generate_field_expression_trace_from_postflight::<_, BLOCKS>(
                    chip,
                    postflight,
                    offset,
                    address_bits,
                )
            },
        )
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    type GpuHarness<const BLOCKS: usize> = GpuTestChipHarness<
        F,
        ModularExecutor<BLOCKS>,
        ModularAir<BLOCKS>,
        HybridModularChip<F, BLOCKS>,
        ModularChip<F, BLOCKS>,
    >;

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn create_cuda_harness<const BLOCKS: usize>(
        tester: &GpuChipTestBuilder,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> GpuHarness<BLOCKS> {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let air = get_modular_muldiv_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            range_bus,
            tester.address_bits(),
            offset,
        );
        let executor =
            get_modular_muldiv_executor(config.clone(), range_bus.range_max_bits, offset);

        let cpu_chip = get_modular_muldiv_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            tester.address_bits(),
        );

        // Use hybrid chip wrapping the CPU chip
        let replay_cpu_chip = get_modular_muldiv_chip(
            config,
            tester.cpu_memory_helper(),
            tester.cpu_range_checker(),
            tester.address_bits(),
        );
        let hybrid_chip = HybridModularChip::new_with_replay(
            replay_cpu_chip,
            tester.range_checker().device_ctx.clone(),
            offset,
            tester.range_checker(),
        )
        .unwrap();
        let address_bits = tester.address_bits();
        GpuHarness::with_capacity(executor, air, hybrid_chip, cpu_chip, MAX_INS_CAPACITY)
            .with_trace_generators(
                move |chip, postflight| {
                    generate_field_expression_trace_from_postflight::<_, BLOCKS>(
                        chip,
                        postflight,
                        offset,
                        address_bits,
                    )
                },
                |chip, program, transcript, plan| {
                    chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                },
            )
    }

    fn set_and_execute_muldiv<const BLOCKS: usize, const NUM_LIMBS: usize>(
        tester: &mut impl TestBuilder<F>,
        executor: &mut ModularExecutor<BLOCKS>,
        preflight: &mut TestPreflight,
        rng: &mut StdRng,
        modulus: &BigUint,
        is_setup: bool,
        offset: usize,
    ) {
        let (a, b, op) = if is_setup {
            (modulus.clone(), BigUint::zero(), MUL_LOCAL + 2)
        } else {
            let a = generate_random_biguint(modulus);
            let b = generate_random_biguint(modulus);

            let op = rng.random_range(0..2) + MUL_LOCAL; // 0 for add, 1 for sub

            (a, b, op)
        };

        let expected_answer = match op - MUL_LOCAL {
            0 => (&a * &b) % modulus,
            1 => (&a * b.modinv(modulus).unwrap()) % modulus,
            2 => a.clone() % modulus,
            _ => panic!(),
        };

        // Write to memories
        // For each biguint (a, b, r), there are 2 writes:
        // 1. address_ptr which stores the actual address
        // 2. actual address which stores the biguint limbs
        // The write of result r is done in the chip.
        let ptr_as = REGISTER_AS as usize;
        let addr_ptr1 = 0;
        let addr_ptr2 = 3 * REGISTER_NUM_LIMBS;
        let addr_ptr3 = 6 * REGISTER_NUM_LIMBS;

        let data_as = MEMORY_AS as usize;
        let address1 = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u32;
        let address2 = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u32;
        let address3 = gen_pointer(rng, MEMORY_BLOCK_BYTES) as u32;

        write_ptr_reg(tester, ptr_as, addr_ptr1, address1.into());
        write_ptr_reg(tester, ptr_as, addr_ptr2, address2.into());
        write_ptr_reg(tester, ptr_as, addr_ptr3, address3.into());

        let a_limbs: Vec<F> = biguint_to_limbs_vec(&a, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();
        let b_limbs: Vec<F> = biguint_to_limbs_vec(&b, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();

        for i in (0..NUM_LIMBS).step_by(MEMORY_BLOCK_BYTES) {
            tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
                data_as,
                address1 as usize + i,
                a_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
            );
            tester.write_bytes::<{ MEMORY_BLOCK_BYTES }>(
                data_as,
                address2 as usize + i,
                b_limbs[i..i + MEMORY_BLOCK_BYTES].try_into().unwrap(),
            );
        }

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(offset + op),
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );
        tester.execute(executor, preflight, &instruction);

        let expected_limbs: Vec<F> = biguint_to_limbs_vec(&expected_answer, NUM_LIMBS)
            .into_iter()
            .map(F::from_u8)
            .collect();

        for i in (0..NUM_LIMBS).step_by(MEMORY_BLOCK_BYTES) {
            let read_vals =
                tester.read_bytes::<{ MEMORY_BLOCK_BYTES }>(data_as, address3 as usize + i);
            let expected_limbs: [F; MEMORY_BLOCK_BYTES] = expected_limbs[i..i + MEMORY_BLOCK_BYTES]
                .try_into()
                .unwrap();
            assert_eq!(read_vals, expected_limbs);
        }
    }

    fn run_test_muldiv<const BLOCKS: usize, const NUM_LIMBS: usize>(
        opcode_offset: usize,
        modulus: BigUint,
        num_ops: usize,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let offset = ModularArithmeticOpcode::CLASS_OFFSET + opcode_offset;

        let mut harness = create_harness::<BLOCKS>(&tester, config, offset);

        for i in 0..num_ops {
            set_and_execute_muldiv::<BLOCKS, NUM_LIMBS>(
                &mut tester,
                &mut harness.executor,
                &mut harness.preflight,
                &mut rng,
                &modulus,
                i == 0,
                offset,
            );
        }
        let tester = tester.build().load(harness).finalize();

        tester.simple_test().expect("Verification failed");
    }

    #[test]
    fn test_modular_muldiv_32limb_small() {
        run_test_muldiv::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(
            0,
            BigUint::from_str("357686312646216567629137").unwrap(),
            50,
        );
    }

    #[test]
    fn test_modular_muldiv_32limb_secp256k1() {
        run_test_muldiv::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(0, secp256k1_coord_prime(), 50);
        run_test_muldiv::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(4, secp256k1_scalar_prime(), 50);
    }

    #[test]
    fn test_modular_muldiv_32limb_bn254() {
        run_test_muldiv::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(0, BN254_MODULUS.clone(), 50);
    }

    #[test]
    fn test_modular_muldiv_48limb_bls12_381() {
        run_test_muldiv::<MODULAR_BLOCKS_48, NUM_LIMBS_48>(0, BLS12_381_MODULUS.clone(), 50);
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn run_cuda_muldiv_test_with_config<const BLOCKS: usize, const NUM_LIMBS: usize>(
        opcode_offset: usize,
        modulus: BigUint,
        num_ops: usize,
    ) {
        let mut rng = create_seeded_rng();

        let mut tester = GpuChipTestBuilder::default();

        let offset = ModularArithmeticOpcode::CLASS_OFFSET + opcode_offset;
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness = create_cuda_harness::<BLOCKS>(&tester, config, offset);

        for i in 0..num_ops {
            set_and_execute_muldiv::<BLOCKS, NUM_LIMBS>(
                &mut tester,
                &mut harness.executor,
                &mut harness.preflight,
                &mut rng,
                &modulus,
                i == 0,
                offset,
            );
        }

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn cuda_test_modular_muldiv() {
        run_cuda_muldiv_test_with_config::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(
            0,
            BigUint::from_str("357686312646216567629137").unwrap(),
            50,
        );
        run_cuda_muldiv_test_with_config::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(
            0,
            secp256k1_coord_prime(),
            50,
        );
        run_cuda_muldiv_test_with_config::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(
            4,
            secp256k1_scalar_prime(),
            50,
        );
        run_cuda_muldiv_test_with_config::<MODULAR_BLOCKS_32, NUM_LIMBS_32>(
            0,
            BN254_MODULUS.clone(),
            50,
        );
        run_cuda_muldiv_test_with_config::<MODULAR_BLOCKS_48, NUM_LIMBS_48>(
            0,
            BLS12_381_MODULUS.clone(),
            50,
        );
    }
}

#[cfg(test)]
mod is_equal_tests {
    use openvm_circuit::arch::{MemoryConfig, Postflight};
    use openvm_instructions::program::Program;
    use openvm_mod_circuit_builder::test_utils::biguint_to_limbs;
    use openvm_riscv_adapters::IsEqualModU16AdapterAir;
    use openvm_riscv_circuit::adapters::U16_BITS;
    use openvm_stark_backend::{
        p3_air::BaseAir,
        p3_matrix::{
            dense::{DenseMatrix, RowMajorMatrix},
            Matrix,
        },
        utils::disable_debug_builder,
    };
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    use {
        openvm_circuit::arch::{
            cuda::postflight::GpuPostflightProgram, PreflightHistory, PreflightMemoryLog,
        },
        openvm_instructions::SystemOpcode,
        rvr_state::{
            PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent, PREFLIGHT_WRITE_BIT,
        },
    };

    use super::*;
    use crate::trace::generate_modular_is_equal_trace_from_postflight;

    type Harness<const NUM_LANES: usize, const TOTAL_LIMBS: usize> = TestChipHarness<
        F,
        VmModularIsEqualU16Executor<NUM_LANES, TOTAL_LIMBS>,
        ModularIsEqualU16Air<NUM_LANES, TOTAL_LIMBS>,
        ModularIsEqualU16Chip<F, TOTAL_LIMBS>,
    >;

    fn create_harness<const NUM_LANES: usize, const TOTAL_LIMBS: usize>(
        tester: &mut VmChipTestBuilder<F>,
        modulus: &BigUint,
        modulus_limbs: [u16; TOTAL_LIMBS],
        offset: usize,
    ) -> Harness<NUM_LANES, TOTAL_LIMBS> {
        let air = ModularIsEqualU16Air::new(
            IsEqualModU16AdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                tester.range_checker().bus(),
                tester.address_bits(),
            ),
            ModularIsEqualCoreAir::new(modulus.clone(), tester.range_checker().bus(), offset),
        );
        let executor = VmModularIsEqualU16Executor::new(offset, modulus_limbs);
        let chip = ModularIsEqualU16Chip::<F, TOTAL_LIMBS>::new(
            ModularIsEqualFiller::new(offset, modulus_limbs, tester.range_checker()),
            tester.memory_helper(),
        );
        let address_bits = tester.address_bits();
        Harness::with_capacity(
            executor,
            air,
            chip,
            MAX_INS_CAPACITY,
            move |chip, postflight| {
                generate_modular_is_equal_trace_from_postflight::<_, NUM_LANES, TOTAL_LIMBS>(
                    chip,
                    postflight,
                    offset,
                    address_bits,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn set_and_execute_is_equal<const NUM_LANES: usize, const TOTAL_LIMBS: usize>(
        tester: &mut impl TestBuilder<F>,
        executor: &mut VmModularIsEqualU16Executor<NUM_LANES, TOTAL_LIMBS>,
        preflight: &mut TestPreflight,
        rng: &mut StdRng,
        modulus: &BigUint,
        modulus_limbs: [F; TOTAL_LIMBS],
        offset: usize,
        is_setup: bool,
        b: Option<[F; TOTAL_LIMBS]>,
        c: Option<[F; TOTAL_LIMBS]>,
    ) {
        let (b, c, opcode) = if is_setup {
            (
                modulus_limbs,
                [F::ZERO; TOTAL_LIMBS],
                offset + ModularArithmeticOpcode::SETUP_ISEQ as usize,
            )
        } else {
            let b = b.unwrap_or(
                generate_field_element::<TOTAL_LIMBS, U16_BITS>(modulus, rng).map(F::from_u32),
            );
            let c = c.unwrap_or(if rng.random_bool(0.5) {
                b
            } else {
                generate_field_element::<TOTAL_LIMBS, U16_BITS>(modulus, rng).map(F::from_u32)
            });

            (b, c, offset + ModularArithmeticOpcode::IS_EQ as usize)
        };

        let instruction = write_u16_heap_default::<TOTAL_LIMBS>(tester, vec![b], vec![c], opcode);

        tester.execute(executor, preflight, &instruction);
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // POSITIVE TESTS
    //
    // Randomly generate computations and execute, ensuring that the generated trace
    // passes all constraints.
    //////////////////////////////////////////////////////////////////////////////////////

    fn test_is_equal<const NUM_LANES: usize, const TOTAL_LIMBS: usize>(
        opcode_offset: usize,
        modulus: BigUint,
        num_tests: usize,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();

        let modulus_limbs =
            biguint_to_limbs::<TOTAL_LIMBS>(modulus.clone(), U16_BITS).map(|x| x as u16);

        let mut harness = create_harness::<NUM_LANES, TOTAL_LIMBS>(
            &mut tester,
            &modulus,
            modulus_limbs,
            opcode_offset,
        );

        let modulus_limbs = modulus_limbs.map(F::from_u16);

        for i in 0..num_tests {
            set_and_execute_is_equal(
                &mut tester,
                &mut harness.executor,
                &mut harness.preflight,
                &mut rng,
                &modulus,
                modulus_limbs,
                opcode_offset,
                i == 0, // the first test is a setup test
                None,
                None,
            );
        }

        // Special case where b == c are close to the prime
        let mut b = modulus_limbs;
        b[0] -= F::ONE;
        set_and_execute_is_equal(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            &modulus,
            modulus_limbs,
            opcode_offset,
            false,
            Some(b),
            Some(b),
        );

        let tester = tester.build().load(harness).finalize();
        tester.simple_test().expect("Verification failed");
    }

    #[test]
    fn test_modular_is_equal_32limb() {
        test_is_equal::<MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>(17, secp256k1_coord_prime(), 100);
    }

    #[test]
    fn test_modular_is_equal_48limb() {
        test_is_equal::<MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>(17, BLS12_381_MODULUS.clone(), 100);
    }

    #[test]
    fn postflight_modular_is_equal_generates_trace_and_rejects_bad_output() {
        const BLOCKS: usize = MODULAR_BLOCKS_32;
        const LIMBS: usize = NUM_LIMBS_32_U16;
        let mut tester = VmChipTestBuilder::default();
        let modulus = secp256k1_coord_prime();
        let offset = ModularArithmeticOpcode::CLASS_OFFSET;
        let modulus_limbs =
            biguint_to_limbs::<LIMBS>(modulus.clone(), U16_BITS).map(|limb| limb as u16);
        let mut harness =
            create_harness::<BLOCKS, LIMBS>(&mut tester, &modulus, modulus_limbs, offset);
        let rd_register = 24u32;
        let lhs_register = 8u32;
        let rhs_register = 16u32;
        let lhs_pointer = 0x100u32;
        let rhs_pointer = 0x200u32;
        for (register, pointer) in [(lhs_register, lhs_pointer), (rhs_register, rhs_pointer)] {
            unsafe {
                tester.memory.memory.data.write_bytes(
                    REGISTER_AS,
                    register,
                    u64::from(pointer).to_le_bytes(),
                );
            }
        }
        let value: [u16; LIMBS] = std::array::from_fn(|index| (index == 0) as u16 * 5);
        for pointer in [lhs_pointer, rhs_pointer] {
            for block in 0..BLOCKS {
                unsafe {
                    tester.memory.memory.data.write::<u16, 4>(
                        MEMORY_AS,
                        pointer / 2 + (block * BLOCK_FE_WIDTH) as u32,
                        std::array::from_fn(|lane| value[block * BLOCK_FE_WIDTH + lane]),
                    );
                }
            }
        }
        let instruction = Instruction::from_usize(
            VmOpcode::from_usize(offset + ModularArithmeticOpcode::IS_EQ as usize),
            [
                rd_register as usize,
                lhs_register as usize,
                rhs_register as usize,
                REGISTER_AS as usize,
                MEMORY_AS as usize,
            ],
        );
        let sentinel = instruction.clone();
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.preflight,
            &instruction,
            0,
        );
        let history = &mut harness.preflight.executions[0].history;
        let program = Program::new_without_debug_infos(&[instruction, sentinel], 0);
        let memory_config = MemoryConfig::default();
        let postflight = Postflight::new(&program, history, &memory_config, None).unwrap();
        let actual = generate_modular_is_equal_trace_from_postflight::<_, BLOCKS, LIMBS>(
            &harness.chip,
            &postflight,
            offset,
            tester.address_bits(),
        )
        .unwrap();
        assert_eq!(actual.height(), 1);

        history.memory.accesses.last_mut().unwrap().value[0] ^= 1;
        let corrupt = Postflight::new(&program, history, &memory_config, None).unwrap();
        assert!(
            generate_modular_is_equal_trace_from_postflight::<_, BLOCKS, LIMBS>(
                &harness.chip,
                &corrupt,
                offset,
                tester.address_bits(),
            )
            .is_err()
        );
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    type GpuHarness<const NUM_LANES: usize, const TOTAL_LIMBS: usize> = GpuTestChipHarness<
        F,
        VmModularIsEqualU16Executor<NUM_LANES, TOTAL_LIMBS>,
        ModularIsEqualU16Air<NUM_LANES, TOTAL_LIMBS>,
        HybridModularIsEqualChip<F, NUM_LANES, TOTAL_LIMBS>,
        ModularIsEqualU16Chip<F, TOTAL_LIMBS>,
    >;

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn create_cuda_harness<const NUM_LANES: usize, const TOTAL_LIMBS: usize>(
        tester: &GpuChipTestBuilder,
        modulus: BigUint,
        modulus_limbs: [u16; TOTAL_LIMBS],
        offset: usize,
    ) -> GpuHarness<NUM_LANES, TOTAL_LIMBS> {
        let range_bus = default_var_range_checker_bus();
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let air = ModularIsEqualU16Air::new(
            IsEqualModU16AdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                range_bus,
                tester.address_bits(),
            ),
            ModularIsEqualCoreAir::new(modulus.clone(), range_bus, offset),
        );

        let executor = VmModularIsEqualU16Executor::new(offset, modulus_limbs);

        let cpu_chip = ModularIsEqualU16Chip::<F, TOTAL_LIMBS>::new(
            ModularIsEqualFiller::new(offset, modulus_limbs, dummy_range_checker_chip),
            tester.dummy_memory_helper(),
        );

        let gpu_cpu_chip = ModularIsEqualU16Chip::<F, TOTAL_LIMBS>::new(
            ModularIsEqualFiller::new(offset, modulus_limbs, tester.cpu_range_checker()),
            tester.cpu_memory_helper(),
        );
        let hybrid_chip = HybridModularIsEqualChip::new_with_replay(
            gpu_cpu_chip,
            tester.range_checker().device_ctx.clone(),
            modulus_limbs,
            offset,
            tester.address_bits(),
            tester.timestamp_max_bits(),
            tester.range_checker(),
        )
        .unwrap();
        let address_bits = tester.address_bits();
        GpuHarness::with_capacity(executor, air, hybrid_chip, cpu_chip, MAX_INS_CAPACITY)
            .with_trace_generators(
                move |chip, postflight| {
                    generate_modular_is_equal_trace_from_postflight::<_, NUM_LANES, TOTAL_LIMBS>(
                        chip,
                        postflight,
                        offset,
                        address_bits,
                    )
                },
                |chip, program, transcript, plan| {
                    chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                },
            )
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn run_cuda_is_equal_test_with_config<const NUM_LANES: usize, const TOTAL_LIMBS: usize>(
        opcode_offset: usize,
        modulus: BigUint,
        num_ops: usize,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default();

        let modulus_limbs =
            biguint_to_limbs::<TOTAL_LIMBS>(modulus.clone(), U16_BITS).map(|x| x as u16);

        let mut harness = create_cuda_harness::<NUM_LANES, TOTAL_LIMBS>(
            &tester,
            modulus.clone(),
            modulus_limbs,
            opcode_offset,
        );

        let modulus_limbs = modulus_limbs.map(F::from_u16);

        for i in 0..num_ops {
            set_and_execute_is_equal(
                &mut tester,
                &mut harness.executor,
                &mut harness.preflight,
                &mut rng,
                &modulus,
                modulus_limbs,
                opcode_offset,
                i == 0, // the first test is a setup test
                None,
                None,
            );
        }

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn cuda_test_modular_is_equal() {
        run_cuda_is_equal_test_with_config::<MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>(
            17,
            secp256k1_coord_prime(),
            50,
        );
        run_cuda_is_equal_test_with_config::<MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>(
            17,
            secp256k1_scalar_prime(),
            50,
        );
        run_cuda_is_equal_test_with_config::<MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>(
            17,
            BLS12_381_MODULUS.clone(),
            50,
        );
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn run_preflight_replay_is_equal_test<const BLOCKS: usize, const LIMBS: usize>(
        modulus: BigUint,
    ) {
        let opcode_base = ModularArithmeticOpcode::CLASS_OFFSET;
        let modulus_limbs =
            biguint_to_limbs::<LIMBS>(modulus.clone(), U16_BITS).map(|limb| limb as u16);
        let value_limbs: [u16; LIMBS] = std::array::from_fn(|index| if index == 0 { 5 } else { 0 });

        let setup_rd = 8usize;
        let modulus_reg = 16usize;
        let is_eq_rd = 24usize;
        let b_reg = 32usize;
        let c_reg = 40usize;
        let modulus_ptr = 0x100u32;
        let b_ptr = 0x200u32;
        let c_ptr = 0x300u32;

        let setup: Instruction = Instruction::from_usize(
            VmOpcode::from_usize(opcode_base + ModularArithmeticOpcode::SETUP_ISEQ as usize),
            [
                setup_rd,
                modulus_reg,
                0,
                REGISTER_AS as usize,
                MEMORY_AS as usize,
            ],
        );
        let is_eq: Instruction = Instruction::from_usize(
            VmOpcode::from_usize(opcode_base + ModularArithmeticOpcode::IS_EQ as usize),
            [
                is_eq_rd,
                b_reg,
                c_reg,
                REGISTER_AS as usize,
                MEMORY_AS as usize,
            ],
        );
        let program = Program::from_instructions(&[
            setup.clone(),
            is_eq.clone(),
            Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
        ]);

        let tester = GpuChipTestBuilder::default();
        let harness =
            create_cuda_harness::<BLOCKS, LIMBS>(&tester, modulus, modulus_limbs, opcode_base);

        let register_block = |pointer: u32| [pointer as u16, (pointer >> U16_BITS) as u16, 0, 0];
        let mut memory_log = Vec::with_capacity(2 * (2 + 2 * BLOCKS + 1));
        memory_log.push(PreflightMemoryEvent {
            timestamp: 1,
            address_space_and_kind: REGISTER_AS,
            pointer: modulus_reg as u32 / 2,
            value: register_block(modulus_ptr),
        });
        memory_log.push(PreflightMemoryEvent {
            timestamp: 2,
            address_space_and_kind: REGISTER_AS,
            pointer: 0,
            value: [0; 4],
        });
        for block in 0..BLOCKS {
            memory_log.push(PreflightMemoryEvent {
                timestamp: 3 + block as u32,
                address_space_and_kind: MEMORY_AS,
                pointer: modulus_ptr / 2 + (block * 4) as u32,
                value: std::array::from_fn(|limb| modulus_limbs[block * 4 + limb]),
            });
        }
        for block in 0..BLOCKS {
            memory_log.push(PreflightMemoryEvent {
                timestamp: 3 + BLOCKS as u32 + block as u32,
                address_space_and_kind: MEMORY_AS,
                pointer: (block * 4) as u32,
                value: [0; 4],
            });
        }
        let first_delta = (2 + 2 * BLOCKS + 1) as u32;
        memory_log.push(PreflightMemoryEvent {
            timestamp: first_delta,
            address_space_and_kind: REGISTER_AS | PREFLIGHT_WRITE_BIT,
            pointer: setup_rd as u32 / 2,
            value: [0; 4],
        });
        memory_log.push(PreflightMemoryEvent {
            timestamp: 1 + first_delta,
            address_space_and_kind: REGISTER_AS,
            pointer: b_reg as u32 / 2,
            value: register_block(b_ptr),
        });
        memory_log.push(PreflightMemoryEvent {
            timestamp: 2 + first_delta,
            address_space_and_kind: REGISTER_AS,
            pointer: c_reg as u32 / 2,
            value: register_block(c_ptr),
        });
        for read in 0..2 {
            let pointer = if read == 0 { b_ptr } else { c_ptr };
            for block in 0..BLOCKS {
                memory_log.push(PreflightMemoryEvent {
                    timestamp: 3 + first_delta + (read * BLOCKS + block) as u32,
                    address_space_and_kind: MEMORY_AS,
                    pointer: pointer / 2 + (block * 4) as u32,
                    value: std::array::from_fn(|limb| value_limbs[block * 4 + limb]),
                });
            }
        }
        let final_timestamp = 1 + 2 * first_delta;
        memory_log.push(PreflightMemoryEvent {
            timestamp: final_timestamp - 1,
            address_space_and_kind: REGISTER_AS | PREFLIGHT_WRITE_BIT,
            pointer: is_eq_rd as u32 / 2,
            value: [1, 0, 0, 0],
        });
        let history = PreflightHistory {
            program: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: 1 + first_delta,
                },
                PreflightProgramEvent {
                    pc: 8,
                    timestamp: final_timestamp,
                },
                PreflightProgramEvent {
                    pc: 8,
                    timestamp: final_timestamp,
                },
            ],
            memory: PreflightMemoryLog {
                accesses: memory_log,
                initial_writes: vec![
                    PreflightInitialWrite {
                        address_space: REGISTER_AS,
                        pointer: setup_rd as u32 / 2,
                        initial_value: [0; 4],
                    },
                    PreflightInitialWrite {
                        address_space: REGISTER_AS,
                        pointer: is_eq_rd as u32 / 2,
                        initial_value: [0; 4],
                    },
                ],
                ..Default::default()
            },
        };

        let memory_config = openvm_circuit::arch::MemoryConfig::default();
        let device_ctx = &tester.range_checker().device_ctx;
        let gpu_program =
            GpuPostflightProgram::upload(&program, &memory_config, device_ctx).unwrap();
        let mut corrupt = history;
        corrupt.memory.accesses[1 + first_delta as usize].value[0] |= 1;
        let counts_before = tester.range_checker().count.to_host_on(device_ctx).unwrap();
        let (gpu_corrupt, corrupt_plan) = gpu_program
            .upload_history_for_test(&program, &corrupt, Some(0))
            .unwrap();
        assert!(harness
            .gpu_chip
            .generate_proving_ctx_from_postflight(&gpu_program, &gpu_corrupt, &corrupt_plan)
            .is_err());
        assert_ne!(gpu_corrupt.error_code().unwrap(), 0);
        assert_eq!(
            tester.range_checker().count.to_host_on(device_ctx).unwrap(),
            counts_before
        );
    }

    #[cfg(all(feature = "cuda", feature = "rvr"))]
    #[test]
    fn cuda_preflight_replay_modular_is_equal_rejects_odd_pointer() {
        run_preflight_replay_is_equal_test::<MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>(
            secp256k1_coord_prime(),
        );
        run_preflight_replay_is_equal_test::<MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>(
            BLS12_381_MODULUS.clone(),
        );
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // NEGATIVE TESTS
    //
    // Given a fake trace of a single operation, setup a chip and run the test. We replace
    // part of the trace and check that the chip throws the expected error.
    //////////////////////////////////////////////////////////////////////////////////////

    /// Negative tests test for 3 "type" of errors determined by the value of b[0]:
    fn run_negative_is_equal_test<const NUM_LANES: usize, const READ_LIMBS: usize>(
        modulus: BigUint,
        opcode_offset: usize,
        test_case: usize,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();

        let modulus_limbs =
            biguint_to_limbs::<READ_LIMBS>(modulus.clone(), U16_BITS).map(|x| x as u16);

        let mut harness = create_harness::<NUM_LANES, READ_LIMBS>(
            &mut tester,
            &modulus,
            modulus_limbs,
            opcode_offset,
        );

        let modulus_limbs = modulus_limbs.map(F::from_u16);

        set_and_execute_is_equal(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            &modulus,
            modulus_limbs,
            opcode_offset,
            true,
            None,
            None,
        );

        let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
        let modify_trace = |trace: &mut DenseMatrix<F>| {
            let mut trace_row = trace
                .row_slice(0)
                .expect("trace row should be present")
                .to_vec();
            let cols: &mut ModularIsEqualCoreCols<_, READ_LIMBS> =
                trace_row.split_at_mut(adapter_width).1.borrow_mut();
            if test_case == 1 {
                // test the constraint that c_lt_mark = 2 when is_setup = 1
                cols.b[0] = F::from_u32(1);
                cols.c_lt_mark = F::ONE;
                cols.lt_marker = [F::ZERO; READ_LIMBS];
                cols.lt_marker[READ_LIMBS - 1] = F::ONE;
                cols.c_lt_diff = modulus_limbs[READ_LIMBS - 1] - cols.c[READ_LIMBS - 1];
                cols.b_lt_diff = modulus_limbs[READ_LIMBS - 1] - cols.b[READ_LIMBS - 1];
            } else if test_case == 2 {
                // test the constraint that b[i] = N[i] for all i when prefix_sum is not 1 or
                // lt_marker_sum - is_setup
                cols.b[0] = F::from_u32(2);
                cols.c_lt_mark = F::from_u8(2);
                cols.lt_marker = [F::ZERO; READ_LIMBS];
                cols.lt_marker[READ_LIMBS - 1] = F::from_u8(2);
                cols.c_lt_diff = modulus_limbs[READ_LIMBS - 1] - cols.c[READ_LIMBS - 1];
            } else if test_case == 3 {
                // test the constraint that sum_i lt_marker[i] = 2 when is_setup = 1
                cols.b[0] = F::from_u32(3);
                cols.c_lt_mark = F::from_u8(2);
                cols.lt_marker = [F::ZERO; READ_LIMBS];
                cols.lt_marker[READ_LIMBS - 1] = F::from_u8(2);
                cols.lt_marker[0] = F::ONE;
                cols.b_lt_diff = modulus_limbs[0] - cols.b[0];
                cols.c_lt_diff = modulus_limbs[READ_LIMBS - 1] - cols.c[READ_LIMBS - 1];
            }
            *trace = RowMajorMatrix::new(trace_row, trace.width());
        };

        disable_debug_builder();
        let tester = tester
            .build()
            .load_and_prank_trace(harness, modify_trace)
            .finalize();
        tester
            .simple_test()
            .expect_err("Expected verification to fail, but it passed");
    }

    #[test]
    fn negative_test_modular_is_equal_32limb() {
        run_negative_is_equal_test::<MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>(
            secp256k1_coord_prime(),
            17,
            1,
        );

        run_negative_is_equal_test::<MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>(
            secp256k1_coord_prime(),
            17,
            2,
        );

        run_negative_is_equal_test::<MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>(
            secp256k1_coord_prime(),
            17,
            3,
        );
    }

    #[test]
    fn negative_test_modular_is_equal_48limb() {
        run_negative_is_equal_test::<MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>(
            BLS12_381_MODULUS.clone(),
            17,
            1,
        );

        run_negative_is_equal_test::<MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>(
            BLS12_381_MODULUS.clone(),
            17,
            2,
        );

        run_negative_is_equal_test::<MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>(
            BLS12_381_MODULUS.clone(),
            17,
            3,
        );
    }
}
