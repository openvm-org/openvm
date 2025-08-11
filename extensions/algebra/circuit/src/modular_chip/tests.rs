use std::{borrow::BorrowMut, str::FromStr, sync::Arc};

use num_bigint::BigUint;
use num_traits::Zero;
use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
use openvm_circuit::arch::{
    instructions::LocalOpcode,
    testing::{memory::gen_pointer, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    Arena, DenseRecordArena, MatrixRecordArena, PreflightExecutor,
};
use openvm_circuit_primitives::{
    bigint::utils::{secp256k1_coord_prime, secp256k1_scalar_prime},
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS},
    VmOpcode,
};
use openvm_mod_circuit_builder::{
    test_utils::{generate_field_element, generate_random_biguint},
    utils::biguint_to_limbs_vec,
    ExprBuilderConfig, FieldExpressionCoreRecordMut,
};
use openvm_pairing_guest::{bls12_381::BLS12_381_MODULUS, bn254::BN254_MODULUS};
use openvm_rv32_adapters::{rv32_write_heap_default, write_ptr_reg, Rv32VecHeapAdapterRecord};
use openvm_rv32im_circuit::adapters::RV32_REGISTER_NUM_LIMBS;
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use crate::modular_chip::{
    get_modular_addsub_air, get_modular_addsub_chip, get_modular_addsub_step,
    get_modular_muldiv_air, get_modular_muldiv_chip, get_modular_muldiv_step, ModularAir,
    ModularChip, ModularExecutor, ModularIsEqualAir, ModularIsEqualChip, ModularIsEqualCoreAir,
    ModularIsEqualCoreCols, ModularIsEqualFiller, VmModularIsEqualExecutor,
};

const LIMB_BITS: usize = 8;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

#[cfg(test)]
mod addsub_tests {

    use test_case::test_case;

    use super::*;

    type Harness<RA, const BLOCKS: usize, const BLOCK_SIZE: usize> = TestChipHarness<
        F,
        ModularExecutor<BLOCKS, BLOCK_SIZE>,
        ModularAir<BLOCKS, BLOCK_SIZE>,
        ModularChip<F, BLOCKS, BLOCK_SIZE>,
        RA,
    >;
    const ADD_LOCAL: usize = Rv32ModularArithmeticOpcode::ADD as usize;

    fn create_test_chip<RA: Arena, const BLOCKS: usize, const BLOCK_SIZE: usize>(
        tester: &VmChipTestBuilder<F>,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> (
        Harness<RA, BLOCKS, BLOCK_SIZE>,
        (
            BitwiseOperationLookupAir<RV32_CELL_BITS>,
            SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        ),
    ) {
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));

        let air = get_modular_addsub_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            tester.range_checker().bus(),
            bitwise_bus,
            tester.address_bits(),
            offset,
        );
        let executor = get_modular_addsub_step(
            config.clone(),
            tester.range_checker().bus(),
            tester.address_bits(),
            offset,
        );
        let chip = get_modular_addsub_chip(
            config,
            tester.memory_helper(),
            tester.range_checker(),
            bitwise_chip.clone(),
            tester.address_bits(),
        );
        let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

        (harness, (bitwise_chip.air, bitwise_chip))
    }

    fn set_and_execute_addsub<
        RA: Arena,
        const BLOCKS: usize,
        const BLOCK_SIZE: usize,
        const NUM_LIMBS: usize,
    >(
        tester: &mut VmChipTestBuilder<F>,
        harness: &mut Harness<RA, BLOCKS, BLOCK_SIZE>,
        rng: &mut StdRng,
        modulus: &BigUint,
        is_setup: bool,
        offset: usize,
    ) where
        ModularExecutor<BLOCKS, BLOCK_SIZE>: PreflightExecutor<F, RA>,
    {
        let (a, b, op) = if is_setup {
            (modulus.clone(), BigUint::zero(), ADD_LOCAL + 2)
        } else {
            let a = generate_random_biguint(modulus);
            let b = generate_random_biguint(modulus);

            let op = rng.gen_range(0..2) + ADD_LOCAL; // 0 for add, 1 for sub
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
        let ptr_as = RV32_REGISTER_AS as usize;
        let addr_ptr1 = 0;
        let addr_ptr2 = 3 * RV32_REGISTER_NUM_LIMBS;
        let addr_ptr3 = 6 * RV32_REGISTER_NUM_LIMBS;

        let data_as = RV32_MEMORY_AS as usize;
        let address1 = gen_pointer(rng, BLOCK_SIZE) as u32;
        let address2 = gen_pointer(rng, BLOCK_SIZE) as u32;
        let address3 = gen_pointer(rng, BLOCK_SIZE) as u32;

        write_ptr_reg(tester, ptr_as, addr_ptr1, address1);
        write_ptr_reg(tester, ptr_as, addr_ptr2, address2);
        write_ptr_reg(tester, ptr_as, addr_ptr3, address3);

        let a_limbs: Vec<F> = biguint_to_limbs_vec(&a, NUM_LIMBS)
            .into_iter()
            .map(F::from_canonical_u8)
            .collect();
        let b_limbs: Vec<F> = biguint_to_limbs_vec(&b, NUM_LIMBS)
            .into_iter()
            .map(F::from_canonical_u8)
            .collect();

        for i in (0..NUM_LIMBS).step_by(BLOCK_SIZE) {
            tester.write::<BLOCK_SIZE>(
                data_as,
                address1 as usize + i,
                a_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
            );
            tester.write::<BLOCK_SIZE>(
                data_as,
                address2 as usize + i,
                b_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
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
        tester.execute(harness, &instruction);

        let expected_limbs: Vec<F> = biguint_to_limbs_vec(&expected_answer, NUM_LIMBS)
            .into_iter()
            .map(F::from_canonical_u8)
            .collect();

        for i in (0..NUM_LIMBS).step_by(BLOCK_SIZE) {
            let read_vals = tester.read::<BLOCK_SIZE>(data_as, address3 as usize + i);
            let expected_limbs: [F; BLOCK_SIZE] =
                expected_limbs[i..i + BLOCK_SIZE].try_into().unwrap();
            assert_eq!(read_vals, expected_limbs);
        }
    }

    fn run_addsub_test<const BLOCKS: usize, const BLOCK_SIZE: usize, const NUM_LIMBS: usize>(
        opcode_offset: usize,
        modulus: BigUint,
        num_ops: usize,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let offset = Rv32ModularArithmeticOpcode::CLASS_OFFSET + opcode_offset;
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let (mut harness, bitwise) =
            create_test_chip::<MatrixRecordArena<F>, BLOCKS, BLOCK_SIZE>(&tester, config, offset);

        for i in 0..num_ops {
            set_and_execute_addsub::<_, BLOCKS, BLOCK_SIZE, NUM_LIMBS>(
                &mut tester,
                &mut harness,
                &mut rng,
                &modulus,
                i == 0,
                offset,
            );
        }

        let tester = tester
            .build()
            .load(harness)
            .load_periphery(bitwise)
            .finalize();
        tester.simple_test().expect("Verification failed");
    }

    #[test]
    fn test_modular_addsub_1x32_small() {
        run_addsub_test::<1, 32, 32>(
            0,
            BigUint::from_str("357686312646216567629137").unwrap(),
            50,
        );
    }

    #[test]
    fn test_modular_addsub_1x32_secp256k1() {
        run_addsub_test::<1, 32, 32>(0, secp256k1_coord_prime(), 50);
        run_addsub_test::<1, 32, 32>(4, secp256k1_scalar_prime(), 50);
    }

    #[test]
    fn test_modular_addsub_1x32_bn254() {
        run_addsub_test::<1, 32, 32>(0, BN254_MODULUS.clone(), 50);
    }

    #[test]
    fn test_modular_addsub_3x16_bls12_381() {
        run_addsub_test::<3, 16, 48>(0, BLS12_381_MODULUS.clone(), 50);
    }

    #[test_case(0, secp256k1_coord_prime(), 50)]
    #[test_case(4, secp256k1_scalar_prime(), 50)]
    fn dense_record_arena_test(opcode_offset: usize, modulus: BigUint, num_ops: usize) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: 32,
            limb_bits: LIMB_BITS,
        };
        let offset = Rv32ModularArithmeticOpcode::CLASS_OFFSET + opcode_offset;

        let (mut sparse_harness, bitwise) =
            create_test_chip::<MatrixRecordArena<F>, 1, 32>(&tester, config.clone(), offset);

        {
            // doing 1xNUM_LIMBS reads and writes
            let mut dense_harness =
                create_test_chip::<DenseRecordArena, 1, 32>(&tester, config, offset).0;

            for i in 0..num_ops {
                set_and_execute_addsub::<_, 1, 32, 32>(
                    &mut tester,
                    &mut dense_harness,
                    &mut rng,
                    &modulus,
                    i == 0,
                    offset,
                );
            }

            type Record<'a> = (
                &'a mut Rv32VecHeapAdapterRecord<2, 1, 1, 32, 32>,
                FieldExpressionCoreRecordMut<'a>,
            );
            let mut record_interpreter = dense_harness.arena.get_record_seeker::<Record, _>();
            record_interpreter.transfer_to_matrix_arena(
                &mut sparse_harness.arena,
                dense_harness.executor.get_record_layout::<F>(),
            );
        }

        let tester = tester
            .build()
            .load(sparse_harness)
            .load_periphery(bitwise)
            .finalize();
        tester.simple_test().expect("Verification failed");
    }
}

#[cfg(test)]
mod muldiv_tests {
    use super::*;

    const MUL_LOCAL: usize = Rv32ModularArithmeticOpcode::MUL as usize;
    type Harness<const BLOCKS: usize, const BLOCK_SIZE: usize> = TestChipHarness<
        F,
        ModularExecutor<BLOCKS, BLOCK_SIZE>,
        ModularAir<BLOCKS, BLOCK_SIZE>,
        ModularChip<F, BLOCKS, BLOCK_SIZE>,
    >;

    fn create_test_chip<const BLOCKS: usize, const BLOCK_SIZE: usize>(
        tester: &VmChipTestBuilder<F>,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> (
        Harness<BLOCKS, BLOCK_SIZE>,
        (
            BitwiseOperationLookupAir<RV32_CELL_BITS>,
            SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        ),
    ) {
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));

        let air = get_modular_muldiv_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            tester.range_checker().bus(),
            bitwise_bus,
            tester.address_bits(),
            offset,
        );

        let executor = get_modular_muldiv_step(
            config.clone(),
            tester.range_checker().bus(),
            tester.address_bits(),
            offset,
        );

        let chip = get_modular_muldiv_chip(
            config,
            tester.memory_helper(),
            tester.range_checker(),
            bitwise_chip.clone(),
            tester.address_bits(),
        );
        let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

        (harness, (bitwise_chip.air, bitwise_chip))
    }

    fn set_and_execute_muldiv<
        const BLOCKS: usize,
        const BLOCK_SIZE: usize,
        const NUM_LIMBS: usize,
    >(
        tester: &mut VmChipTestBuilder<F>,
        harness: &mut Harness<BLOCKS, BLOCK_SIZE>,
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

            let op = rng.gen_range(0..2) + MUL_LOCAL; // 0 for add, 1 for sub

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
        let ptr_as = RV32_REGISTER_AS as usize;
        let addr_ptr1 = 0;
        let addr_ptr2 = 12;
        let addr_ptr3 = 24;

        let data_as = RV32_MEMORY_AS as usize;
        let address1 = gen_pointer(rng, BLOCK_SIZE) as u32;
        let address2 = gen_pointer(rng, BLOCK_SIZE) as u32;
        let address3 = gen_pointer(rng, BLOCK_SIZE) as u32;

        write_ptr_reg(tester, ptr_as, addr_ptr1, address1);
        write_ptr_reg(tester, ptr_as, addr_ptr2, address2);
        write_ptr_reg(tester, ptr_as, addr_ptr3, address3);

        let a_limbs: Vec<F> = biguint_to_limbs_vec(&a, NUM_LIMBS)
            .into_iter()
            .map(F::from_canonical_u8)
            .collect();
        let b_limbs: Vec<F> = biguint_to_limbs_vec(&b, NUM_LIMBS)
            .into_iter()
            .map(F::from_canonical_u8)
            .collect();

        for i in (0..NUM_LIMBS).step_by(BLOCK_SIZE) {
            tester.write::<BLOCK_SIZE>(
                data_as,
                address1 as usize + i,
                a_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
            );
            tester.write::<BLOCK_SIZE>(
                data_as,
                address2 as usize + i,
                b_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
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
        tester.execute(harness, &instruction);

        let expected_limbs: Vec<F> = biguint_to_limbs_vec(&expected_answer, NUM_LIMBS)
            .into_iter()
            .map(F::from_canonical_u8)
            .collect();

        for i in (0..NUM_LIMBS).step_by(BLOCK_SIZE) {
            let read_vals = tester.read::<BLOCK_SIZE>(data_as, address3 as usize + i);
            let expected_limbs: [F; BLOCK_SIZE] =
                expected_limbs[i..i + BLOCK_SIZE].try_into().unwrap();
            assert_eq!(read_vals, expected_limbs);
        }
    }

    fn run_test_muldiv<const BLOCKS: usize, const BLOCK_SIZE: usize, const NUM_LIMBS: usize>(
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
        let offset = Rv32ModularArithmeticOpcode::CLASS_OFFSET + opcode_offset;

        let (mut harness, bitwise) =
            create_test_chip::<BLOCKS, BLOCK_SIZE>(&tester, config, offset);

        for i in 0..num_ops {
            set_and_execute_muldiv::<BLOCKS, BLOCK_SIZE, NUM_LIMBS>(
                &mut tester,
                &mut harness,
                &mut rng,
                &modulus,
                i == 0,
                offset,
            );
        }
        let tester = tester
            .build()
            .load(harness)
            .load_periphery(bitwise)
            .finalize();

        tester.simple_test().expect("Verification failed");
    }

    #[test]
    fn test_modular_muldiv_1x32_small() {
        run_test_muldiv::<1, 32, 32>(
            0,
            BigUint::from_str("357686312646216567629137").unwrap(),
            50,
        );
    }

    #[test]
    fn test_modular_muldiv_1x32_secp256k1() {
        run_test_muldiv::<1, 32, 32>(0, secp256k1_coord_prime(), 50);
        run_test_muldiv::<1, 32, 32>(4, secp256k1_scalar_prime(), 50);
    }

    #[test]
    fn test_modular_muldiv_1x32_bn254() {
        run_test_muldiv::<1, 32, 32>(0, BN254_MODULUS.clone(), 50);
    }

    #[test]
    fn test_modular_muldiv_3x16_bls12_381() {
        run_test_muldiv::<3, 16, 48>(0, BLS12_381_MODULUS.clone(), 50);
    }
}

#[cfg(test)]
mod is_equal_tests {
    use openvm_rv32_adapters::{
        Rv32IsEqualModAdapterAir, Rv32IsEqualModAdapterExecutor, Rv32IsEqualModAdapterFiller,
    };
    use openvm_stark_backend::{
        p3_air::BaseAir,
        p3_matrix::{
            dense::{DenseMatrix, RowMajorMatrix},
            Matrix,
        },
        utils::disable_debug_builder,
        verifier::VerificationError,
    };

    use super::*;

    type Harness<const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize> =
        TestChipHarness<
            F,
            VmModularIsEqualExecutor<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
            ModularIsEqualAir<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
            ModularIsEqualChip<F, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        >;

    fn create_test_chips<
        const NUM_LANES: usize,
        const LANE_SIZE: usize,
        const TOTAL_LIMBS: usize,
    >(
        tester: &mut VmChipTestBuilder<F>,
        modulus: &BigUint,
        modulus_limbs: [u8; TOTAL_LIMBS],
        offset: usize,
    ) -> (
        Harness<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        (
            BitwiseOperationLookupAir<LIMB_BITS>,
            SharedBitwiseOperationLookupChip<LIMB_BITS>,
        ),
    ) {
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<LIMB_BITS>::new(bitwise_bus));

        let air = ModularIsEqualAir::new(
            Rv32IsEqualModAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise_bus,
                tester.address_bits(),
            ),
            ModularIsEqualCoreAir::new(modulus.clone(), bitwise_bus, offset),
        );
        let executor = VmModularIsEqualExecutor::new(
            Rv32IsEqualModAdapterExecutor::new(tester.address_bits()),
            offset,
            modulus_limbs,
        );
        let chip = ModularIsEqualChip::<F, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>::new(
            ModularIsEqualFiller::new(
                Rv32IsEqualModAdapterFiller::new(tester.address_bits(), bitwise_chip.clone()),
                offset,
                modulus_limbs,
                bitwise_chip.clone(),
            ),
            tester.memory_helper(),
        );
        let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

        (harness, (bitwise_chip.air, bitwise_chip))
    }

    #[allow(clippy::too_many_arguments)]
    fn set_and_execute_is_equal<
        const NUM_LANES: usize,
        const LANE_SIZE: usize,
        const TOTAL_LIMBS: usize,
    >(
        tester: &mut VmChipTestBuilder<F>,
        harness: &mut Harness<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        rng: &mut StdRng,
        modulus: &BigUint,
        offset: usize,
        modulus_limbs: [F; TOTAL_LIMBS],
        is_setup: bool,
        b: Option<[F; TOTAL_LIMBS]>,
        c: Option<[F; TOTAL_LIMBS]>,
    ) {
        let instruction = if is_setup {
            rv32_write_heap_default::<TOTAL_LIMBS>(
                tester,
                vec![modulus_limbs],
                vec![[F::ZERO; TOTAL_LIMBS]],
                offset + Rv32ModularArithmeticOpcode::SETUP_ISEQ as usize,
            )
        } else {
            let b = b.unwrap_or(
                generate_field_element::<TOTAL_LIMBS, LIMB_BITS>(modulus, rng)
                    .map(F::from_canonical_u32),
            );
            let c = c.unwrap_or(if rng.gen_bool(0.5) {
                b
            } else {
                generate_field_element::<TOTAL_LIMBS, LIMB_BITS>(modulus, rng)
                    .map(F::from_canonical_u32)
            });

            rv32_write_heap_default::<TOTAL_LIMBS>(
                tester,
                vec![b],
                vec![c],
                offset + Rv32ModularArithmeticOpcode::IS_EQ as usize,
            )
        };
        tester.execute(harness, &instruction);
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // POSITIVE TESTS
    //
    // Randomly generate computations and execute, ensuring that the generated trace
    // passes all constraints.
    //////////////////////////////////////////////////////////////////////////////////////

    #[test]
    fn test_modular_is_equal_1x32() {
        test_is_equal::<1, 32, 32>(17, secp256k1_coord_prime(), 100);
    }

    #[test]
    fn test_modular_is_equal_3x16() {
        test_is_equal::<3, 16, 48>(17, BLS12_381_MODULUS.clone(), 100);
    }

    fn test_is_equal<const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize>(
        opcode_offset: usize,
        modulus: BigUint,
        num_tests: usize,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();

        let modulus_limbs: [u8; TOTAL_LIMBS] = biguint_to_limbs_vec(&modulus, TOTAL_LIMBS)
            .try_into()
            .unwrap();

        let (mut harness, bitwise) = create_test_chips::<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>(
            &mut tester,
            &modulus,
            modulus_limbs,
            opcode_offset,
        );

        let modulus_limbs = modulus_limbs.map(F::from_canonical_u8);

        for i in 0..num_tests {
            set_and_execute_is_equal(
                &mut tester,
                &mut harness,
                &mut rng,
                &modulus,
                opcode_offset,
                modulus_limbs,
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
            &mut harness,
            &mut rng,
            &modulus,
            opcode_offset,
            modulus_limbs,
            false,
            Some(b),
            Some(b),
        );

        let tester = tester
            .build()
            .load(harness)
            .load_periphery(bitwise)
            .finalize();
        tester.simple_test().expect("Verification failed");
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // NEGATIVE TESTS
    //
    // Given a fake trace of a single operation, setup a chip and run the test. We replace
    // part of the trace and check that the chip throws the expected error.
    //////////////////////////////////////////////////////////////////////////////////////

    /// Negative tests test for 3 "type" of errors determined by the value of b[0]:
    fn run_negative_is_equal_test<
        const NUM_LANES: usize,
        const LANE_SIZE: usize,
        const READ_LIMBS: usize,
    >(
        modulus: BigUint,
        opcode_offset: usize,
        test_case: usize,
        expected_error: VerificationError,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();

        let modulus_limbs: [u8; READ_LIMBS] = biguint_to_limbs_vec(&modulus, READ_LIMBS)
            .try_into()
            .unwrap();

        let (mut harness, bitwise) = create_test_chips::<NUM_LANES, LANE_SIZE, READ_LIMBS>(
            &mut tester,
            &modulus,
            modulus_limbs,
            opcode_offset,
        );

        let modulus_limbs = modulus_limbs.map(F::from_canonical_u8);

        set_and_execute_is_equal(
            &mut tester,
            &mut harness,
            &mut rng,
            &modulus,
            opcode_offset,
            modulus_limbs,
            true,
            None,
            None,
        );

        let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
        let modify_trace = |trace: &mut DenseMatrix<F>| {
            let mut trace_row = trace.row_slice(0).to_vec();
            let cols: &mut ModularIsEqualCoreCols<_, READ_LIMBS> =
                trace_row.split_at_mut(adapter_width).1.borrow_mut();
            if test_case == 1 {
                // test the constraint that c_lt_mark = 2 when is_setup = 1
                cols.b[0] = F::from_canonical_u32(1);
                cols.c_lt_mark = F::ONE;
                cols.lt_marker = [F::ZERO; READ_LIMBS];
                cols.lt_marker[READ_LIMBS - 1] = F::ONE;
                cols.c_lt_diff = modulus_limbs[READ_LIMBS - 1] - cols.c[READ_LIMBS - 1];
                cols.b_lt_diff = modulus_limbs[READ_LIMBS - 1] - cols.b[READ_LIMBS - 1];
            } else if test_case == 2 {
                // test the constraint that b[i] = N[i] for all i when prefix_sum is not 1 or
                // lt_marker_sum - is_setup
                cols.b[0] = F::from_canonical_u32(2);
                cols.c_lt_mark = F::from_canonical_u8(2);
                cols.lt_marker = [F::ZERO; READ_LIMBS];
                cols.lt_marker[READ_LIMBS - 1] = F::from_canonical_u8(2);
                cols.c_lt_diff = modulus_limbs[READ_LIMBS - 1] - cols.c[READ_LIMBS - 1];
            } else if test_case == 3 {
                // test the constraint that sum_i lt_marker[i] = 2 when is_setup = 1
                cols.b[0] = F::from_canonical_u32(3);
                cols.c_lt_mark = F::from_canonical_u8(2);
                cols.lt_marker = [F::ZERO; READ_LIMBS];
                cols.lt_marker[READ_LIMBS - 1] = F::from_canonical_u8(2);
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
            .load_periphery(bitwise)
            .finalize();
        tester.simple_test_with_expected_error(expected_error);
    }

    #[test]
    fn negative_test_modular_is_equal_1x32() {
        run_negative_is_equal_test::<1, 32, 32>(
            secp256k1_coord_prime(),
            17,
            1,
            VerificationError::OodEvaluationMismatch,
        );

        run_negative_is_equal_test::<1, 32, 32>(
            secp256k1_coord_prime(),
            17,
            2,
            VerificationError::OodEvaluationMismatch,
        );

        run_negative_is_equal_test::<1, 32, 32>(
            secp256k1_coord_prime(),
            17,
            3,
            VerificationError::OodEvaluationMismatch,
        );
    }

    #[test]
    fn negative_test_modular_is_equal_3x16() {
        run_negative_is_equal_test::<3, 16, 48>(
            BLS12_381_MODULUS.clone(),
            17,
            1,
            VerificationError::OodEvaluationMismatch,
        );

        run_negative_is_equal_test::<3, 16, 48>(
            BLS12_381_MODULUS.clone(),
            17,
            2,
            VerificationError::OodEvaluationMismatch,
        );

        run_negative_is_equal_test::<3, 16, 48>(
            BLS12_381_MODULUS.clone(),
            17,
            3,
            VerificationError::OodEvaluationMismatch,
        );
    }
}
