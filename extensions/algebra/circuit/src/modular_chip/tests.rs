use std::array::from_fn;

use num_bigint_dig::BigUint;
use num_traits::Zero;
use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
use openvm_circuit::arch::{
    instructions::UsizeOpcode, testing::VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
};
use openvm_circuit_primitives::{
    bigint::utils::{
        big_uint_mod_inverse, big_uint_to_limbs, secp256k1_coord_prime, secp256k1_scalar_prime,
    },
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
};
use openvm_instructions::{instruction::Instruction, riscv::RV32_CELL_BITS, VmOpcode};
use openvm_mod_circuit_builder::{
    test_utils::{biguint_to_limbs, generate_field_element},
    ExprBuilderConfig,
};
use openvm_pairing_guest::bls12_381::BLS12_381_MODULUS;
use openvm_rv32_adapters::{
    rv32_write_heap_default, write_ptr_reg, Rv32IsEqualModAdapterChip, Rv32VecHeapAdapterChip,
};
use openvm_rv32im_circuit::adapters::RV32_REGISTER_NUM_LIMBS;
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::Rng;

use super::{ModularAddSubChip, ModularIsEqualChip, ModularIsEqualCoreChip, ModularMulDivChip};

const NUM_LIMBS: usize = 32;
const LIMB_BITS: usize = 8;
const BLOCK_SIZE: usize = 32;
type F = BabyBear;

const ADD_LOCAL: usize = Rv32ModularArithmeticOpcode::ADD as usize;
const MUL_LOCAL: usize = Rv32ModularArithmeticOpcode::MUL as usize;

#[test]
fn test_coord_addsub() {
    let opcode_offset = 0;
    let modulus = secp256k1_coord_prime();
    test_addsub(opcode_offset, modulus);
}

#[test]
fn test_scalar_addsub() {
    let opcode_offset = 4;
    let modulus = secp256k1_scalar_prime();
    test_addsub(opcode_offset, modulus);
}

fn test_addsub(opcode_offset: usize, modulus: BigUint) {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: modulus.clone(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    // doing 1xNUM_LIMBS reads and writes
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 1, 1, BLOCK_SIZE, BLOCK_SIZE>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_bridge(),
        tester.address_bits(),
        bitwise_chip.clone(),
    );
    let mut chip = ModularAddSubChip::new(
        adapter,
        config,
        Rv32ModularArithmeticOpcode::default_offset() + opcode_offset,
        tester.range_checker(),
        tester.offline_memory_mutex_arc(),
    );
    let mut rng = create_seeded_rng();
    let num_tests = 50;
    let mut all_ops = vec![ADD_LOCAL + 2]; // setup
    let mut all_a = vec![modulus.clone()];
    let mut all_b = vec![BigUint::zero()];

    // First loop: generate all random test data.
    for _ in 0..num_tests {
        let a_digits: Vec<_> = (0..NUM_LIMBS)
            .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
            .collect();
        let mut a = BigUint::new(a_digits.clone());
        let b_digits: Vec<_> = (0..NUM_LIMBS)
            .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
            .collect();
        let mut b = BigUint::new(b_digits.clone());

        let op = rng.gen_range(0..2) + ADD_LOCAL; // 0 for add, 1 for sub
        a %= &modulus;
        b %= &modulus;

        all_ops.push(op);
        all_a.push(a);
        all_b.push(b);
    }
    // Second loop: actually run the tests.
    for i in 0..=num_tests {
        let op = all_ops[i];
        let a = all_a[i].clone();
        let b = all_b[i].clone();
        if i > 0 {
            // if not setup
            assert!(a < modulus);
            assert!(b < modulus);
        }
        let expected_answer = match op - ADD_LOCAL {
            0 => (&a + &b) % &modulus,
            1 => (&a + &modulus - &b) % &modulus,
            2 => a.clone() % &modulus,
            _ => panic!(),
        };

        // Write to memories
        // For each bigunint (a, b, r), there are 2 writes:
        // 1. address_ptr which stores the actual address
        // 2. actual address which stores the biguint limbs
        // The write of result r is done in the chip.
        let ptr_as = 1;
        let addr_ptr1 = 0;
        let addr_ptr2 = 3 * RV32_REGISTER_NUM_LIMBS;
        let addr_ptr3 = 6 * RV32_REGISTER_NUM_LIMBS;

        let data_as = 2;
        let address1 = 0u32;
        let address2 = 128u32;
        let address3 = (1 << 28) + 1234; // a large memory address to test heap adapter

        write_ptr_reg(&mut tester, ptr_as, addr_ptr1, address1);
        write_ptr_reg(&mut tester, ptr_as, addr_ptr2, address2);
        write_ptr_reg(&mut tester, ptr_as, addr_ptr3, address3);

        let a_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(a.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address1 as usize, a_limbs);
        let b_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(b.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address2 as usize, b_limbs);

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(chip.0.core.air.offset + op),
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );
        tester.execute(&mut chip, &instruction);

        let expected_limbs = biguint_to_limbs::<NUM_LIMBS>(expected_answer, LIMB_BITS);
        for (i, expected) in expected_limbs.into_iter().enumerate() {
            let address = address3 as usize + i;
            let read_val = tester.read_cell(data_as, address);
            assert_eq!(BabyBear::from_canonical_u32(expected), read_val);
        }
    }
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();

    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_coord_muldiv() {
    let opcode_offset = 0;
    let modulus = secp256k1_coord_prime();
    test_muldiv(opcode_offset, modulus);
}

#[test]
fn test_scalar_muldiv() {
    let opcode_offset = 4;
    let modulus = secp256k1_scalar_prime();
    test_muldiv(opcode_offset, modulus);
}

fn test_muldiv(opcode_offset: usize, modulus: BigUint) {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: modulus.clone(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    // doing 1xNUM_LIMBS reads and writes
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 1, 1, BLOCK_SIZE, BLOCK_SIZE>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_bridge(),
        tester.address_bits(),
        bitwise_chip.clone(),
    );
    let mut chip = ModularMulDivChip::new(
        adapter,
        config,
        Rv32ModularArithmeticOpcode::default_offset() + opcode_offset,
        tester.range_checker(),
        tester.offline_memory_mutex_arc(),
    );
    let mut rng = create_seeded_rng();
    let num_tests = 50;
    let mut all_ops = vec![MUL_LOCAL + 2];
    let mut all_a = vec![modulus.clone()];
    let mut all_b = vec![BigUint::zero()];

    // First loop: generate all random test data.
    for _ in 0..num_tests {
        let a_digits: Vec<_> = (0..NUM_LIMBS)
            .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
            .collect();
        let mut a = BigUint::new(a_digits.clone());
        let b_digits: Vec<_> = (0..NUM_LIMBS)
            .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
            .collect();
        let mut b = BigUint::new(b_digits.clone());

        // let op = rng.gen_range(2..4); // 2 for mul, 3 for div
        let op = MUL_LOCAL;
        a %= &modulus;
        b %= &modulus;

        all_ops.push(op);
        all_a.push(a);
        all_b.push(b);
    }
    // Second loop: actually run the tests.
    for i in 0..=num_tests {
        let op = all_ops[i];
        let a = all_a[i].clone();
        let b = all_b[i].clone();
        if i > 0 {
            // if not setup
            assert!(a < modulus);
            assert!(b < modulus);
        }
        let expected_answer = match op - MUL_LOCAL {
            0 => (&a * &b) % &modulus,
            1 => (&a * big_uint_mod_inverse(&b, &modulus)) % &modulus,
            2 => a.clone() % &modulus,
            _ => panic!(),
        };

        // Write to memories
        // For each bigunint (a, b, r), there are 2 writes:
        // 1. address_ptr which stores the actual address
        // 2. actual address which stores the biguint limbs
        // The write of result r is done in the chip.
        let ptr_as = 1;
        let addr_ptr1 = 0;
        let addr_ptr2 = 12;
        let addr_ptr3 = 24;

        let data_as = 2;
        let address1 = 0;
        let address2 = 128;
        let address3 = 256;

        write_ptr_reg(&mut tester, ptr_as, addr_ptr1, address1);
        write_ptr_reg(&mut tester, ptr_as, addr_ptr2, address2);
        write_ptr_reg(&mut tester, ptr_as, addr_ptr3, address3);

        let a_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(a.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address1 as usize, a_limbs);
        let b_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(b.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address2 as usize, b_limbs);

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(chip.0.core.air.offset + op),
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );
        tester.execute(&mut chip, &instruction);

        let expected_limbs = biguint_to_limbs::<NUM_LIMBS>(expected_answer, LIMB_BITS);
        for (i, expected) in expected_limbs.into_iter().enumerate() {
            let address = address3 as usize + i;
            let read_val = tester.read_cell(data_as, address);
            assert_eq!(BabyBear::from_canonical_u32(expected), read_val);
        }
    }
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();

    tester.simple_test().expect("Verification failed");
}

fn test_is_equal<const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize>(
    opcode_offset: usize,
    modulus: BigUint,
    num_tests: usize,
) {
    let mut rng = create_seeded_rng();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<LIMB_BITS>::new(bitwise_bus);

    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let mut chip = ModularIsEqualChip::<F, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>::new(
        Rv32IsEqualModAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_bridge(),
            tester.address_bits(),
            bitwise_chip.clone(),
        ),
        ModularIsEqualCoreChip::new(modulus.clone(), bitwise_chip.clone(), opcode_offset),
        tester.offline_memory_mutex_arc(),
    );

    {
        let vec = big_uint_to_limbs(&modulus, LIMB_BITS);
        let modulus_limbs: [F; TOTAL_LIMBS] = std::array::from_fn(|i| {
            if i < vec.len() {
                F::from_canonical_usize(vec[i])
            } else {
                F::ZERO
            }
        });

        let setup_instruction = rv32_write_heap_default::<TOTAL_LIMBS>(
            &mut tester,
            vec![modulus_limbs],
            vec![[F::ZERO; TOTAL_LIMBS]],
            opcode_offset + Rv32ModularArithmeticOpcode::SETUP_ISEQ as usize,
        );
        tester.execute(&mut chip, &setup_instruction);
    }
    for _ in 0..num_tests {
        let b = generate_field_element::<TOTAL_LIMBS, LIMB_BITS>(&modulus, &mut rng);
        let c = if rng.gen_bool(0.5) {
            b
        } else {
            generate_field_element::<TOTAL_LIMBS, LIMB_BITS>(&modulus, &mut rng)
        };

        let instruction = rv32_write_heap_default::<TOTAL_LIMBS>(
            &mut tester,
            vec![b.map(F::from_canonical_u32)],
            vec![c.map(F::from_canonical_u32)],
            opcode_offset + Rv32ModularArithmeticOpcode::IS_EQ as usize,
        );
        tester.execute(&mut chip, &instruction);
    }

    // Special case where b == c are close to the prime
    let b_vec = big_uint_to_limbs(&modulus, LIMB_BITS);
    let mut b = from_fn(|i| if i < b_vec.len() { b_vec[i] as u32 } else { 0 });
    b[0] -= 1;
    let instruction = rv32_write_heap_default::<TOTAL_LIMBS>(
        &mut tester,
        vec![b.map(F::from_canonical_u32)],
        vec![b.map(F::from_canonical_u32)],
        opcode_offset + Rv32ModularArithmeticOpcode::IS_EQ as usize,
    );
    tester.execute(&mut chip, &instruction);

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_modular_is_equal_1x32() {
    test_is_equal::<1, 32, 32>(17, secp256k1_coord_prime(), 100);
}

#[test]
fn test_modular_is_equal_3x16() {
    test_is_equal::<3, 16, 48>(17, BLS12_381_MODULUS.clone(), 100);
}
