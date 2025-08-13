use std::{str::FromStr, sync::Arc};

use num_bigint::BigUint;
use num_traits::Zero;
use openvm_algebra_transpiler::Fp2Opcode;
use openvm_circuit::arch::testing::{
    memory::gen_pointer, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
};
use openvm_circuit_primitives::{
    bigint::utils::secp256k1_coord_prime,
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode, VmOpcode,
};
use openvm_mod_circuit_builder::{
    test_utils::generate_random_biguint, utils::biguint_to_limbs_vec, ExprBuilderConfig,
};
use openvm_pairing_guest::{bls12_381::BLS12_381_MODULUS, bn254::BN254_MODULUS};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use crate::fp2_chip::{
    get_fp2_addsub_air, get_fp2_addsub_chip, get_fp2_addsub_step, get_fp2_muldiv_air,
    get_fp2_muldiv_chip, get_fp2_muldiv_step, Fp2Air, Fp2Chip, Fp2Executor,
};

const LIMB_BITS: usize = 8;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness<const BLOCKS: usize, const BLOCK_SIZE: usize> = TestChipHarness<
    F,
    Fp2Executor<BLOCKS, BLOCK_SIZE>,
    Fp2Air<BLOCKS, BLOCK_SIZE>,
    Fp2Chip<F, BLOCKS, BLOCK_SIZE>,
>;

fn create_addsub_test_chips<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    tester: &mut VmChipTestBuilder<F>,
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

    let air = get_fp2_addsub_air(
        tester.execution_bridge(),
        tester.memory_bridge(),
        config.clone(),
        tester.range_checker().bus(),
        bitwise_bus,
        tester.address_bits(),
        offset,
    );
    let executor = get_fp2_addsub_step(
        config.clone(),
        tester.range_checker().bus(),
        tester.address_bits(),
        offset,
    );
    let chip = get_fp2_addsub_chip(
        config,
        tester.memory_helper(),
        tester.range_checker(),
        bitwise_chip.clone(),
        tester.address_bits(),
    );
    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (bitwise_chip.air, bitwise_chip))
}

fn create_muldiv_test_chips<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    tester: &mut VmChipTestBuilder<F>,
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

    let air = get_fp2_muldiv_air(
        tester.execution_bridge(),
        tester.memory_bridge(),
        config.clone(),
        tester.range_checker().bus(),
        bitwise_bus,
        tester.address_bits(),
        offset,
    );
    let executor = get_fp2_muldiv_step(
        config.clone(),
        tester.range_checker().bus(),
        tester.address_bits(),
        offset,
    );
    let chip = get_fp2_muldiv_chip(
        config,
        tester.memory_helper(),
        tester.range_checker(),
        bitwise_chip.clone(),
        tester.address_bits(),
    );
    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (bitwise_chip.air, bitwise_chip))
}

fn set_and_execute_fp2<const BLOCKS: usize, const BLOCK_SIZE: usize, const NUM_LIMBS: usize>(
    tester: &mut VmChipTestBuilder<F>,
    harness: &mut Harness<BLOCKS, BLOCK_SIZE>,
    rng: &mut StdRng,
    modulus: &BigUint,
    is_setup: bool,
    is_addsub: bool,
    offset: usize,
) {
    let (a_c0, a_c1, b_c0, b_c1, op_local) = if is_setup {
        (
            modulus.clone(),
            BigUint::zero(),
            BigUint::zero(),
            BigUint::zero(),
            if is_addsub {
                Fp2Opcode::SETUP_ADDSUB as usize
            } else {
                Fp2Opcode::SETUP_MULDIV as usize
            },
        )
    } else {
        let a_c0 = generate_random_biguint(modulus);
        let a_c1 = generate_random_biguint(modulus);

        let b_c0 = generate_random_biguint(modulus);
        let b_c1 = generate_random_biguint(modulus);

        let op = rng.gen_range(0..2);
        let op = if is_addsub {
            match op {
                0 => Fp2Opcode::ADD as usize,
                1 => Fp2Opcode::SUB as usize,
                _ => panic!(),
            }
        } else {
            match op {
                0 => Fp2Opcode::MUL as usize,
                1 => Fp2Opcode::DIV as usize,
                _ => panic!(),
            }
        };
        (a_c0, a_c1, b_c0, b_c1, op)
    };

    let ptr_as = RV32_REGISTER_AS as usize;
    let data_as = RV32_MEMORY_AS as usize;

    let rs1_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
    let rs2_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
    let rd_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

    let a_base_addr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS) as u32;
    let b_base_addr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS) as u32;
    let result_base_addr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS) as u32;

    tester.write::<RV32_REGISTER_NUM_LIMBS>(
        ptr_as,
        rs1_ptr,
        a_base_addr.to_le_bytes().map(F::from_canonical_u8),
    );
    tester.write::<RV32_REGISTER_NUM_LIMBS>(
        ptr_as,
        rs2_ptr,
        b_base_addr.to_le_bytes().map(F::from_canonical_u8),
    );
    tester.write::<RV32_REGISTER_NUM_LIMBS>(
        ptr_as,
        rd_ptr,
        result_base_addr.to_le_bytes().map(F::from_canonical_u8),
    );

    let a_c0_limbs: Vec<F> = biguint_to_limbs_vec(&a_c0, NUM_LIMBS)
        .into_iter()
        .map(F::from_canonical_u8)
        .collect();
    let a_c1_limbs: Vec<F> = biguint_to_limbs_vec(&a_c1, NUM_LIMBS)
        .into_iter()
        .map(F::from_canonical_u8)
        .collect();
    let b_c0_limbs: Vec<F> = biguint_to_limbs_vec(&b_c0, NUM_LIMBS)
        .into_iter()
        .map(F::from_canonical_u8)
        .collect();
    let b_c1_limbs: Vec<F> = biguint_to_limbs_vec(&b_c1, NUM_LIMBS)
        .into_iter()
        .map(F::from_canonical_u8)
        .collect();

    for i in (0..NUM_LIMBS).step_by(BLOCK_SIZE) {
        tester.write::<BLOCK_SIZE>(
            data_as,
            a_base_addr as usize + i,
            a_c0_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
        );

        tester.write::<BLOCK_SIZE>(
            data_as,
            (a_base_addr + NUM_LIMBS as u32) as usize + i,
            a_c1_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
        );

        tester.write::<BLOCK_SIZE>(
            data_as,
            b_base_addr as usize + i,
            b_c0_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
        );

        tester.write::<BLOCK_SIZE>(
            data_as,
            (b_base_addr + NUM_LIMBS as u32) as usize + i,
            b_c1_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
        );
    }

    let instruction = Instruction::from_isize(
        VmOpcode::from_usize(offset + op_local),
        rd_ptr as isize,
        rs1_ptr as isize,
        rs2_ptr as isize,
        ptr_as as isize,
        data_as as isize,
    );
    tester.execute(harness, &instruction);
}

fn run_test_with_config<const BLOCKS: usize, const BLOCK_SIZE: usize, const NUM_LIMBS: usize>(
    modulus: BigUint,
    num_ops: usize,
    is_addsub: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: modulus.clone(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };

    let offset = Fp2Opcode::CLASS_OFFSET;

    let (mut harness, bitwise) = if is_addsub {
        create_addsub_test_chips::<BLOCKS, BLOCK_SIZE>(&mut tester, config, offset)
    } else {
        create_muldiv_test_chips::<BLOCKS, BLOCK_SIZE>(&mut tester, config, offset)
    };

    for i in 0..num_ops {
        set_and_execute_fp2::<BLOCKS, BLOCK_SIZE, NUM_LIMBS>(
            &mut tester,
            &mut harness,
            &mut rng,
            &modulus,
            i == 0,
            is_addsub,
            offset,
        );
    }

    tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
fn test_fp2_addsub_2x32_small() {
    run_test_with_config::<2, 32, 32>(
        BigUint::from_str("357686312646216567629137").unwrap(),
        50,
        true,
    );
}

#[test]
fn test_fp2_addsub_2x32_secp256k1() {
    run_test_with_config::<2, 32, 32>(secp256k1_coord_prime(), 50, true);
}

#[test]
fn test_fp2_addsub_2x32_bn254() {
    run_test_with_config::<2, 32, 32>(BN254_MODULUS.clone(), 50, true);
}

#[test]
fn test_fp2_addsub_6x16() {
    run_test_with_config::<6, 16, 48>(BLS12_381_MODULUS.clone(), 50, true);
}

#[test]
fn test_fp2_muldiv_2x32_small() {
    run_test_with_config::<2, 32, 32>(
        BigUint::from_str("357686312646216567629137").unwrap(),
        50,
        false,
    );
}

#[test]
fn test_fp2_muldiv_2x32_secp256k1() {
    run_test_with_config::<2, 32, 32>(secp256k1_coord_prime(), 50, false);
}

#[test]
fn test_fp2_muldiv_2x32_bn254() {
    run_test_with_config::<2, 32, 32>(BN254_MODULUS.clone(), 50, false);
}

#[test]
fn test_fp2_muldiv_6x16() {
    run_test_with_config::<6, 16, 48>(BLS12_381_MODULUS.clone(), 50, false);
}
