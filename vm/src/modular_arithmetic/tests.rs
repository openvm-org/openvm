use afs_primitives::bigint::utils::{
    big_uint_mod_inverse, secp256k1_coord_prime, secp256k1_scalar_prime,
};
use ax_sdk::{config::setup_tracing, utils::create_seeded_rng};
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::{Rng, RngCore};

use super::biguint_to_limbs;
use crate::{
    arch::{
        instructions::{
            Opcode::*, SECP256K1_COORD_MODULAR_ARITHMETIC_INSTRUCTIONS,
            SECP256K1_SCALAR_MODULAR_ARITHMETIC_INSTRUCTIONS,
        },
        testing::MachineChipTestBuilder,
    },
    cpu::trace::Instruction,
    modular_arithmetic::ModularArithmeticChip,
};

#[test]
fn test_modular_multiplication() {
    setup_tracing();
    const NUM_LIMBS: usize = 32;

    let mut tester: MachineChipTestBuilder<BabyBear> = MachineChipTestBuilder::default();
    let mut coord_add_chip = ModularArithmeticChip::new(
        tester.execution_bus(),
        tester.memory_chip(),
        secp256k1_coord_prime(),
        super::ModularArithmeticOp::Add,
    );
    let mut coord_sub_chip = ModularArithmeticChip::new(
        tester.execution_bus(),
        tester.memory_chip(),
        secp256k1_coord_prime(),
        super::ModularArithmeticOp::Sub,
    );
    let mut coord_mul_chip = ModularArithmeticChip::new(
        tester.execution_bus(),
        tester.memory_chip(),
        secp256k1_coord_prime(),
        super::ModularArithmeticOp::Mul,
    );
    let mut coord_div_chip = ModularArithmeticChip::new(
        tester.execution_bus(),
        tester.memory_chip(),
        secp256k1_coord_prime(),
        super::ModularArithmeticOp::Div,
    );
    let mut scalar_add_chip = ModularArithmeticChip::new(
        tester.execution_bus(),
        tester.memory_chip(),
        secp256k1_scalar_prime(),
        super::ModularArithmeticOp::Add,
    );
    let mut scalar_sub_chip = ModularArithmeticChip::new(
        tester.execution_bus(),
        tester.memory_chip(),
        secp256k1_scalar_prime(),
        super::ModularArithmeticOp::Sub,
    );
    let mut scalar_mul_chip = ModularArithmeticChip::new(
        tester.execution_bus(),
        tester.memory_chip(),
        secp256k1_scalar_prime(),
        super::ModularArithmeticOp::Mul,
    );
    let mut scalar_div_chip = ModularArithmeticChip::new(
        tester.execution_bus(),
        tester.memory_chip(),
        secp256k1_scalar_prime(),
        super::ModularArithmeticOp::Div,
    );
    let mut rng = create_seeded_rng();

    for _ in 0..100 {
        let num_digits = 8;

        let a_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
        let mut a = BigUint::new(a_digits);
        let b_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
        let mut b = BigUint::new(b_digits);
        let is_scalar = rng.gen_bool(0.5);
        let (modulus, opcode) = if is_scalar {
            (
                secp256k1_scalar_prime(),
                SECP256K1_SCALAR_MODULAR_ARITHMETIC_INSTRUCTIONS[rng.gen_range(0..4)],
            )
        } else {
            (
                secp256k1_coord_prime(),
                SECP256K1_COORD_MODULAR_ARITHMETIC_INSTRUCTIONS[rng.gen_range(0..4)],
            )
        };
        a %= modulus.clone();
        b %= modulus.clone();
        assert!(a < modulus);
        assert!(b < modulus);
        let r = match opcode {
            SECP256K1_COORD_ADD | SECP256K1_SCALAR_ADD => a.clone() + b.clone(),
            SECP256K1_COORD_SUB | SECP256K1_SCALAR_SUB => a.clone() + modulus.clone() - b.clone(),
            SECP256K1_COORD_MUL | SECP256K1_SCALAR_MUL => a.clone() * b.clone(),
            SECP256K1_COORD_DIV | SECP256K1_SCALAR_DIV => {
                a.clone() * big_uint_mod_inverse(&b, &modulus)
            }

            _ => panic!(),
        } % modulus;

        // Write to memories
        // For each bigunint (a, b, r), there are 2 writes:
        // 1. address_ptr which stores the actual address
        // 2. actual address which stores the biguint limbs
        // The write of result r is done in the chip.
        let ptr_as = 1;
        let addr_ptr1 = 0;
        let addr_ptr2 = 10;
        let addr_ptr3 = 20;

        let data_as = 2;
        let address1 = 0;
        let address2 = 1000;
        let address3 = 2000;

        tester.write_cell(ptr_as, addr_ptr1, BabyBear::from_canonical_usize(address1));
        tester.write_cell(ptr_as, addr_ptr2, BabyBear::from_canonical_usize(address2));
        tester.write_cell(ptr_as, addr_ptr3, BabyBear::from_canonical_usize(address3));

        let a_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(a.clone()).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address1, a_limbs);
        let b_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(b.clone()).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address2, b_limbs);

        let instruction = Instruction::from_isize(
            opcode,
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );

        let chip = match opcode {
            SECP256K1_COORD_ADD | SECP256K1_SCALAR_ADD => {
                if is_scalar {
                    &mut scalar_add_chip
                } else {
                    &mut coord_add_chip
                }
            }
            SECP256K1_COORD_SUB | SECP256K1_SCALAR_SUB => {
                if is_scalar {
                    &mut scalar_sub_chip
                } else {
                    &mut coord_sub_chip
                }
            }
            SECP256K1_COORD_MUL | SECP256K1_SCALAR_MUL => {
                if is_scalar {
                    &mut scalar_mul_chip
                } else {
                    &mut coord_mul_chip
                }
            }
            SECP256K1_COORD_DIV | SECP256K1_SCALAR_DIV => {
                if is_scalar {
                    &mut scalar_div_chip
                } else {
                    &mut coord_div_chip
                }
            }
            _ => panic!("Unexpected opcode"),
        };
        tester.execute(chip, instruction);
        let r_limbs = biguint_to_limbs(r.clone());
        for (i, &elem) in r_limbs.iter().enumerate() {
            let address = address3 + i;
            let read_val = tester.read_cell(data_as, address);
            assert_eq!(BabyBear::from_canonical_u32(elem), read_val);
        }
    }
    let tester = tester
        .build()
        .load(coord_add_chip)
        .load(coord_sub_chip)
        .load(coord_mul_chip)
        .load(coord_div_chip)
        .load(scalar_add_chip)
        .load(scalar_sub_chip)
        .load(scalar_mul_chip)
        .load(scalar_div_chip)
        .finalize();

    tester.simple_test().expect("Verification failed");
}
