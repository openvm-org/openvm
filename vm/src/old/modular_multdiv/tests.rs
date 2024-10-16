use afs_primitives::bigint::utils::{secp256k1_coord_prime, secp256k1_scalar_prime};
use ax_sdk::{config::setup_tracing, utils::create_seeded_rng};
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::Rng;

use crate::{
    arch::{
        instructions::{ModularArithmeticOpcode, UsizeOpcode},
        testing::VmChipTestBuilder,
    },
    old::modular_multdiv::{ModularMultDivChip, SECP256K1_COORD_PRIME, SECP256K1_SCALAR_PRIME},
    system::program::Instruction,
    utils::biguint_to_limbs,
};
const NUM_LIMBS: usize = 32;
const LIMB_SIZE: usize = 8;
const CARRY_LIMBS: usize = 2 * NUM_LIMBS - 1;
type F = BabyBear;

#[test]
fn test_modular_multdiv() {
    setup_tracing();

    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let mut coord_chip: ModularMultDivChip<F, CARRY_LIMBS, NUM_LIMBS, LIMB_SIZE> =
        ModularMultDivChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
            secp256k1_coord_prime(),
            0,
        );
    let mut scalar_chip: ModularMultDivChip<F, CARRY_LIMBS, NUM_LIMBS, LIMB_SIZE> =
        ModularMultDivChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
            secp256k1_scalar_prime(),
            4,
        );
    let mut rng = create_seeded_rng();
    let num_tests = 100;

    for _ in 0..num_tests {
        let a_digits = (0..NUM_LIMBS)
            .map(|_| rng.gen_range(0..(1 << LIMB_SIZE)))
            .collect();
        let mut a = BigUint::new(a_digits);
        let b_digits = (0..NUM_LIMBS)
            .map(|_| rng.gen_range(0..(1 << LIMB_SIZE)))
            .collect();
        let mut b = BigUint::new(b_digits);

        let is_scalar = rng.gen_bool(0.5);
        let opcode = ModularArithmeticOpcode::from_usize(rng.gen_range(2..4));
        let modulus = if is_scalar {
            SECP256K1_SCALAR_PRIME.clone()
        } else {
            SECP256K1_COORD_PRIME.clone()
        };
        a %= modulus.clone();
        b %= modulus.clone();
        assert!(a < modulus);
        assert!(b < modulus);

        let r = if is_scalar {
            scalar_chip.solve(opcode, a.clone(), b.clone())
        } else {
            coord_chip.solve(opcode, a.clone(), b.clone())
        };

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
        let address2 = 1024;
        let address3 = 2048;

        tester.write_cell(ptr_as, addr_ptr1, BabyBear::from_canonical_usize(address1));
        tester.write_cell(ptr_as, addr_ptr2, BabyBear::from_canonical_usize(address2));
        tester.write_cell(ptr_as, addr_ptr3, BabyBear::from_canonical_usize(address3));

        let a_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(a.clone(), LIMB_SIZE).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address1, a_limbs);
        let b_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(b.clone(), LIMB_SIZE).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address2, b_limbs);

        let instruction = Instruction::from_isize(
            opcode as usize + if is_scalar { 4 } else { 0 },
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );

        let chip = if is_scalar {
            &mut scalar_chip
        } else {
            &mut coord_chip
        };
        tester.execute(chip, instruction);
        let r_limbs = biguint_to_limbs::<NUM_LIMBS>(r.clone(), LIMB_SIZE);

        for (i, &elem) in r_limbs.iter().enumerate() {
            let address = address3 + i;
            let read_val = tester.read_cell(data_as, address);
            assert_eq!(BabyBear::from_canonical_u32(elem), read_val);
        }
    }
    let tester = tester.build().load(coord_chip).load(scalar_chip).finalize();

    tester.simple_test().expect("Verification failed");
}
