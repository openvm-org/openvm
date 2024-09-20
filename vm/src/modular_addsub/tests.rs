use afs_primitives::bigint::utils::{secp256k1_coord_prime, secp256k1_scalar_prime};
use ax_sdk::{config::setup_tracing, utils::create_seeded_rng};
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::Rng;

use super::{ModularAddSubChip, SECP256K1_COORD_PRIME, SECP256K1_SCALAR_PRIME};
use crate::{
    arch::{
        instructions::{Opcode::*, MODULAR_ADDSUB_INSTRUCTIONS},
        testing::MachineChipTestBuilder,
    },
    program::Instruction,
};

const NUM_LIMBS: usize = 32;
const LIMB_SIZE: usize = 8;
type F = BabyBear;

#[test]
fn test_modular_addsub() {
    setup_tracing();

    let mut tester: MachineChipTestBuilder<F> = MachineChipTestBuilder::default();
    let mut coord_chip: ModularAddSubChip<F, NUM_LIMBS, LIMB_SIZE> = ModularAddSubChip::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        secp256k1_coord_prime(),
    );
    let mut scalar_chip: ModularAddSubChip<F, NUM_LIMBS, LIMB_SIZE> = ModularAddSubChip::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        secp256k1_scalar_prime(),
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

        let opcode = MODULAR_ADDSUB_INSTRUCTIONS[rng.gen_range(0..4)];

        let (is_scalar, modulus) = match opcode {
            SECP256K1_SCALAR_ADD | SECP256K1_SCALAR_SUB => (true, SECP256K1_SCALAR_PRIME.clone()),
            SECP256K1_COORD_ADD | SECP256K1_COORD_SUB => (false, SECP256K1_COORD_PRIME.clone()),
            _ => unreachable!(),
        };

        a %= modulus.clone();
        b %= modulus.clone();
        assert!(a < modulus);
        assert!(b < modulus);

        let r = ModularAddSubChip::<F, NUM_LIMBS, LIMB_SIZE>::solve(opcode, a.clone(), b.clone());

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
        let address2 = 100;
        let address3 = 200;

        tester.write_cell(ptr_as, addr_ptr1, BabyBear::from_canonical_usize(address1));
        tester.write_cell(ptr_as, addr_ptr2, BabyBear::from_canonical_usize(address2));
        tester.write_cell(ptr_as, addr_ptr3, BabyBear::from_canonical_usize(address3));

        let a_limbs: [BabyBear; NUM_LIMBS] =
            ModularAddSubChip::<F, NUM_LIMBS, LIMB_SIZE>::biguint_to_limbs(a.clone())
                .map(BabyBear::from_canonical_u32);
        tester.write(data_as, address1, a_limbs);
        let b_limbs: [BabyBear; NUM_LIMBS] =
            ModularAddSubChip::<F, NUM_LIMBS, LIMB_SIZE>::biguint_to_limbs(b.clone())
                .map(BabyBear::from_canonical_u32);
        tester.write(data_as, address2, b_limbs);

        let instruction = Instruction::from_isize(
            opcode,
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
        let r_limbs = ModularAddSubChip::<F, NUM_LIMBS, LIMB_SIZE>::biguint_to_limbs(r.clone());

        for (i, &elem) in r_limbs.iter().enumerate() {
            let address = address3 + i;
            let read_val = tester.read_cell(data_as, address);
            assert_eq!(BabyBear::from_canonical_u32(elem), read_val);
        }
    }
    let tester = tester.build().load(coord_chip).load(scalar_chip).finalize();

    tester.simple_test().expect("Verification failed");
}
