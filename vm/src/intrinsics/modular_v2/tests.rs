use afs_primitives::bigint::utils::{secp256k1_coord_prime, secp256k1_scalar_prime};
use ax_sdk::{config::setup_tracing, utils::create_seeded_rng};
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::Rng;

use super::ModularAddSubV2CoreChip;
use crate::{
    arch::{testing::VmChipTestBuilder, ExecutionBridge, VmChipWrapper},
    rv32im::adapters::Rv32HeapAdapter,
    system::program::Instruction,
    utils::biguint_to_limbs,
};

const NUM_LIMBS: usize = 32;
const LIMB_SIZE: usize = 8;
type F = BabyBear;

#[test]
fn test_modular_add() {
    setup_tracing();
    let coord_modulus = secp256k1_coord_prime();
    let scalar_modulus = secp256k1_scalar_prime();
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();

    let execution_bridge = ExecutionBridge::new(tester.execution_bus(), tester.program_bus());
    let memory_bridge = tester.memory_chip().borrow().memory_bridge();
    let adapter = Rv32HeapAdapter::new(execution_bridge, memory_bridge);
    let coord_chip = ModularAddSubV2CoreChip::<NUM_LIMBS, LIMB_SIZE>::new(
        coord_modulus.clone(),
        tester.memory_chip().borrow().range_checker.clone(),
    );
    let scalar_chip = ModularAddSubV2CoreChip::<NUM_LIMBS, LIMB_SIZE>::new(
        scalar_modulus.clone(),
        tester.memory_chip().borrow().range_checker.clone(),
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

        let opcode = rng.gen_range(0..2);
        let is_scalar = rng.gen_bool(0.5);
        let modulus = if is_scalar {
            scalar_modulus.clone()
        } else {
            coord_modulus.clone()
        };

        a %= modulus.clone();
        b %= modulus.clone();
        assert!(a < modulus);
        assert!(b < modulus);

        let r = (&a + &b) % modulus;

        // TODO: make it handle + or -
        // let r = if is_scalar {
        //     scalar_chip.solve(
        //         ModularArithmeticOpcode::from_usize(opcode),
        //         a.clone(),
        //         b.clone(),
        //     )
        // } else {
        //     coord_chip.solve(
        //         ModularArithmeticOpcode::from_usize(opcode),
        //         a.clone(),
        //         b.clone(),
        //     )
        // };

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
            opcode + if is_scalar { 4 } else { 0 },
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );

        let chip = if is_scalar {
            scalar_chip.clone()
        } else {
            coord_chip.clone()
        };
        let mut chip_wrapper = VmChipWrapper::new(adapter.clone(), chip, tester.memory_chip());
        tester.execute(&mut chip_wrapper, instruction);
        let r_limbs = biguint_to_limbs::<NUM_LIMBS>(r.clone(), LIMB_SIZE);

        for (i, &elem) in r_limbs.iter().enumerate() {
            let address = address3 + i;
            let read_val = tester.read_cell(data_as, address);
            assert_eq!(BabyBear::from_canonical_u32(elem), read_val);
        }
    }
    // let tester = tester.build().load(coord_chip).load(scalar_chip).finalize();

    // tester.simple_test().expect("Verification failed");
}
