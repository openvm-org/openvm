use afs_primitives::bigint::utils::{secp256k1_coord_prime, secp256k1_scalar_prime};
use ax_sdk::{config::setup_tracing, utils::create_seeded_rng};
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::Rng;

use super::{ModularAddSubV2Chip, ModularMulDivV2Chip};
use crate::{
    arch::{
        instructions::{ModularArithmeticOpcode, UsizeOpcode},
        testing::MachineChipTestBuilder,
        ExecutionBridge, MachineChipWrapper, Rv32HeapAdapter,
    },
    program::Instruction,
    utils::biguint_to_limbs,
};

const NUM_LIMBS: usize = 32;
const LIMB_SIZE: usize = 8;
type F = BabyBear;

#[test]
fn test_modular_addsub() {
    setup_tracing();
    let coord_modulus = secp256k1_coord_prime();
    let scalar_modulus = secp256k1_scalar_prime();
    let mut tester: MachineChipTestBuilder<F> = MachineChipTestBuilder::default();

    let execution_bridge = ExecutionBridge::new(tester.execution_bus(), tester.program_bus());
    let memory_bridge = tester.memory_chip().borrow().memory_bridge();
    let adapter = Rv32HeapAdapter::new(execution_bridge, memory_bridge);
    let coord_chip = ModularAddSubV2Chip::<NUM_LIMBS, LIMB_SIZE>::new(
        coord_modulus.clone(),
        tester.memory_chip().borrow().range_checker.clone(),
        ModularArithmeticOpcode::default_offset(),
    );
    let scalar_chip = ModularAddSubV2Chip::<NUM_LIMBS, LIMB_SIZE>::new(
        scalar_modulus.clone(),
        tester.memory_chip().borrow().range_checker.clone(),
        ModularArithmeticOpcode::default_offset() + 4,
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

        let op = rng.gen_range(0..2); // 0 for add, 1 for sub
        let is_scalar = rng.gen_bool(0.5);
        let modulus = if is_scalar {
            scalar_modulus.clone()
        } else {
            coord_modulus.clone()
        };
        let chip = if is_scalar {
            scalar_chip.clone()
        } else {
            coord_chip.clone()
        };

        a %= modulus.clone();
        b %= modulus.clone();
        assert!(a < modulus);
        assert!(b < modulus);

        let r = chip.chip.execute(vec![a.clone(), b.clone()], vec![op == 0]);
        let r = r[0].clone();

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
            chip.offset + op,
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );

        let mut chip_wrapper = MachineChipWrapper::new(adapter.clone(), chip, tester.memory_chip());
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

#[test]
fn test_modular_muldiv() {
    setup_tracing();
    let coord_modulus = secp256k1_coord_prime();
    let scalar_modulus = secp256k1_scalar_prime();
    let mut tester: MachineChipTestBuilder<F> = MachineChipTestBuilder::default();

    let execution_bridge = ExecutionBridge::new(tester.execution_bus(), tester.program_bus());
    let memory_bridge = tester.memory_chip().borrow().memory_bridge();
    let adapter = Rv32HeapAdapter::new(execution_bridge, memory_bridge);
    let coord_chip = ModularMulDivV2Chip::<NUM_LIMBS, LIMB_SIZE>::new(
        coord_modulus.clone(),
        tester.memory_chip().borrow().range_checker.clone(),
        ModularArithmeticOpcode::default_offset(),
    );
    let scalar_chip = ModularMulDivV2Chip::<NUM_LIMBS, LIMB_SIZE>::new(
        scalar_modulus.clone(),
        tester.memory_chip().borrow().range_checker.clone(),
        ModularArithmeticOpcode::default_offset() + 4,
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

        let op = rng.gen_range(2..4); // 2 for mul, 3 for div
        let is_scalar = rng.gen_bool(0.5);
        let modulus = if is_scalar {
            scalar_modulus.clone()
        } else {
            coord_modulus.clone()
        };
        let chip = if is_scalar {
            scalar_chip.clone()
        } else {
            coord_chip.clone()
        };

        a %= modulus.clone();
        b %= modulus.clone();
        assert!(a < modulus);
        assert!(b < modulus);

        let r = chip.chip.execute(vec![a.clone(), b.clone()], vec![op == 2]);
        let r = r[0].clone();

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
            chip.offset + op,
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );

        let mut chip_wrapper = MachineChipWrapper::new(adapter.clone(), chip, tester.memory_chip());
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
