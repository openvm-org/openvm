use afs_primitives::bigint::utils::{
    big_uint_mod_inverse, secp256k1_coord_prime, secp256k1_scalar_prime,
};
use ax_sdk::utils::create_seeded_rng;
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::Rng;

use super::{ModularAddSubV2Chip, ModularAddSubV2CoreChip, ModularMulDivV2CoreChip};
use crate::{
    arch::{
        instructions::{ModularArithmeticOpcode, UsizeOpcode},
        testing::{TestAdapterChip, VmChipTestBuilder},
        ExecutionBridge, VmChipWrapper,
    },
    rv32im::adapters::Rv32VecHeapAdapterChip,
    system::program::Instruction,
};

const NUM_LIMBS: usize = 32;
const LIMB_SIZE: usize = 8;
type F = BabyBear;
const READ_CELLS: usize = 64;

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
    let execution_bridge = ExecutionBridge::new(tester.execution_bus(), tester.program_bus());
    let core = ModularAddSubV2CoreChip::<NUM_LIMBS, LIMB_SIZE>::new(
        modulus.clone(),
        tester.memory_controller().borrow().range_checker.clone(),
        ModularArithmeticOpcode::default_offset() + opcode_offset,
    );
    let mut adapter = TestAdapterChip::new(vec![], vec![None], execution_bridge);
    let mut rng = create_seeded_rng();
    let num_tests = 50;
    let mut all_ops = vec![];
    let mut all_a = vec![];
    let mut all_b = vec![];

    // First loop: generate all random test data.
    for _ in 0..num_tests {
        let a_digits: Vec<_> = (0..NUM_LIMBS)
            .map(|_| rng.gen_range(0..(1 << LIMB_SIZE)))
            .collect();
        let mut a = BigUint::new(a_digits.clone());
        let b_digits: Vec<_> = (0..NUM_LIMBS)
            .map(|_| rng.gen_range(0..(1 << LIMB_SIZE)))
            .collect();
        let mut b = BigUint::new(b_digits.clone());
        let interface_reads: [BabyBear; READ_CELLS] = [a_digits, b_digits]
            .concat()
            .into_iter()
            .map(BabyBear::from_canonical_u32)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let op = rng.gen_range(0..2); // 0 for add, 1 for sub
        a %= modulus.clone();
        b %= modulus.clone();

        all_ops.push(op);
        all_a.push(a.clone());
        all_b.push(b.clone());
        adapter.prank_reads.push_back(interface_reads.to_vec());
    }
    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());
    // Second loop: actually run the tests.
    for i in 0..num_tests {
        let op = all_ops[i];
        let a = all_a[i].clone();
        let b = all_b[i].clone();
        assert!(a < modulus);
        assert!(b < modulus);
        let expected_answer = if op == 0 {
            (&a + &b) % &modulus
        } else {
            (&a + &modulus - &b) % &modulus
        };

        let r = chip
            .core
            .air
            .expr
            .execute(vec![a.clone(), b.clone()], vec![op == 0]);
        let r = r[0].clone();
        assert_eq!(expected_answer, r);
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
        let _address1 = 0;
        let _address2 = 128;
        let _address3 = 256;

        // TODO: uncomment memory part when switch to vectorized adapter
        /*
        tester.write_cell(ptr_as, addr_ptr1, BabyBear::from_canonical_usize(address1));
        tester.write_cell(ptr_as, addr_ptr2, BabyBear::from_canonical_usize(address2));
        tester.write_cell(ptr_as, addr_ptr3, BabyBear::from_canonical_usize(address3));

        let a_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(a.clone(), LIMB_SIZE).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address1, a_limbs);
        let b_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(b.clone(), LIMB_SIZE).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address2, b_limbs);
        */

        let instruction = Instruction::from_isize(
            chip.core.air.offset + op,
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );
        tester.execute(&mut chip, instruction);

        // TODO: uncomment when switch to vectorized adapter
        /*
        let r_limbs = biguint_to_limbs::<NUM_LIMBS>(r.clone(), LIMB_SIZE);
        for (i, &elem) in r_limbs.iter().enumerate() {
            let address = address3 + i;
            let read_val = tester.read_cell(data_as, address);
            assert_eq!(BabyBear::from_canonical_u32(elem), read_val);
        }
        */
    }
    let tester = tester.build().load(chip).finalize();

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
    let execution_bridge = ExecutionBridge::new(tester.execution_bus(), tester.program_bus());
    let core = ModularMulDivV2CoreChip::<NUM_LIMBS, LIMB_SIZE>::new(
        modulus.clone(),
        tester.memory_controller().borrow().range_checker.clone(),
        ModularArithmeticOpcode::default_offset() + opcode_offset,
    );
    let mut adapter = TestAdapterChip::new(vec![], vec![None], execution_bridge);
    let mut rng = create_seeded_rng();
    let num_tests = 50;
    let mut all_ops = vec![];
    let mut all_a = vec![];
    let mut all_b = vec![];

    // First loop: generate all random test data.
    for _ in 0..num_tests {
        let a_digits: Vec<_> = (0..NUM_LIMBS)
            .map(|_| rng.gen_range(0..(1 << LIMB_SIZE)))
            .collect();
        let mut a = BigUint::new(a_digits.clone());
        let b_digits: Vec<_> = (0..NUM_LIMBS)
            .map(|_| rng.gen_range(0..(1 << LIMB_SIZE)))
            .collect();
        let mut b = BigUint::new(b_digits.clone());
        let interface_reads: [BabyBear; READ_CELLS] = [a_digits, b_digits]
            .concat()
            .into_iter()
            .map(BabyBear::from_canonical_u32)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // let op = rng.gen_range(2..4); // 2 for mul, 3 for div
        let op = 2;
        a %= modulus.clone();
        b %= modulus.clone();

        all_ops.push(op);
        all_a.push(a.clone());
        all_b.push(b.clone());
        adapter.prank_reads.push_back(interface_reads.to_vec());
    }
    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());
    // Second loop: actually run the tests.
    for i in 0..num_tests {
        let op = all_ops[i];
        let a = all_a[i].clone();
        let b = all_b[i].clone();
        assert!(a < modulus);
        assert!(b < modulus);
        let expected_answer = if op == 2 {
            (&a * &b) % &modulus
        } else {
            (&a * big_uint_mod_inverse(&b, &modulus)) % &modulus
        };

        let r = chip
            .core
            .air
            .expr
            .execute(vec![a.clone(), b.clone()], vec![op == 2]);
        let r = r[0].clone();
        assert_eq!(expected_answer, r);
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
        let _address1 = 0;
        let _address2 = 128;
        let _address3 = 256;

        // TODO: uncomment memory part when switch to vectorized adapter
        /*
        tester.write_cell(ptr_as, addr_ptr1, BabyBear::from_canonical_usize(address1));
        tester.write_cell(ptr_as, addr_ptr2, BabyBear::from_canonical_usize(address2));
        tester.write_cell(ptr_as, addr_ptr3, BabyBear::from_canonical_usize(address3));

        let a_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(a.clone(), LIMB_SIZE).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address1, a_limbs);
        let b_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(b.clone(), LIMB_SIZE).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address2, b_limbs);
        */

        let instruction = Instruction::from_isize(
            chip.core.air.offset + op,
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );
        tester.execute(&mut chip, instruction);

        // TODO: uncomment when switch to vectorized adapter
        /*
        let r_limbs = biguint_to_limbs::<NUM_LIMBS>(r.clone(), LIMB_SIZE);
        for (i, &elem) in r_limbs.iter().enumerate() {
            let address = address3 + i;
            let read_val = tester.read_cell(data_as, address);
            assert_eq!(BabyBear::from_canonical_u32(elem), read_val);
        }
        */
    }
    let tester = tester.build().load(chip).finalize();

    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_coord_full_addsub() {
    let opcode_offset = 0;
    let modulus = secp256k1_coord_prime();
    test_full_addsub(opcode_offset, modulus);
}

#[test]
fn test_scalar_full_addsub() {
    let opcode_offset = 4;
    let modulus = secp256k1_scalar_prime();
    test_full_addsub(opcode_offset, modulus);
}

fn test_full_addsub(opcode_offset: usize, modulus: BigUint) {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let execution_bridge = ExecutionBridge::new(tester.execution_bus(), tester.program_bus());
    let chip = ModularAddSubV2Chip::<F, NUM_LIMBS, LIMB_SIZE>::new(
        Rv32VecHeapAdapterChip::<F, 1, 1, NUM_LIMBS, NUM_LIMBS>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        ModularAddSubV2CoreChip::<NUM_LIMBS, LIMB_SIZE>::new(
            modulus.clone(),
            tester.memory_controller().borrow().range_checker.clone(),
            opcode_offset,
        ),
        tester.memory_controller(),
    );
}
