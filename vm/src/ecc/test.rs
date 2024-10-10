use afs_primitives::ecc::SampleEcPoints;
use ax_sdk::config::setup_tracing;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::{EcAddUnequalChip, EcDoubleChip, LIMB_SIZE, NUM_LIMBS, TWO_NUM_LIMBS};
use crate::{
    arch::{instructions::EccOpcode, testing::MachineChipTestBuilder},
    program::Instruction,
    utils::biguint_to_limbs,
};

#[test]
fn test_ec_add() {
    setup_tracing();

    let mut tester: MachineChipTestBuilder<BabyBear> = MachineChipTestBuilder::default();

    let mut ec_chip = EcAddUnequalChip::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        0,
    );

    let (p1_x, p1_y) = SampleEcPoints[0].clone();
    let (p2_x, p2_y) = SampleEcPoints[1].clone();

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
    let mut p1_limbs = [BabyBear::zero(); TWO_NUM_LIMBS];
    p1_limbs[..NUM_LIMBS].copy_from_slice(
        &biguint_to_limbs::<NUM_LIMBS>(p1_x, LIMB_SIZE).map(BabyBear::from_canonical_u32),
    );
    p1_limbs[NUM_LIMBS..].copy_from_slice(
        &biguint_to_limbs::<NUM_LIMBS>(p1_y, LIMB_SIZE).map(BabyBear::from_canonical_u32),
    );
    tester.write(data_as, address1, p1_limbs);
    let mut p2_limbs = [BabyBear::zero(); TWO_NUM_LIMBS];
    p2_limbs[..NUM_LIMBS].copy_from_slice(
        &biguint_to_limbs::<NUM_LIMBS>(p2_x, LIMB_SIZE).map(BabyBear::from_canonical_u32),
    );
    p2_limbs[NUM_LIMBS..].copy_from_slice(
        &biguint_to_limbs::<NUM_LIMBS>(p2_y, LIMB_SIZE).map(BabyBear::from_canonical_u32),
    );
    tester.write(data_as, address2, p2_limbs);

    let add_opcode = EccOpcode::EC_ADD_NE;
    let instructions = Instruction::from_isize(
        add_opcode as usize,
        addr_ptr3 as isize,
        addr_ptr1 as isize,
        addr_ptr2 as isize,
        ptr_as as isize,
        data_as as isize,
    );

    // Do 3 times to trigger padding.
    tester.execute(&mut ec_chip, instructions.clone());
    tester.execute(&mut ec_chip, instructions.clone());
    tester.execute(&mut ec_chip, instructions);

    let (p3_x, p3_y) = SampleEcPoints[2].clone();
    let mut p3_limbs = [BabyBear::zero(); TWO_NUM_LIMBS];
    p3_limbs[..NUM_LIMBS].copy_from_slice(
        &biguint_to_limbs::<NUM_LIMBS>(p3_x, LIMB_SIZE).map(BabyBear::from_canonical_u32),
    );
    p3_limbs[NUM_LIMBS..].copy_from_slice(
        &biguint_to_limbs::<NUM_LIMBS>(p3_y, LIMB_SIZE).map(BabyBear::from_canonical_u32),
    );
    let read_p3 = tester.read::<TWO_NUM_LIMBS>(data_as, address3);
    assert_eq!(p3_limbs, read_p3);

    let tester = tester.build().load(ec_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_ec_double() {
    setup_tracing();

    let mut tester: MachineChipTestBuilder<BabyBear> = MachineChipTestBuilder::default();

    let mut ec_chip = EcDoubleChip::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        0,
    );

    let (p1_x, p1_y) = SampleEcPoints[1].clone();

    let ptr_as = 1;
    let addr_ptr1 = 0;
    let addr_ptr2 = 10;

    let data_as = 2;
    let address1 = 0;
    let address2 = 1024;

    tester.write_cell(ptr_as, addr_ptr1, BabyBear::from_canonical_usize(address1));
    tester.write_cell(ptr_as, addr_ptr2, BabyBear::from_canonical_usize(address2));
    let mut p1_limbs = [BabyBear::zero(); TWO_NUM_LIMBS];
    p1_limbs[..NUM_LIMBS].copy_from_slice(
        &biguint_to_limbs::<NUM_LIMBS>(p1_x, LIMB_SIZE).map(BabyBear::from_canonical_u32),
    );
    p1_limbs[NUM_LIMBS..].copy_from_slice(
        &biguint_to_limbs::<NUM_LIMBS>(p1_y, LIMB_SIZE).map(BabyBear::from_canonical_u32),
    );
    tester.write(data_as, address1, p1_limbs);
    tester.write(data_as, address1, p1_limbs);

    let double_opcode = EccOpcode::EC_DOUBLE;
    let instructions = Instruction::from_isize(
        double_opcode as usize,
        addr_ptr2 as isize,
        addr_ptr1 as isize,
        0, // unused c
        ptr_as as isize,
        data_as as isize,
    );

    // Do 3 times to trigger padding.
    tester.execute(&mut ec_chip, instructions.clone());
    tester.execute(&mut ec_chip, instructions.clone());
    tester.execute(&mut ec_chip, instructions);

    let (p2_x, p2_y) = SampleEcPoints[3].clone();
    let mut p2_limbs = [BabyBear::zero(); TWO_NUM_LIMBS];
    p2_limbs[..NUM_LIMBS].copy_from_slice(
        &biguint_to_limbs::<NUM_LIMBS>(p2_x, LIMB_SIZE).map(BabyBear::from_canonical_u32),
    );
    p2_limbs[NUM_LIMBS..].copy_from_slice(
        &biguint_to_limbs::<NUM_LIMBS>(p2_y, LIMB_SIZE).map(BabyBear::from_canonical_u32),
    );
    let read_p2 = tester.read::<TWO_NUM_LIMBS>(data_as, address2);
    assert_eq!(p2_limbs, read_p2);

    let tester = tester.build().load(ec_chip).finalize();
    tester.simple_test().expect("Verification failed");
}
