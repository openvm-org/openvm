use afs_primitives::{bigint::utils::secp256k1_coord_prime, ecc::SampleEcPoints};
use axvm_instructions::UsizeOpcode;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::{super::FIELD_ELEMENT_BITS, SwEcAddNeCoreChip};
use crate::{
    arch::{instructions::EccOpcode, testing::VmChipTestBuilder, VmChipWrapper},
    intrinsics::ecc::sw::SwEcDoubleCoreChip,
    rv32im::adapters::{Rv32VecHeapAdapterChip, RV32_REGISTER_NUM_LIMBS},
    system::program::Instruction,
    utils::biguint_to_limbs,
};

const NUM_LIMBS: usize = 32;
const LIMB_BITS: usize = 8;
type F = BabyBear;

#[test]
fn test_add_ne() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = secp256k1_coord_prime();
    let core = SwEcAddNeCoreChip::new(
        modulus.clone(),
        NUM_LIMBS,
        LIMB_BITS,
        FIELD_ELEMENT_BITS - 1,
        tester.memory_controller().borrow().range_checker.clone(),
        EccOpcode::default_offset(),
    );
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 2, 2, NUM_LIMBS, NUM_LIMBS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );

    let (p1_x, p1_y) = SampleEcPoints[0].clone();
    let (p2_x, p2_y) = SampleEcPoints[1].clone();

    let p1_x_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p1_y_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p2_x_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p2_x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p2_y_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p2_y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);

    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

    let r = chip
        .core
        .air
        .expr
        .execute(vec![p1_x, p1_y, p2_x, p2_y], vec![]);
    assert_eq!(r.len(), 3); // lambda, x3, y3
    assert_eq!(r[1], SampleEcPoints[2].0);
    assert_eq!(r[2], SampleEcPoints[2].1);

    let ptr_as = 1;
    let addr_ptr1 = 0;
    let addr_ptr2 = 3 * RV32_REGISTER_NUM_LIMBS;
    let addr_ptr3 = 6 * RV32_REGISTER_NUM_LIMBS;

    let data_as = 2;
    let address1 = 0u32;
    let address2 = 128u32;
    let address3 = 256u32;
    let mut write_reg = |reg_addr, value: u32| {
        tester.write(
            ptr_as,
            reg_addr,
            value.to_le_bytes().map(BabyBear::from_canonical_u8),
        );
    };

    write_reg(addr_ptr1, address1);
    write_reg(addr_ptr2, address2);
    write_reg(addr_ptr3, address3);
    tester.write(data_as, address1 as usize, p1_x_limbs);
    tester.write(data_as, address1 as usize + NUM_LIMBS, p1_y_limbs);
    tester.write(data_as, address2 as usize, p2_x_limbs);
    tester.write(data_as, address2 as usize + NUM_LIMBS, p2_y_limbs);

    let instruction = Instruction::from_isize(
        chip.core.air.offset + EccOpcode::EC_ADD_NE as usize,
        addr_ptr3 as isize,
        addr_ptr1 as isize,
        addr_ptr2 as isize,
        ptr_as as isize,
        data_as as isize,
    );
    tester.execute(&mut chip, instruction);

    let tester = tester.build().load(chip).finalize();

    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_double() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = secp256k1_coord_prime();
    let core = SwEcDoubleCoreChip::new(
        modulus.clone(),
        NUM_LIMBS,
        LIMB_BITS,
        FIELD_ELEMENT_BITS - 1,
        tester.memory_controller().borrow().range_checker.clone(),
        EccOpcode::default_offset(),
    );
    let adapter = Rv32VecHeapAdapterChip::<F, 1, 2, 2, NUM_LIMBS, NUM_LIMBS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );

    let (p1_x, p1_y) = SampleEcPoints[1].clone();
    let p1_x_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p1_y_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);

    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

    let r = chip.core.air.expr.execute(vec![p1_x, p1_y], vec![]);
    assert_eq!(r.len(), 3); // lambda, x3, y3
    assert_eq!(r[1], SampleEcPoints[3].0);
    assert_eq!(r[2], SampleEcPoints[3].1);
    let ptr_as = 1;
    let addr_ptr1 = 0;
    let addr_ptr2 = 0; // unused
    let addr_ptr3 = 6 * RV32_REGISTER_NUM_LIMBS;

    let data_as = 2;
    let address1 = 0u32;
    let address3 = 256u32;
    let mut write_reg = |reg_addr, value: u32| {
        tester.write(
            ptr_as,
            reg_addr,
            value.to_le_bytes().map(BabyBear::from_canonical_u8),
        );
    };
    write_reg(addr_ptr1, address1);
    write_reg(addr_ptr3, address3);
    tester.write(data_as, address1 as usize, p1_x_limbs);
    tester.write(data_as, address1 as usize + NUM_LIMBS, p1_y_limbs);

    let instruction = Instruction::from_isize(
        chip.core.air.offset + EccOpcode::EC_DOUBLE as usize,
        addr_ptr3 as isize,
        addr_ptr1 as isize,
        addr_ptr2 as isize,
        ptr_as as isize,
        data_as as isize,
    );
    tester.execute(&mut chip, instruction);
}
