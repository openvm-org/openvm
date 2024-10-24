use ax_ecc_primitives::test_utils::{bn254_prime, fq12_random};
use axvm_instructions::FP12Opcode;
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::Fp12MultiplyCoreChip;
use crate::{
    arch::{testing::VmChipTestBuilder, ExecutionBridge, VmChipWrapper},
    intrinsics::{ecc::FIELD_ELEMENT_BITS, test_utils::write_ptr_reg},
    rv32im::adapters::{Rv32VecHeapAdapterChip, RV32_REGISTER_NUM_LIMBS},
    system::program::Instruction,
    utils::biguint_to_limbs,
};

const NUM_LIMBS: usize = 32;
const LIMB_BITS: usize = 8;
type F = BabyBear;

#[test]
fn test_fp12_multiply() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = bn254_prime();
    let core = Fp12MultiplyCoreChip::new(
        modulus.clone(),
        NUM_LIMBS,
        LIMB_BITS,
        FIELD_ELEMENT_BITS - 1,
        tester.memory_controller().borrow().range_checker.clone(),
        // FP12Opcode::default_offset(),
        0x198 + 2,
    );
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 14, 12, NUM_LIMBS, NUM_LIMBS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );

    let x = fq12_random();
    let y = fq12_random();
    let xi = vec![BigUint::from(9u32), BigUint::from(1u32)];

    let x_limbs = x
        .iter()
        .map(|x| {
            biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32)
        })
        .collect::<Vec<[BabyBear; NUM_LIMBS]>>();
    let y_limbs = y
        .iter()
        .map(|y| {
            biguint_to_limbs::<NUM_LIMBS>(y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32)
        })
        .collect::<Vec<[BabyBear; NUM_LIMBS]>>();
    let xi_limbs = xi
        .iter()
        .map(|xi| {
            biguint_to_limbs::<NUM_LIMBS>(xi.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32)
        })
        .collect::<Vec<[BabyBear; NUM_LIMBS]>>();

    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

    let res = chip
        .core
        .air
        .expr
        .execute([x, xi.clone(), y, xi].concat(), vec![]);
    assert_eq!(res.len(), 38);

    let ptr_as = 1;
    let addr_ptr1 = 0;
    let addr_ptr2 = 14 * RV32_REGISTER_NUM_LIMBS;
    let addr_ptr3 = addr_ptr2 + 14 * RV32_REGISTER_NUM_LIMBS;

    let data_as = 2;
    let address1 = 0u32;
    let address2 = 14 * 128u32;
    let address3 = address2 + 14 * 128u32;

    write_ptr_reg(&mut tester, ptr_as, addr_ptr1, address1);
    write_ptr_reg(&mut tester, ptr_as, addr_ptr2, address2);
    write_ptr_reg(&mut tester, ptr_as, addr_ptr3, address3);

    // Write x and y into address1 and address2
    for i in 0..12 {
        tester.write(data_as, address1 as usize + i * NUM_LIMBS, x_limbs[i]);
        tester.write(data_as, address2 as usize + i * NUM_LIMBS, y_limbs[i]);
    }

    // Tmp: Write xi into address2
    for i in 0..2 {
        tester.write(
            data_as,
            address1 as usize + (12 + i) * NUM_LIMBS,
            xi_limbs[i],
        );
        tester.write(
            data_as,
            address2 as usize + (12 + i) * NUM_LIMBS,
            xi_limbs[i],
        );
    }

    let instruction = Instruction::from_isize(
        chip.core.air.offset + FP12Opcode::MUL as usize,
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
