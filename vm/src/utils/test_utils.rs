use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use crate::{
    arch::testing::VmChipTestBuilder, rv32im::adapters::RV32_REGISTER_NUM_LIMBS,
    system::program::Instruction,
};

// little endian.
// Warning: This function only returns the last NUM_LIMBS*LIMB_BITS bits of
//          the input, while the input can have more than that.
pub fn biguint_to_limbs<const NUM_LIMBS: usize>(
    mut x: BigUint,
    limb_size: usize,
) -> [u32; NUM_LIMBS] {
    let mut result = [0; NUM_LIMBS];
    let base = BigUint::from_u32(1 << limb_size).unwrap();
    for r in result.iter_mut() {
        *r = (x.clone() % &base).to_u32().unwrap();
        x /= &base;
    }
    assert!(x.is_zero());
    result
}

pub fn rv32_write_memory<const NUM_LIMBS: usize>(
    tester: &mut VmChipTestBuilder<BabyBear>,
    addr1_writes: Vec<[BabyBear; NUM_LIMBS]>,
    addr2_writes: Vec<[BabyBear; NUM_LIMBS]>,
    opcode_with_offset: usize,
) -> Instruction<BabyBear> {
    let ptr_as = 1;
    let addr_ptr1 = 0;
    let addr_ptr2 = if addr2_writes.is_empty() {
        0
    } else {
        3 * RV32_REGISTER_NUM_LIMBS
    };
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
    if !addr2_writes.is_empty() {
        write_reg(addr_ptr2, address2);
    }
    write_reg(addr_ptr3, address3);
    for (i, &addr1_write) in addr1_writes.iter().enumerate() {
        tester.write(data_as, address1 as usize + i * NUM_LIMBS, addr1_write);
    }
    for (i, &addr2_write) in addr2_writes.iter().enumerate() {
        tester.write(data_as, address2 as usize + i * NUM_LIMBS, addr2_write);
    }

    Instruction::from_isize(
        opcode_with_offset,
        addr_ptr3 as isize,
        addr_ptr1 as isize,
        addr_ptr2 as isize,
        ptr_as as isize,
        data_as as isize,
    )
}
