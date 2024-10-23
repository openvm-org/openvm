use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use crate::arch::testing::VmChipTestBuilder;

pub fn write_ptr_reg(
    tester: &mut VmChipTestBuilder<BabyBear>,
    ptr_as: usize,
    reg_addr: usize,
    value: u32,
) {
    tester.write(
        ptr_as,
        reg_addr,
        value.to_le_bytes().map(BabyBear::from_canonical_u8),
    );
}
