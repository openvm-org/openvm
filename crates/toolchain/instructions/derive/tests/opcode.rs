use openvm_instructions::LocalOpcode;
use openvm_instructions_derive::LocalOpcode;
use strum_macros::{EnumCount, EnumIter, FromRepr};

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x0]
#[repr(usize)]
pub enum TestOpcode {
    A,
    B,
    C,
}

#[derive(LocalOpcode)]
#[opcode_offset = 0x123]
pub struct WrapperOpcode(TestOpcode);

#[test]
fn test_opcode_macro() {
    assert_eq!(TestOpcode::A.local_usize(), 0);
    assert_eq!(TestOpcode::B.local_usize(), 1);
    assert_eq!(TestOpcode::C.local_usize(), 2);
    assert_eq!(TestOpcode::CLASS_OFFSET, 0x0);

    assert_eq!(WrapperOpcode::CLASS_OFFSET, 0x123);
    assert_eq!(
        WrapperOpcode(TestOpcode::A).global_opcode().as_usize(),
        0x123
    );
    assert_eq!(
        WrapperOpcode(TestOpcode::B).global_opcode().as_usize(),
        0x124
    );
    assert_eq!(
        WrapperOpcode(TestOpcode::C).global_opcode().as_usize(),
        0x125
    );
}
