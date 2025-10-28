pub const MAX_LIMBS: usize = 49; // Actual max limbs is 48; set to 49 for overflow
pub const LIMB_BITS: usize = 8;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExprType {
    Input = 0,
    Var = 1,
    Const = 2,
    Add = 3,
    Sub = 4,
    Mul = 5,
    Div = 6,
    IntAdd = 7,
    IntMul = 8,
    Select = 9,
}
