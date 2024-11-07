use core::{
    cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd},
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul,
        MulAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
};

#[cfg(not(target_os = "zkvm"))]
use {
    super::bigint_to_limbs,
    num_bigint_dig::{BigInt, Sign},
};
#[cfg(target_os = "zkvm")]
use {
    axvm_platform::constants::{Custom0Funct3, Int256Funct7, CUSTOM_0},
    axvm_platform::custom_insn_r,
    core::{arch::asm, mem::MaybeUninit},
};

use crate::impl_bin_op;

/// A 256-bit signed integer type.
#[derive(Clone, Debug)]
#[repr(align(32), C)]
pub struct I256 {
    limbs: [u8; 32],
}

impl I256 {
    /// Value of this I256 as a BigInt.
    #[cfg(not(target_os = "zkvm"))]
    pub fn as_bigint(&self) -> BigInt {
        let sign = if self.limbs[31] & 0x80 != 0 {
            Sign::Minus
        } else {
            Sign::Plus
        };
        BigInt::from_bytes_le(sign, &self.limbs)
    }

    /// Creates a new I256 from a BigInt.
    #[cfg(not(target_os = "zkvm"))]
    pub fn from_bigint(value: &BigInt) -> Self {
        Self {
            limbs: bigint_to_limbs(value),
        }
    }

    /// Creates a new I256 that equals to the given i8 value.
    pub fn from_i8(value: i8) -> Self {
        let mut limbs = if value < 0 { [u8::MAX; 32] } else { [0u8; 32] };
        limbs[0] = value as u8;
        Self { limbs }
    }

    /// Creates a new I256 that equals to the given i32 value.
    pub fn from_i32(value: i32) -> Self {
        let mut limbs = if value < 0 { [u8::MAX; 32] } else { [0u8; 32] };
        let value = value as u32;
        limbs[..4].copy_from_slice(&value.to_le_bytes());
        Self { limbs }
    }
}

impl_bin_op!(
    I256,
    Add,
    AddAssign,
    add,
    add_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::Add as u8,
    +=,
    |lhs: &I256, rhs: &I256| -> I256 {I256::from_bigint(&(lhs.as_bigint() + rhs.as_bigint()))}
);

impl_bin_op!(
    I256,
    Sub,
    SubAssign,
    sub,
    sub_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::Sub as u8,
    -=,
    |lhs: &I256, rhs: &I256| -> I256 {I256::from_bigint(&(lhs.as_bigint() - rhs.as_bigint()))}
);

impl_bin_op!(
    I256,
    Mul,
    MulAssign,
    mul,
    mul_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::Mul as u8,
    *=,
    |lhs: &I256, rhs: &I256| -> I256 {I256::from_bigint(&(lhs.as_bigint() * rhs.as_bigint()))}
);

impl_bin_op!(
    I256,
    BitXor,
    BitXorAssign,
    bitxor,
    bitxor_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::Xor as u8,
    ^=,
    |lhs: &I256, rhs: &I256| -> I256 {I256::from_bigint(&(lhs.as_bigint() ^ rhs.as_bigint()))}
);

impl_bin_op!(
    I256,
    BitAnd,
    BitAndAssign,
    bitand,
    bitand_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::And as u8,
    &=,
    |lhs: &I256, rhs: &I256| -> I256 {I256::from_bigint(&(lhs.as_bigint() & rhs.as_bigint()))}
);

impl_bin_op!(
    I256,
    BitOr,
    BitOrAssign,
    bitor,
    bitor_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::Or as u8,
    |=,
    |lhs: &I256, rhs: &I256| -> I256 {I256::from_bigint(&(lhs.as_bigint() | rhs.as_bigint()))}
);

impl_bin_op!(
    I256,
    Shl,
    ShlAssign,
    shl,
    shl_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::Sll as u8,
    <<=,
    |lhs: &I256, rhs: &I256| -> I256 {I256::from_bigint(&(lhs.as_bigint() << rhs.limbs[0] as usize))}
);

impl_bin_op!(
    I256,
    Shr,
    ShrAssign,
    shr,
    shr_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::Sra as u8,
    >>=,
    |lhs: &I256, rhs: &I256| -> I256 {I256::from_bigint(&(lhs.as_bigint() >> rhs.limbs[0] as usize))}
);

impl PartialEq for I256 {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_os = "zkvm")]
        {
            let mut is_equal: u32;
            unsafe {
                asm!("li {res}, 1",
                    ".insn b {opcode}, {func3}, {rs1}, {rs2}, 8",
                    "li {res}, 0",
                    opcode = const CUSTOM_0,
                    func3 = const Custom0Funct3::Beq256 as u8,
                    rs1 = in(reg) self as *const Self,
                    rs2 = in(reg) other as *const Self,
                    res = out(reg) is_equal
                );
            }
            return is_equal == 1;
        }
        #[cfg(not(target_os = "zkvm"))]
        return self.as_bigint() == other.as_bigint();
    }
}

impl Eq for I256 {}

impl PartialOrd for I256 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for I256 {
    fn cmp(&self, other: &Self) -> Ordering {
        #[cfg(target_os = "zkvm")]
        {
            let mut cmp_result = unsafe { MaybeUninit::<I256>::uninit().assume_init() };
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Slt as u8,
                &mut cmp_result as *mut I256,
                self as *const Self,
                other as *const Self
            );
            if cmp_result.limbs[0] != 0 {
                return Ordering::Less;
            }
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Slt as u8,
                &mut cmp_result as *mut I256,
                other as *const Self,
                self as *const Self
            );
            if cmp_result.limbs[0] != 0 {
                return Ordering::Greater;
            }
            return Ordering::Equal;
        }
        #[cfg(not(target_os = "zkvm"))]
        return self.as_bigint().cmp(&other.as_bigint());
    }
}
