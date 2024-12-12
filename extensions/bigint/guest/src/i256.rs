use core::{
    cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd},
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul,
        MulAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
};

#[cfg(not(target_os = "zkvm"))]
use {super::bigint_to_limbs, num_bigint_dig::BigInt};
#[cfg(target_os = "zkvm")]
use {
    super::{Int256Funct7, BEQ256_FUNCT3, INT256_FUNCT3, OPCODE},
    core::{arch::asm, mem::MaybeUninit},
    openvm_platform::custom_insn_r,
};

use crate::impl_bin_op;

/// A 256-bit signed integer type.
#[derive(Debug)]
#[repr(align(32), C)]
pub struct I256 {
    limbs: [u8; 32],
}

impl I256 {
    /// The minimum value of an I256.
    pub const MIN: Self = Self::generate_min();

    /// The maximum value of an I256.
    pub const MAX: Self = Self::generate_max();

    /// The zero constant.
    pub const ZERO: Self = Self { limbs: [0u8; 32] };

    /// Value of this I256 as a BigInt.
    #[cfg(not(target_os = "zkvm"))]
    pub fn as_bigint(&self) -> BigInt {
        BigInt::from_signed_bytes_le(&self.limbs)
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

    /// Creates a new I256 that equals to the given i64 value.
    pub fn from_i64(value: i64) -> Self {
        let mut limbs = if value < 0 { [u8::MAX; 32] } else { [0u8; 32] };
        let value = value as u64;
        limbs[..8].copy_from_slice(&value.to_le_bytes());
        Self { limbs }
    }

    /// A constant private helper function to generate the minimum value of an I256.
    const fn generate_min() -> Self {
        let mut limbs = [0u8; 32];
        limbs[31] = i8::MIN as u8;
        Self { limbs }
    }

    /// A constant private helper function to generate the maximum value of an I256.
    const fn generate_max() -> Self {
        let mut limbs = [u8::MAX; 32];
        limbs[31] = i8::MAX as u8;
        Self { limbs }
    }
}

impl_bin_op!(
    I256,
    Add,
    AddAssign,
    add,
    add_assign,
    OPCODE,
    INT256_FUNCT3,
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
    OPCODE,
    INT256_FUNCT3,
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
    OPCODE,
    INT256_FUNCT3,
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
    OPCODE,
    INT256_FUNCT3,
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
    OPCODE,
    INT256_FUNCT3,
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
    OPCODE,
    INT256_FUNCT3,
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
    OPCODE,
    INT256_FUNCT3,
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
    OPCODE,
    INT256_FUNCT3,
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
                    opcode = const OPCODE,
                    func3 = const BEQ256_FUNCT3,
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
            let mut cmp_result = MaybeUninit::<I256>::uninit();
            custom_insn_r!(
                OPCODE,
                INT256_FUNCT3,
                Int256Funct7::Slt as u8,
                cmp_result.as_mut_ptr(),
                self as *const Self,
                other as *const Self
            );
            let mut cmp_result = unsafe { cmp_result.assume_init() };
            if cmp_result.limbs[0] != 0 {
                return Ordering::Less;
            }
            custom_insn_r!(
                OPCODE,
                INT256_FUNCT3,
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

impl Clone for I256 {
    fn clone(&self) -> Self {
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<Self> = MaybeUninit::uninit();
            custom_insn_r!(
                OPCODE,
                INT256_FUNCT3,
                Int256Funct7::Add as u8,
                uninit.as_mut_ptr(),
                self as *const Self,
                &Self::ZERO as *const Self
            );
            unsafe { uninit.assume_init() }
        }
        #[cfg(not(target_os = "zkvm"))]
        return Self { limbs: self.limbs };
    }
}
