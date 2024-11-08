use core::{
    cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd},
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul,
        MulAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
};

#[cfg(not(target_os = "zkvm"))]
use {super::biguint_to_limbs, num_bigint_dig::BigUint, num_traits::One};
#[cfg(target_os = "zkvm")]
use {
    axvm_platform::constants::{Custom0Funct3, Int256Funct7, CUSTOM_0},
    axvm_platform::custom_insn_r,
    core::{arch::asm, mem::MaybeUninit},
};

use crate::impl_bin_op;

/// A 256-bit unsigned integer type.
#[derive(Debug)]
#[repr(align(32), C)]
pub struct U256 {
    limbs: [u8; 32],
}

impl U256 {
    /// The maximum value of a U256.
    pub const MAX: Self = Self {
        limbs: [u8::MAX; 32],
    };

    /// The minimum value of a U256.
    pub const MIN: Self = Self { limbs: [0u8; 32] };

    /// The zero constant.
    pub const ZERO: Self = Self { limbs: [0u8; 32] };

    /// Value of this U256 as a BigUint.
    #[cfg(not(target_os = "zkvm"))]
    pub fn as_biguint(&self) -> BigUint {
        BigUint::from_bytes_le(&self.limbs)
    }

    /// Creates a new U256 from a BigUint.
    #[cfg(not(target_os = "zkvm"))]
    pub fn from_biguint(value: &BigUint) -> Self {
        Self {
            limbs: biguint_to_limbs(value),
        }
    }

    /// Creates a new U256 that equals to the given u8 value.
    pub fn from_u8(value: u8) -> Self {
        let mut limbs = [0u8; 32];
        limbs[0] = value;
        Self { limbs }
    }

    /// Creates a new U256 that equals to the given u32 value.
    pub fn from_u32(value: u32) -> Self {
        let mut limbs = [0u8; 32];
        limbs[..4].copy_from_slice(&value.to_le_bytes());
        Self { limbs }
    }

    /// Creates a new U256 that equals to the given u64 value.
    pub fn from_u64(value: u64) -> Self {
        let mut limbs = [0u8; 32];
        limbs[..8].copy_from_slice(&value.to_le_bytes());
        Self { limbs }
    }
}

impl_bin_op!(
    U256,
    Add,
    AddAssign,
    add,
    add_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::Add as u8,
    +=,
    |lhs: &U256, rhs: &U256| -> U256 {U256::from_biguint(&(lhs.as_biguint() + rhs.as_biguint()))}
);

impl_bin_op!(
    U256,
    Sub,
    SubAssign,
    sub,
    sub_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::Sub as u8,
    -=,
    |lhs: &U256, rhs: &U256| -> U256 {U256::from_biguint(&(U256::MAX.as_biguint() + BigUint::one() + lhs.as_biguint() - rhs.as_biguint()))}
);

impl_bin_op!(
    U256,
    Mul,
    MulAssign,
    mul,
    mul_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::Mul as u8,
    *=,
    |lhs: &U256, rhs: &U256| -> U256 {U256::from_biguint(&(lhs.as_biguint() * rhs.as_biguint()))}
);

impl_bin_op!(
    U256,
    BitXor,
    BitXorAssign,
    bitxor,
    bitxor_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::Xor as u8,
    ^=,
    |lhs: &U256, rhs: &U256| -> U256 {U256::from_biguint(&(lhs.as_biguint() ^ rhs.as_biguint()))}
);

impl_bin_op!(
    U256,
    BitAnd,
    BitAndAssign,
    bitand,
    bitand_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::And as u8,
    &=,
    |lhs: &U256, rhs: &U256| -> U256 {U256::from_biguint(&(lhs.as_biguint() & rhs.as_biguint()))}
);

impl_bin_op!(
    U256,
    BitOr,
    BitOrAssign,
    bitor,
    bitor_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::Or as u8,
    |=,
    |lhs: &U256, rhs: &U256| -> U256 {U256::from_biguint(&(lhs.as_biguint() | rhs.as_biguint()))}
);

impl_bin_op!(
    U256,
    Shl,
    ShlAssign,
    shl,
    shl_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::Sll as u8,
    <<=,
    |lhs: &U256, rhs: &U256| -> U256 {U256::from_biguint(&(lhs.as_biguint() << rhs.limbs[0] as usize))}
);

impl_bin_op!(
    U256,
    Shr,
    ShrAssign,
    shr,
    shr_assign,
    CUSTOM_0,
    Custom0Funct3::Int256 as u8,
    Int256Funct7::Srl as u8,
    >>=,
    |lhs: &U256, rhs: &U256| -> U256 {U256::from_biguint(&(lhs.as_biguint() >> rhs.limbs[0] as usize))}
);

impl PartialEq for U256 {
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
        return self.as_biguint() == other.as_biguint();
    }
}

impl Eq for U256 {}

impl PartialOrd for U256 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for U256 {
    fn cmp(&self, other: &Self) -> Ordering {
        #[cfg(target_os = "zkvm")]
        {
            let mut cmp_result = unsafe { MaybeUninit::<U256>::uninit().assume_init() };
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Sltu as u8,
                &mut cmp_result as *mut U256,
                self as *const Self,
                other as *const Self
            );
            if cmp_result.limbs[0] != 0 {
                return Ordering::Less;
            }
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Sltu as u8,
                &mut cmp_result as *mut U256,
                other as *const Self,
                self as *const Self
            );
            if cmp_result.limbs[0] != 0 {
                return Ordering::Greater;
            }
            return Ordering::Equal;
        }
        #[cfg(not(target_os = "zkvm"))]
        return self.as_biguint().cmp(&other.as_biguint());
    }
}

impl Clone for U256 {
    fn clone(&self) -> Self {
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<Self> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
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
