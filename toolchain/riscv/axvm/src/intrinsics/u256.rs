use core::{
    cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd},
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul,
        MulAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
};

#[cfg(not(target_os = "zkvm"))]
use {super::biguint_to_limbs, num_bigint_dig::BigUint};
#[cfg(target_os = "zkvm")]
use {
    axvm_platform::constants::{Custom0Funct3, Int256Funct7, CUSTOM_0},
    axvm_platform::custom_insn_r,
    core::{arch::asm, mem::MaybeUninit},
};

/// A 256-bit unsigned integer type.
#[derive(Clone, Debug)]
#[repr(align(32), C)]
pub struct U256 {
    limbs: [u8; 32],
}

impl U256 {
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

    /// Allocating memory for U256 without initializing it.
    #[cfg(target_os = "zkvm")]
    #[inline(always)]
    fn alloc() -> Self {
        let uninit = MaybeUninit::<Self>::uninit();
        let init = unsafe { uninit.assume_init() };
        init
    }
}

/// Addition
impl<'a> AddAssign<&'a U256> for U256 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        custom_insn_r!(
            CUSTOM_0,
            Custom0Funct3::Int256 as u8,
            Int256Funct7::Add as u8,
            self as *mut Self,
            self as *const Self,
            rhs as *const Self
        );
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() + rhs.as_biguint()));
        }
    }
}

impl AddAssign<U256> for U256 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: U256) {
        *self += &rhs;
    }
}

impl<'a> Add<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn add(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = U256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Add as u8,
                &mut ret as *mut U256,
                self as *const U256,
                rhs as *const U256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() + rhs.as_biguint()));
    }
}

impl<'a> Add<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        self += &rhs;
        self
    }
}

/// Subtraction
impl<'a> SubAssign<&'a U256> for U256 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        custom_insn_r!(
            CUSTOM_0,
            Custom0Funct3::Int256 as u8,
            Int256Funct7::Sub as u8,
            self as *mut Self,
            self as *const Self,
            rhs as *const Self
        );
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() - rhs.as_biguint()));
        }
    }
}

impl SubAssign<U256> for U256 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: U256) {
        *self -= &rhs;
    }
}

impl<'a> Sub<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn sub(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = U256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Sub as u8,
                &mut ret as *mut U256,
                self as *const U256,
                rhs as *const U256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() - rhs.as_biguint()));
    }
}

impl<'a> Sub<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, rhs: &'a Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl Sub<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= &rhs;
        self
    }
}

/// Multiplication
impl<'a> MulAssign<&'a U256> for U256 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        custom_insn_r!(
            CUSTOM_0,
            Custom0Funct3::Int256 as u8,
            Int256Funct7::Mul as u8,
            self as *mut Self,
            self as *const Self,
            rhs as *const Self
        );
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() * rhs.as_biguint()));
        }
    }
}

impl MulAssign<U256> for U256 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: U256) {
        *self *= &rhs;
    }
}

impl<'a> Mul<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn mul(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = U256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Mul as u8,
                &mut ret as *mut U256,
                self as *const U256,
                rhs as *const U256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() * rhs.as_biguint()));
    }
}

impl<'a> Mul<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, rhs: &'a Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl Mul<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= &rhs;
        self
    }
}

/// Bitwise XOR
impl<'a> BitXorAssign<&'a U256> for U256 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        custom_insn_r!(
            CUSTOM_0,
            Custom0Funct3::Int256 as u8,
            Int256Funct7::Xor as u8,
            self as *mut Self,
            self as *const Self,
            rhs as *const Self
        );
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() ^ rhs.as_biguint()));
        }
    }
}

impl BitXorAssign<U256> for U256 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: U256) {
        *self ^= &rhs;
    }
}

impl<'a> BitXor<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn bitxor(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = U256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Xor as u8,
                &mut ret as *mut U256,
                self as *const U256,
                rhs as *const U256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() ^ rhs.as_biguint()));
    }
}

impl<'a> BitXor<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(mut self, rhs: &'a Self) -> Self::Output {
        self ^= rhs;
        self
    }
}

impl BitXor<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(mut self, rhs: Self) -> Self::Output {
        self ^= &rhs;
        self
    }
}

/// Bitwise AND
impl<'a> BitAndAssign<&'a U256> for U256 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        custom_insn_r!(
            CUSTOM_0,
            Custom0Funct3::Int256 as u8,
            Int256Funct7::And as u8,
            self as *mut Self,
            self as *const Self,
            rhs as *const Self
        );
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() & rhs.as_biguint()));
        }
    }
}

impl BitAndAssign<U256> for U256 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: U256) {
        *self &= &rhs;
    }
}

impl<'a> BitAnd<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn bitand(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = U256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::And as u8,
                &mut ret as *mut U256,
                self as *const U256,
                rhs as *const U256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() & rhs.as_biguint()));
    }
}

impl<'a> BitAnd<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn bitand(mut self, rhs: &'a Self) -> Self::Output {
        self &= rhs;
        self
    }
}

impl BitAnd<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn bitand(mut self, rhs: Self) -> Self::Output {
        self &= &rhs;
        self
    }
}

/// Bitwise OR
impl<'a> BitOrAssign<&'a U256> for U256 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        custom_insn_r!(
            CUSTOM_0,
            Custom0Funct3::Int256 as u8,
            Int256Funct7::Or as u8,
            self as *mut Self,
            self as *const Self,
            rhs as *const Self
        );
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() | rhs.as_biguint()));
        }
    }
}

impl BitOrAssign<U256> for U256 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: U256) {
        *self |= &rhs;
    }
}

impl<'a> BitOr<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn bitor(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = U256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Or as u8,
                &mut ret as *mut U256,
                self as *const U256,
                rhs as *const U256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() | rhs.as_biguint()));
    }
}

impl<'a> BitOr<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn bitor(mut self, rhs: &'a Self) -> Self::Output {
        self |= rhs;
        self
    }
}

impl BitOr<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn bitor(mut self, rhs: Self) -> Self::Output {
        self |= &rhs;
        self
    }
}

/// Left shift
impl<'a> ShlAssign<&'a U256> for U256 {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        custom_insn_r!(
            CUSTOM_0,
            Custom0Funct3::Int256 as u8,
            Int256Funct7::Sll as u8,
            self as *mut Self,
            self as *const Self,
            rhs as *const Self
        );
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() << rhs.limbs[0] as usize));
        }
    }
}

impl ShlAssign<U256> for U256 {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: U256) {
        *self <<= &rhs;
    }
}

impl<'a> Shl<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn shl(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = U256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Sll as u8,
                &mut ret as *mut U256,
                self as *const U256,
                rhs as *const U256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() << rhs.limbs[0] as usize));
    }
}

impl<'a> Shl<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn shl(mut self, rhs: &'a Self) -> Self::Output {
        self <<= rhs;
        self
    }
}

impl Shl<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn shl(mut self, rhs: Self) -> Self::Output {
        self <<= &rhs;
        self
    }
}

/// Right shift
impl<'a> ShrAssign<&'a U256> for U256 {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        custom_insn_r!(
            CUSTOM_0,
            Custom0Funct3::Int256 as u8,
            Int256Funct7::Srl as u8,
            self as *mut Self,
            self as *const Self,
            rhs as *const Self
        );
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() >> rhs.limbs[0] as usize));
        }
    }
}

impl ShrAssign<U256> for U256 {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: U256) {
        *self >>= &rhs;
    }
}

impl<'a> Shr<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn shr(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = U256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Srl as u8,
                &mut ret as *mut U256,
                self as *const U256,
                rhs as *const U256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() >> rhs.limbs[0] as usize));
    }
}

impl<'a> Shr<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn shr(mut self, rhs: &'a Self) -> Self::Output {
        self >>= rhs;
        self
    }
}

impl Shr<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn shr(mut self, rhs: Self) -> Self::Output {
        self >>= &rhs;
        self
    }
}

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
            let mut cmp_result = U256::alloc();
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
