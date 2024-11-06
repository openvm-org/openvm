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

/// A 256-bit unsigned integer type.
#[derive(Copy, Clone, Debug)]
#[repr(align(32), C)]
pub struct I256 {
    limbs: [u8; 32],
}

impl I256 {
    #[cfg(not(target_os = "zkvm"))]
    fn get_sign(&self) -> Sign {
        if self.limbs[31] & 0x80 != 0 {
            Sign::Minus
        } else {
            Sign::Plus
        }
    }
    /// Value of this I256 as a BigInt.
    #[cfg(not(target_os = "zkvm"))]
    pub fn as_bigint(&self) -> BigInt {
        BigInt::from_bytes_le(self.get_sign(), &self.limbs)
    }

    /// Creates a new I256 from a BigInt.
    #[cfg(not(target_os = "zkvm"))]
    pub fn from_bigint(value: &BigInt) -> Self {
        Self {
            limbs: bigint_to_limbs(&value),
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
        limbs[3] = (value >> 24) as u8;
        limbs[2] = (value >> 16) as u8;
        limbs[1] = (value >> 8) as u8;
        limbs[0] = value as u8;
        Self { limbs }
    }

    /// Allocating memory for I256 without initializing it.
    #[cfg(target_os = "zkvm")]
    #[inline(always)]
    fn alloc() -> Self {
        let uninit = MaybeUninit::<Self>::uninit();
        let init = unsafe { uninit.assume_init() };
        init
    }
}

/// Addition
impl<'a> AddAssign<&'a I256> for I256 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a I256) {
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
            *self = Self::from_bigint(&(self.as_bigint() + rhs.as_bigint()));
        }
    }
}

impl AddAssign<I256> for I256 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: I256) {
        *self += &rhs;
    }
}

impl<'a> Add<&'a I256> for &I256 {
    type Output = I256;
    #[inline(always)]
    fn add(self, rhs: &'a I256) -> I256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = I256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Add as u8,
                &mut ret as *mut I256,
                self as *const I256,
                rhs as *const I256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return I256::from_bigint(&(self.as_bigint() + rhs.as_bigint()));
    }
}

impl<'a> Add<&'a I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        self += &rhs;
        self
    }
}

/// Subtraction
impl<'a> SubAssign<&'a I256> for I256 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a I256) {
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
            *self = Self::from_bigint(&(self.as_bigint() - rhs.as_bigint()));
        }
    }
}

impl SubAssign<I256> for I256 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: I256) {
        *self -= &rhs;
    }
}

impl<'a> Sub<&'a I256> for &I256 {
    type Output = I256;
    #[inline(always)]
    fn sub(self, rhs: &'a I256) -> I256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = I256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Sub as u8,
                &mut ret as *mut I256,
                self as *const I256,
                rhs as *const I256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return I256::from_bigint(&(self.as_bigint() - rhs.as_bigint()));
    }
}

impl<'a> Sub<&'a I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, rhs: &'a Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl Sub<I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= &rhs;
        self
    }
}

/// Multiplication
impl<'a> MulAssign<&'a I256> for I256 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a I256) {
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
            *self = Self::from_bigint(&(self.as_bigint() * rhs.as_bigint()));
        }
    }
}

impl MulAssign<I256> for I256 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: I256) {
        *self *= &rhs;
    }
}

impl<'a> Mul<&'a I256> for &I256 {
    type Output = I256;
    #[inline(always)]
    fn mul(self, rhs: &'a I256) -> I256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = I256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Mul as u8,
                &mut ret as *mut I256,
                self as *const I256,
                rhs as *const I256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return I256::from_bigint(&(self.as_bigint() * rhs.as_bigint()));
    }
}

impl<'a> Mul<&'a I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, rhs: &'a Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl Mul<I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= &rhs;
        self
    }
}

/// Bitwise XOR
impl<'a> BitXorAssign<&'a I256> for I256 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: &'a I256) {
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
            *self = Self::from_bigint(&(self.as_bigint() ^ rhs.as_bigint()));
        }
    }
}

impl BitXorAssign<I256> for I256 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: I256) {
        *self ^= &rhs;
    }
}

impl<'a> BitXor<&'a I256> for &I256 {
    type Output = I256;
    #[inline(always)]
    fn bitxor(self, rhs: &'a I256) -> I256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = I256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Xor as u8,
                &mut ret as *mut I256,
                self as *const I256,
                rhs as *const I256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return I256::from_bigint(&(self.as_bigint() ^ rhs.as_bigint()));
    }
}

impl<'a> BitXor<&'a I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(mut self, rhs: &'a Self) -> Self::Output {
        self ^= rhs;
        self
    }
}

impl BitXor<I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(mut self, rhs: Self) -> Self::Output {
        self ^= &rhs;
        self
    }
}

/// Bitwise AND
impl<'a> BitAndAssign<&'a I256> for I256 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: &'a I256) {
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
            *self = Self::from_bigint(&(self.as_bigint() & rhs.as_bigint()));
        }
    }
}

impl BitAndAssign<I256> for I256 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: I256) {
        *self &= &rhs;
    }
}

impl<'a> BitAnd<&'a I256> for &I256 {
    type Output = I256;
    #[inline(always)]
    fn bitand(self, rhs: &'a I256) -> I256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = I256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::And as u8,
                &mut ret as *mut I256,
                self as *const I256,
                rhs as *const I256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return I256::from_bigint(&(self.as_bigint() & rhs.as_bigint()));
    }
}

impl<'a> BitAnd<&'a I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn bitand(mut self, rhs: &'a Self) -> Self::Output {
        self &= rhs;
        self
    }
}

impl BitAnd<I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn bitand(mut self, rhs: Self) -> Self::Output {
        self &= &rhs;
        self
    }
}

/// Bitwise OR
impl<'a> BitOrAssign<&'a I256> for I256 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: &'a I256) {
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
            *self = Self::from_bigint(&(self.as_bigint() | rhs.as_bigint()));
        }
    }
}

impl BitOrAssign<I256> for I256 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: I256) {
        *self |= &rhs;
    }
}

impl<'a> BitOr<&'a I256> for &I256 {
    type Output = I256;
    #[inline(always)]
    fn bitor(self, rhs: &'a I256) -> I256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = I256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Or as u8,
                &mut ret as *mut I256,
                self as *const I256,
                rhs as *const I256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return I256::from_bigint(&(self.as_bigint() | rhs.as_bigint()));
    }
}

impl<'a> BitOr<&'a I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn bitor(mut self, rhs: &'a Self) -> Self::Output {
        self |= rhs;
        self
    }
}

impl BitOr<I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn bitor(mut self, rhs: Self) -> Self::Output {
        self |= &rhs;
        self
    }
}

/// Left shift
impl<'a> ShlAssign<&'a I256> for I256 {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: &'a I256) {
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
            *self = Self::from_bigint(&(self.as_bigint() << rhs.limbs[0] as usize));
        }
    }
}

impl ShlAssign<I256> for I256 {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: I256) {
        *self <<= &rhs;
    }
}

impl<'a> Shl<&'a I256> for &I256 {
    type Output = I256;
    #[inline(always)]
    fn shl(self, rhs: &'a I256) -> I256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = I256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Sll as u8,
                &mut ret as *mut I256,
                self as *const I256,
                rhs as *const I256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return I256::from_bigint(&(self.as_bigint() << rhs.limbs[0] as usize));
    }
}

impl<'a> Shl<&'a I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn shl(mut self, rhs: &'a Self) -> Self::Output {
        self <<= rhs;
        self
    }
}

impl Shl<I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn shl(mut self, rhs: Self) -> Self::Output {
        self <<= &rhs;
        self
    }
}

/// Right shift
impl<'a> ShrAssign<&'a I256> for I256 {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: &'a I256) {
        #[cfg(target_os = "zkvm")]
        custom_insn_r!(
            CUSTOM_0,
            Custom0Funct3::Int256 as u8,
            Int256Funct7::Sra as u8,
            self as *mut Self,
            self as *const Self,
            rhs as *const Self
        );
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_bigint(&(self.as_bigint() >> rhs.limbs[0] as usize));
        }
    }
}

impl ShrAssign<I256> for I256 {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: I256) {
        *self >>= &rhs;
    }
}

impl<'a> Shr<&'a I256> for &I256 {
    type Output = I256;
    #[inline(always)]
    fn shr(self, rhs: &'a I256) -> I256 {
        #[cfg(target_os = "zkvm")]
        {
            let mut ret = I256::alloc();
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Sra as u8,
                &mut ret as *mut I256,
                self as *const I256,
                rhs as *const I256
            );
            return ret;
        }
        #[cfg(not(target_os = "zkvm"))]
        return I256::from_bigint(&(self.as_bigint() >> rhs.limbs[0] as usize));
    }
}

impl<'a> Shr<&'a I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn shr(mut self, rhs: &'a Self) -> Self::Output {
        self >>= rhs;
        self
    }
}

impl Shr<I256> for I256 {
    type Output = Self;
    #[inline(always)]
    fn shr(mut self, rhs: Self) -> Self::Output {
        self >>= &rhs;
        self
    }
}

impl PartialEq for I256 {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_os = "zkvm")]
        {
            let mut is_equal: u32 = 1;
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
            let mut is_less: u32 = 0;
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Slt as u8,
                &mut is_less as *mut u32,
                self as *const Self,
                other as *const Self
            );
            if is_less == 1 {
                return Ordering::Less;
            }
            let mut is_greater: u32 = 0;
            custom_insn_r!(
                CUSTOM_0,
                Custom0Funct3::Int256 as u8,
                Int256Funct7::Slt as u8,
                &mut is_greater as *mut u32,
                other as *const Self,
                self as *const Self
            );
            if is_greater == 1 {
                return Ordering::Greater;
            }
            return Ordering::Equal;
        }
        #[cfg(not(target_os = "zkvm"))]
        return self.as_bigint().cmp(&other.as_bigint());
    }
}
