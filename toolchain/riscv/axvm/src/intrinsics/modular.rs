#[cfg(target_os = "zkvm")]
use core::mem::MaybeUninit;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[cfg(target_os = "zkvm")]
use axvm_platform::{
    constants::{Custom1Funct3, ModArithBaseFunct7, CUSTOM_1},
    custom_insn_r,
};
use hex_literal::hex;
#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::{traits::ModInverse, BigUint, Sign, ToBigInt};

#[cfg(not(target_os = "zkvm"))]
use crate::intrinsics::biguint_to_limbs;

const LIMBS: usize = 32;

/// Class to represent an integer modulo N, which is currently hard-coded to be the
/// secp256k1 prime.
#[derive(Clone, Eq)]
#[repr(C, align(32))]
pub struct IntModN([u8; LIMBS]);

impl IntModN {
    const MODULUS: [u8; LIMBS] =
        hex!("2FFCFFFF FEFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF");
    const _MOD_IDX: usize = 0;

    /// The zero element of the field.
    pub const ZERO: Self = Self([0; LIMBS]);

    /// Creates a new IntModN from an array of bytes.
    pub fn from_bytes(bytes: [u8; LIMBS]) -> Self {
        Self(bytes)
    }

    /// Creates a new IntModN from a u32.
    pub fn from_u32(val: u32) -> Self {
        let mut bytes = [0; LIMBS];
        bytes[..4].copy_from_slice(&val.to_le_bytes());
        Self(bytes)
    }

    /// Value of this IntModN as an array of bytes.
    pub fn as_bytes(&self) -> &[u8; LIMBS] {
        &(self.0)
    }

    /// Returns MODULUS as an array of bytes.
    pub fn modulus() -> [u8; LIMBS] {
        Self::MODULUS
    }

    /// Creates a new IntModN from a BigUint.
    #[cfg(not(target_os = "zkvm"))]
    pub fn from_biguint(biguint: BigUint) -> Self {
        Self(biguint_to_limbs(&biguint))
    }

    /// Value of this IntModN as a BigUint.
    #[cfg(not(target_os = "zkvm"))]
    pub fn as_biguint(&self) -> BigUint {
        BigUint::from_bytes_le(self.as_bytes())
    }

    /// Modulus N as a BigUint.
    #[cfg(not(target_os = "zkvm"))]
    pub fn modulus_biguint() -> BigUint {
        BigUint::from_bytes_le(&Self::MODULUS)
    }

    #[inline(always)]
    fn add_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(
                (self.as_biguint() + other.as_biguint()) % Self::modulus_biguint(),
            );
        }
        #[cfg(target_os = "zkvm")]
        {
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ModularArithmetic as usize,
                ModArithBaseFunct7::AddMod as usize,
                self as *mut Self,
                self as *const Self,
                other as *const Self
            )
        }
    }

    #[inline(always)]
    fn sub_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let modulus = Self::modulus_biguint();
            *self = Self::from_biguint(
                (self.as_biguint() + modulus.clone() - other.as_biguint()) % modulus,
            );
        }
        #[cfg(target_os = "zkvm")]
        {
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ModularArithmetic as usize,
                ModArithBaseFunct7::SubMod as usize,
                self as *mut Self,
                self as *const Self,
                other as *const Self
            )
        }
    }

    #[inline(always)]
    fn mul_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(
                (self.as_biguint() * other.as_biguint()) % Self::modulus_biguint(),
            );
        }
        #[cfg(target_os = "zkvm")]
        {
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ModularArithmetic as usize,
                ModArithBaseFunct7::MulMod as usize,
                self as *mut Self,
                self as *const Self,
                other as *const Self
            )
        }
    }

    #[inline(always)]
    fn div_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let modulus = Self::modulus_biguint();
            let signed_inv = other.as_biguint().mod_inverse(modulus.clone()).unwrap();
            let inv = if signed_inv.sign() == Sign::Minus {
                modulus.to_bigint().unwrap() + signed_inv
            } else {
                signed_inv
            }
            .to_biguint()
            .unwrap();
            *self = Self::from_biguint((self.as_biguint() * inv) % modulus);
        }
        #[cfg(target_os = "zkvm")]
        {
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ModularArithmetic as usize,
                ModArithBaseFunct7::DivMod as usize,
                self as *mut Self,
                self as *const Self,
                other as *const Self
            )
        }
    }
}

impl<'a> AddAssign<&'a IntModN> for IntModN {
    #[inline(always)]
    fn add_assign(&mut self, other: &'a IntModN) {
        self.add_assign_impl(other);
    }
}

impl AddAssign for IntModN {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        self.add_assign_impl(&other);
    }
}

impl Add for IntModN {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

impl<'a> Add<&'a IntModN> for IntModN {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, other: &'a IntModN) -> Self::Output {
        self += other;
        self
    }
}

impl<'a> Add<&'a IntModN> for &IntModN {
    type Output = IntModN;
    #[inline(always)]
    fn add(self, other: &'a IntModN) -> Self::Output {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut res = self.clone();
            res += other;
            res
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<IntModN> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ModularArithmetic as usize,
                ModArithBaseFunct7::AddMod as usize,
                uninit.as_mut_ptr(),
                self as *const IntModN,
                other as *const IntModN
            );
            unsafe { uninit.assume_init() }
        }
    }
}

impl<'a> SubAssign<&'a IntModN> for IntModN {
    #[inline(always)]
    fn sub_assign(&mut self, other: &'a IntModN) {
        self.sub_assign_impl(other);
    }
}

impl SubAssign for IntModN {
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        self.sub_assign_impl(&other);
    }
}

impl Sub for IntModN {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, other: Self) -> Self::Output {
        self -= other;
        self
    }
}

impl<'a> Sub<&'a IntModN> for IntModN {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, other: &'a IntModN) -> Self::Output {
        self -= other;
        self
    }
}

impl<'a> Sub<&'a IntModN> for &IntModN {
    type Output = IntModN;
    #[inline(always)]
    fn sub(self, other: &'a IntModN) -> Self::Output {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut res = self.clone();
            res -= other;
            res
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<IntModN> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ModularArithmetic as usize,
                ModArithBaseFunct7::SubMod as usize,
                uninit.as_mut_ptr(),
                self as *const IntModN,
                other as *const IntModN
            );
            unsafe { uninit.assume_init() }
        }
    }
}

impl<'a> MulAssign<&'a IntModN> for IntModN {
    #[inline(always)]
    fn mul_assign(&mut self, other: &'a IntModN) {
        self.mul_assign_impl(other);
    }
}

impl MulAssign for IntModN {
    #[inline(always)]
    fn mul_assign(&mut self, other: Self) {
        self.mul_assign_impl(&other);
    }
}

impl Mul for IntModN {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, other: Self) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a> Mul<&'a IntModN> for IntModN {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, other: &'a IntModN) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a> Mul<&'a IntModN> for &IntModN {
    type Output = IntModN;
    #[inline(always)]
    fn mul(self, other: &'a IntModN) -> Self::Output {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut res = self.clone();
            res *= other;
            res
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<IntModN> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ModularArithmetic as usize,
                ModArithBaseFunct7::MulMod as usize,
                uninit.as_mut_ptr(),
                self as *const IntModN,
                other as *const IntModN
            );
            unsafe { uninit.assume_init() }
        }
    }
}

impl<'a> DivAssign<&'a IntModN> for IntModN {
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div_assign(&mut self, other: &'a IntModN) {
        self.div_assign_impl(other);
    }
}

impl DivAssign for IntModN {
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div_assign(&mut self, other: Self) {
        self.div_assign_impl(&other);
    }
}

impl Div for IntModN {
    type Output = Self;
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div(mut self, other: Self) -> Self::Output {
        self /= other;
        self
    }
}

impl<'a> Div<&'a IntModN> for IntModN {
    type Output = Self;
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div(mut self, other: &'a IntModN) -> Self::Output {
        self /= other;
        self
    }
}

impl<'a> Div<&'a IntModN> for &IntModN {
    type Output = IntModN;
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div(self, other: &'a IntModN) -> Self::Output {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut res = self.clone();
            res /= other;
            res
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<IntModN> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ModularArithmetic as usize,
                ModArithBaseFunct7::DivMod as usize,
                uninit.as_mut_ptr(),
                self as *const IntModN,
                other as *const IntModN
            );
            unsafe { uninit.assume_init() }
        }
    }
}

impl PartialEq for IntModN {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        #[cfg(not(target_os = "zkvm"))]
        {
            self.as_bytes() == other.as_bytes()
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut x: u32;
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ModularArithmetic as usize,
                ModArithBaseFunct7::IsEqMod as usize,
                x,
                self as *const IntModN,
                other as *const IntModN
            );
            x != 0
        }
    }
}

#[cfg(not(target_os = "zkvm"))]
mod helper {
    use super::*;
    impl Mul<u32> for IntModN {
        type Output = IntModN;
        #[inline(always)]
        fn mul(self, other: u32) -> Self::Output {
            let mut res = self.clone();
            res *= IntModN::from_u32(other);
            res
        }
    }

    impl Mul<u32> for &IntModN {
        type Output = IntModN;
        #[inline(always)]
        fn mul(self, other: u32) -> Self::Output {
            let mut res = self.clone();
            res *= IntModN::from_u32(other);
            res
        }
    }
}
