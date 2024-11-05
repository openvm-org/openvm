use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
#[cfg(target_os = "zkvm")]
use core::{borrow::BorrowMut, mem::MaybeUninit};

use hex_literal::hex;
#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::{traits::ModInverse, BigUint, Sign, ToBigInt};

#[cfg(not(target_os = "zkvm"))]
use crate::intrinsics::biguint_to_limbs;

const LIMBS: usize = 32;

/// Class to represent an integer modulo N, which is currently hard-coded to be the
/// secp256k1 prime.
#[derive(Clone)]
#[repr(C, align(32))]
pub struct IntModN([u8; LIMBS]);

impl IntModN {
    const MODULUS: [u8; LIMBS] =
        hex!("FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F");
    const _MOD_IDX: usize = 0;

    /// Creates a new IntModN from an array of bytes.
    pub fn from_bytes(bytes: [u8; LIMBS]) -> Self {
        Self(bytes)
    }

    /// Value of this IntModN as an array of bytes.
    pub fn as_bytes(&self) -> &[u8; LIMBS] {
        &(self.0)
    }

    /// Creates a new IntModN from a BigUint.
    #[cfg(not(target_os = "zkvm"))]
    pub fn from_biguint(biguint: BigUint) -> Self {
        Self(biguint_to_limbs(biguint))
    }

    /// Value of this IntModN as a BigUint.
    #[cfg(not(target_os = "zkvm"))]
    pub fn as_biguint(&self) -> BigUint {
        BigUint::from_bytes_le(self.as_bytes())
    }

    /// Modulus N as a BigUint.
    #[cfg(not(target_os = "zkvm"))]
    pub fn modulus_biguint() -> BigUint {
        BigUint::from_bytes_be(&Self::MODULUS)
    }

    #[inline(always)]
    fn add_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            self.0 = biguint_to_limbs(
                (self.as_biguint() + other.as_biguint()) % Self::modulus_biguint(),
            );
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }

    #[inline(always)]
    fn sub_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let modulus = Self::modulus_biguint();
            self.0 = biguint_to_limbs(
                (self.as_biguint() + modulus.clone() - other.as_biguint()) % modulus,
            );
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }

    #[inline(always)]
    fn mul_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            self.0 = biguint_to_limbs(
                (self.as_biguint() * other.as_biguint()) % Self::modulus_biguint(),
            );
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
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
            self.0 = biguint_to_limbs((self.as_biguint() * inv) % modulus);
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
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
            let ptr: *mut IntModN = uninit.as_mut_ptr();
            unsafe {
                *ptr = todo!();
                uninit.assume_init()
            }
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
            todo!()
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
            todo!()
        }
    }
}

impl<'a> DivAssign<&'a IntModN> for IntModN {
    #[inline(always)]
    fn div_assign(&mut self, other: &'a IntModN) {
        self.div_assign_impl(other);
    }
}

impl DivAssign for IntModN {
    #[inline(always)]
    fn div_assign(&mut self, other: Self) {
        self.div_assign_impl(&other);
    }
}

impl Div for IntModN {
    type Output = Self;
    #[inline(always)]
    fn div(mut self, other: Self) -> Self::Output {
        self /= other;
        self
    }
}

impl<'a> Div<&'a IntModN> for IntModN {
    type Output = Self;
    #[inline(always)]
    fn div(mut self, other: &'a IntModN) -> Self::Output {
        self /= other;
        self
    }
}

impl<'a> Div<&'a IntModN> for &IntModN {
    type Output = IntModN;
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
            todo!()
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
            todo!()
        }
    }
}
