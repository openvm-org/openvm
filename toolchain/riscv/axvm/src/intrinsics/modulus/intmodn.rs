use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
#[cfg(target_os = "zkvm")]
use core::{borrow::BorrowMut, mem::MaybeUninit};

use hex_literal::hex;
#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::BigUint;
#[cfg(not(target_os = "zkvm"))]
use num_traits::{FromPrimitive, ToPrimitive};

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
        Self(Self::biguint_to_limbs(biguint))
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

    #[cfg(not(target_os = "zkvm"))]
    fn biguint_to_limbs(mut x: BigUint) -> [u8; LIMBS] {
        let mut result = [0; LIMBS];
        let base = BigUint::from_u32(1 << 8).unwrap();
        for r in result.iter_mut() {
            *r = (x.clone() % &base).to_u8().unwrap();
            x /= &base;
        }
        result
    }

    #[inline]
    fn add_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            self.0 = Self::biguint_to_limbs(
                (other.as_biguint() + self.as_biguint()) % Self::modulus_biguint(),
            );
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }

    #[inline]
    fn sub_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            self.0 = Self::biguint_to_limbs(
                (other.as_biguint() - self.as_biguint()) % Self::modulus_biguint(),
            );
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }

    #[inline]
    fn mul_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            self.0 = Self::biguint_to_limbs(
                (other.as_biguint() * self.as_biguint()) % Self::modulus_biguint(),
            );
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }

    #[inline]
    fn div_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            self.0 = Self::biguint_to_limbs(
                (other.as_biguint() / self.as_biguint()) % Self::modulus_biguint(),
            );
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }
}

impl<'a> AddAssign<&'a IntModN> for IntModN {
    #[inline]
    fn add_assign(&mut self, other: &'a IntModN) {
        self.add_assign_impl(other);
    }
}

impl AddAssign for IntModN {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.add_assign_impl(&other);
    }
}

impl Add for IntModN {
    type Output = Self;
    #[inline]
    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

impl<'a> Add<&'a IntModN> for IntModN {
    type Output = Self;
    #[inline]
    fn add(mut self, other: &'a IntModN) -> Self::Output {
        self += other;
        self
    }
}

impl<'a> Add<&'a IntModN> for &IntModN {
    type Output = IntModN;
    #[inline]
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
    #[inline]
    fn sub_assign(&mut self, other: &'a IntModN) {
        self.sub_assign_impl(other);
    }
}

impl SubAssign for IntModN {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.sub_assign_impl(&other);
    }
}

impl Sub for IntModN {
    type Output = Self;
    #[inline]
    fn sub(mut self, other: Self) -> Self::Output {
        self -= other;
        self
    }
}

impl<'a> Sub<&'a IntModN> for IntModN {
    type Output = Self;
    #[inline]
    fn sub(mut self, other: &'a IntModN) -> Self::Output {
        self -= other;
        self
    }
}

impl<'a> Sub<&'a IntModN> for &IntModN {
    type Output = IntModN;
    #[inline]
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
    #[inline]
    fn mul_assign(&mut self, other: &'a IntModN) {
        self.mul_assign_impl(other);
    }
}

impl MulAssign for IntModN {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.mul_assign_impl(&other);
    }
}

impl Mul for IntModN {
    type Output = Self;
    #[inline]
    fn mul(mut self, other: Self) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a> Mul<&'a IntModN> for IntModN {
    type Output = Self;
    #[inline]
    fn mul(mut self, other: &'a IntModN) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a> Mul<&'a IntModN> for &IntModN {
    type Output = IntModN;
    #[inline]
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
    #[inline]
    fn div_assign(&mut self, other: &'a IntModN) {
        self.div_assign_impl(other);
    }
}

impl DivAssign for IntModN {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.div_assign_impl(&other);
    }
}

impl Div for IntModN {
    type Output = Self;
    #[inline]
    fn div(mut self, other: Self) -> Self::Output {
        self /= other;
        self
    }
}

impl<'a> Div<&'a IntModN> for IntModN {
    type Output = Self;
    #[inline]
    fn div(mut self, other: &'a IntModN) -> Self::Output {
        self /= other;
        self
    }
}

impl<'a> Div<&'a IntModN> for &IntModN {
    type Output = IntModN;
    #[inline]
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
    #[inline]
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
