use core::ops::{Add, Div, Mul, Sub};

use hex_literal::hex;
use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, ToPrimitive};

const LIMBS: usize = 32;

#[repr(transparent)]
struct ModularInt([u8; LIMBS]);

/// Class to represent an integer modulo N, which is currently hard-coded to be
/// the secp256k1 prime. Note that rust cannot support multiple repr attributes
/// to a single struct when one is transparent, and thus IntModN wraps helper
/// struct ModularInt.
#[repr(align(32))]
pub struct IntModN(ModularInt);

impl IntModN {
    const MODULUS: [u8; LIMBS] =
        hex!("FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F");
    const _MOD_IDX: usize = 0;

    /// Creates a new IntModN from an array of bytes.
    pub fn from_bytes(bytes: [u8; LIMBS]) -> Self {
        Self(ModularInt(bytes))
    }

    /// Creates a new IntModN from a BigUint.
    pub fn from_biguint(biguint: BigUint) -> Self {
        Self(ModularInt(Self::biguint_to_limbs(biguint)))
    }

    /// Value of this IntModN as an array of bytes.
    pub fn as_bytes(&self) -> &[u8; LIMBS] {
        &(self.0 .0)
    }

    /// Value of this IntModN as a BigUint.
    pub fn as_biguint(&self) -> BigUint {
        BigUint::from_bytes_le(self.as_bytes())
    }

    /// Modulus N as a BigUint.
    pub fn modulus_biguint() -> BigUint {
        BigUint::from_bytes_be(&Self::MODULUS)
    }

    fn biguint_to_limbs(mut x: BigUint) -> [u8; LIMBS] {
        let mut result = [0; LIMBS];
        let base = BigUint::from_u32(1 << 8).unwrap();
        for r in result.iter_mut() {
            *r = (x.clone() % &base).to_u8().unwrap();
            x /= &base;
        }
        result
    }
}

impl Add for &IntModN {
    type Output = IntModN;
    fn add(self, other: Self) -> Self::Output {
        IntModN::from_biguint((self.as_biguint() + other.as_biguint()) % IntModN::modulus_biguint())
    }
}

impl Add for IntModN {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        &self + &other
    }
}

impl Sub for &IntModN {
    type Output = IntModN;
    fn sub(self, other: Self) -> Self::Output {
        IntModN::from_biguint((self.as_biguint() - other.as_biguint()) % IntModN::modulus_biguint())
    }
}

impl Sub for IntModN {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        &self - &other
    }
}

impl Mul for &IntModN {
    type Output = IntModN;
    fn mul(self, other: Self) -> Self::Output {
        IntModN::from_biguint((self.as_biguint() * other.as_biguint()) % IntModN::modulus_biguint())
    }
}

impl Mul for IntModN {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        &self * &other
    }
}

impl Div for &IntModN {
    type Output = IntModN;
    fn div(self, other: Self) -> Self::Output {
        IntModN::from_biguint((self.as_biguint() / other.as_biguint()) % IntModN::modulus_biguint())
    }
}

impl Div for IntModN {
    type Output = Self;
    fn div(self, other: Self) -> Self::Output {
        &self / &other
    }
}
