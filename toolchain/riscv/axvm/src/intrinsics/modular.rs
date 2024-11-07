use core::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::BigUint;

/// Trait definition for AXVM modular integers, where each operation
/// is done modulo MODULUS.
pub trait IntMod<const LIMBS: usize>:
    Sized
    + Eq
    + Clone
    + Debug
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Sum
    + Product
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Sum<&'a Self>
    + for<'a> Product<&'a Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
{
    /// Index of IntMod::MODULUS.
    const MOD_IDX: usize;

    /// Modulus as an array of bytes.
    const MODULUS: [u8; LIMBS];

    /// The zero element (i.e. the additive identity).
    const ZERO: Self;

    /// The one element (i.e. the multiplicative identity).
    const ONE: Self;

    /// Returns MODULUS as an array of bytes.
    fn modulus() -> [u8; LIMBS] {
        Self::MODULUS
    }

    /// Creates a new IntMod from an array of bytes.
    fn from_bytes(bytes: [u8; LIMBS]) -> Self;

    /// Creates a new IntMod from a u32.
    fn from_u8(val: u8) -> Self;

    /// Creates a new IntMod from a u32.
    fn from_u32(val: u32) -> Self;

    /// Creates a new IntMod from a u32.
    fn from_u64(val: u64) -> Self;

    /// Value of this IntMod as an array of bytes.
    fn as_bytes(&self) -> &[u8; LIMBS];

    /// Modulus N as a BigUint.
    #[cfg(not(target_os = "zkvm"))]
    fn modulus_biguint() -> BigUint;

    /// Creates a new IntMod from a BigUint.
    #[cfg(not(target_os = "zkvm"))]
    fn from_biguint(biguint: BigUint) -> Self;

    /// Value of this IntMod as a BigUint.
    #[cfg(not(target_os = "zkvm"))]
    fn as_biguint(&self) -> BigUint;

    /// Doubles this IntMod.
    fn double(&self) -> Self {
        let mut ret = self.clone();
        ret += self;
        ret
    }

    /// Squares this IntMod.
    fn square(&self) -> Self {
        let mut ret = self.clone();
        ret *= self;
        ret
    }

    /// Cubes this IntMod.
    fn cube(&self) -> Self {
        let mut ret = self.square();
        ret *= self;
        ret
    }

    /// Exponentiates this IntMod by 'exp'.
    fn pow(&self, exp: u32) -> Self {
        let mut ret = Self::ONE;
        for _ in 0..exp {
            ret *= self;
        }
        ret
    }
}
