//! Modular arithmetic traits for use with axVM intrinsics.
use core::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::BigUint;

use crate::{DivAssignUnsafe, DivUnsafe};

/// Trait definition for axVM modular integers, where each operation
/// is done modulo MODULUS.
///
/// Division is only defined over the group of units in the ring of integers modulo MODULUS.
/// It is undefined behavior outside of this group.
pub trait IntMod:
    Sized
    + Eq
    + Clone
    + Debug
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + DivUnsafe<Output = Self>
    + Sum
    + Product
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> DivUnsafe<&'a Self, Output = Self>
    + for<'a> Sum<&'a Self>
    + for<'a> Product<&'a Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssignUnsafe
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> DivAssignUnsafe<&'a Self>
{
    /// Underlying representation of IntMod.
    type Repr;
    /// `SelfRef<'a>` should almost always be `&'a Self`. This is a way to include implementations of binary operations where both sides are `&'a Self`.
    type SelfRef<'a>: Add<&'a Self, Output = Self>
        + Sub<&'a Self, Output = Self>
        + Mul<&'a Self, Output = Self>
        + DivUnsafe<&'a Self, Output = Self>
    where
        Self: 'a;

    /// Index of IntMod::MODULUS.
    const MOD_IDX: usize;

    /// Modulus as a Repr.
    const MODULUS: Self::Repr;

    /// Number of bytes in the modulus.
    const NUM_BYTES: usize;

    /// The zero element (i.e. the additive identity).
    const ZERO: Self;

    /// The one element (i.e. the multiplicative identity).
    const ONE: Self;

    /// Creates a new IntMod from an instance of Repr.
    fn from_repr(repr: Self::Repr) -> Self;

    /// Creates a new IntMod from an array of bytes.
    fn from_le_bytes(bytes: &[u8]) -> Self;

    /// Creates a new IntMod from a u8.
    fn from_u8(val: u8) -> Self;

    /// Creates a new IntMod from a u32.
    fn from_u32(val: u32) -> Self;

    /// Creates a new IntMod from a u64.
    fn from_u64(val: u64) -> Self;

    /// Value of this IntMod as an array of bytes.
    fn as_le_bytes(&self) -> &[u8];

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
}
