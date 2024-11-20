#![no_std]

//! Modular arithmetic traits for use with axVM intrinsics.
use core::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub use field::Field;
#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::BigUint;

/// Field traits
pub mod field;
/// Implementation of this library's traits on halo2curves types.
/// Used for testing and also VM runtime execution.
/// These should **only** be importable on a host machine.
#[cfg(all(not(target_os = "zkvm"), feature = "halo2curves"))]
mod halo2curves;

/// Division operation that is undefined behavior when the denominator is not invertible.
pub trait DivUnsafe<Rhs = Self>: Sized {
    /// Output type of `div_unsafe`.
    type Output;

    /// Undefined behavior when denominator is not invertible.
    fn div_unsafe(self, other: Rhs) -> Self::Output;
}

/// Division assignment operation that is undefined behavior when the denominator is not invertible.
pub trait DivAssignUnsafe<Rhs = Self>: Sized {
    /// Undefined behavior when denominator is not invertible.
    fn div_assign_unsafe(&mut self, other: Rhs);
}

// TODO[jpw]: split this into CustomIntrinsic (for MOD_IDX) + IntegralDomain
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
        + Neg<Output = Self>
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

    fn neg_assign(&mut self);

    /// Doubles `self` in-place.
    fn double_assign(&mut self);

    /// Doubles this IntMod.
    fn double(&self) -> Self {
        let mut ret = self.clone();
        ret += self;
        ret
    }

    /// Squares `self` in-place.
    fn square_assign(&mut self);

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
