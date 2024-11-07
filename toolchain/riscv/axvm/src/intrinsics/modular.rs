use core::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::BigUint;

/// Trait definition for axVM modular integers, where each operation
/// is done modulo MODULUS.
pub trait IntMod:
    Sized
    + Eq
    + Clone
    + Debug
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Sum
    + Product
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    + for<'a> Sum<&'a Self>
    + for<'a> Product<&'a Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> DivAssign<&'a Self>
{
    /// Underlying representation of IntMod.
    type Repr;

    /// Index of IntMod::MODULUS.
    const MOD_IDX: usize;

    /// Modulus as an array of bytes.
    const MODULUS: Self::Repr;

    /// The zero element (i.e. the additive identity).
    const ZERO: Self;

    /// The one element (i.e. the multiplicative identity).
    const ONE: Self;

    /// The negative one element (i.e. additive inverse of one).
    const NEG_ONE: Self;

    /// Creates a new IntMod an instance of Repr.
    fn from_repr(repr: Self::Repr) -> Self;

    /// Creates a new IntMod from an array of bytes.
    fn from_le_bytes(bytes: &[u8]) -> Self;

    /// Creates a new IntMod from a u32.
    fn from_u8(val: u8) -> Self;

    /// Creates a new IntMod from a u32.
    fn from_u32(val: u32) -> Self;

    /// Creates a new IntMod from a u32.
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
