#![no_std]
extern crate self as openvm_algebra_guest;

/// This is custom-1 defined in RISC-V spec document
pub const OPCODE: u8 = 0x2b;
pub const MODULAR_ARITHMETIC_FUNCT3: u8 = 0b000;
pub const COMPLEX_EXT_FIELD_FUNCT3: u8 = 0b010;

/// Modular arithmetic is configurable.
/// The funct7 field equals `mod_idx * MODULAR_ARITHMETIC_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum ModArithBaseFunct7 {
    AddMod = 0,
    SubMod,
    MulMod,
    DivMod,
    IsEqMod,
    SetupMod,
}

impl ModArithBaseFunct7 {
    pub const MODULAR_ARITHMETIC_MAX_KINDS: u8 = 8;
}

/// Complex extension field is configurable.
/// The funct7 field equals `fp2_idx * COMPLEX_EXT_FIELD_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum ComplexExtFieldBaseFunct7 {
    Add = 0,
    Sub,
    Mul,
    Div,
    Setup,
}

impl ComplexExtFieldBaseFunct7 {
    pub const COMPLEX_EXT_FIELD_MAX_KINDS: u8 = 8;
}

/// Modular arithmetic traits for use with OpenVM intrinsics.
extern crate alloc;

use alloc::vec::Vec;
use core::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub use field::Field;
#[cfg(not(target_os = "zkvm"))]
use num_bigint::BigUint;
pub use openvm_algebra_complex_macros as complex_macros;
pub use openvm_algebra_moduli_macros as moduli_macros;
pub use serde_big_array::BigArray;
use strum_macros::FromRepr;

/// Field traits
pub mod field;
/// Implementation of this library's traits on halo2curves types.
/// Used for testing and also VM runtime execution.
/// These should **only** be importable on a host machine.
#[cfg(all(not(target_os = "zkvm"), feature = "halo2curves"))]
mod halo2curves;

/// Exponentiation by bytes
mod exp_bytes;
pub use exp_bytes::*;

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

/// Trait definition for OpenVM modular integers, where each operation
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
    /// Underlying representation of IntMod. Usually of the form `[u8; NUM_LIMBS]`.
    type Repr: AsRef<[u8]> + AsMut<[u8]>;
    /// `SelfRef<'a>` should almost always be `&'a Self`. This is a way to include implementations of binary operations where both sides are `&'a Self`.
    type SelfRef<'a>: Add<&'a Self, Output = Self>
        + Sub<&'a Self, Output = Self>
        + Neg<Output = Self>
        + Mul<&'a Self, Output = Self>
        + DivUnsafe<&'a Self, Output = Self>
    where
        Self: 'a;

    /// Modulus as a Repr.
    const MODULUS: Self::Repr;

    /// Number of limbs used to internally represent an element of `Self`.
    const NUM_LIMBS: usize;

    /// The zero element (i.e. the additive identity).
    const ZERO: Self;

    /// The one element (i.e. the multiplicative identity).
    const ONE: Self;

    /// Creates a new IntMod from an instance of Repr.
    fn from_repr(repr: Self::Repr) -> Self;

    /// Creates a new IntMod from an array of bytes, little endian.
    fn from_le_bytes(bytes: &[u8]) -> Self;

    /// Creates a new IntMod from an array of bytes, big endian.
    fn from_be_bytes(bytes: &[u8]) -> Self;

    /// Creates a new IntMod from a u8.
    fn from_u8(val: u8) -> Self;

    /// Creates a new IntMod from a u32.
    fn from_u32(val: u32) -> Self;

    /// Creates a new IntMod from a u64.
    fn from_u64(val: u64) -> Self;

    /// Value of this IntMod as an array of bytes, little endian.
    fn as_le_bytes(&self) -> &[u8];

    /// Value of this IntMod as an array of bytes, big endian.
    fn to_be_bytes(&self) -> Self::Repr;

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

    /// zkVM specific concept: the in-memory values of `Self` will normally
    /// be in their canonical unique form (e.g., less than modulus) but the
    /// zkVM circuit does not constrain it. In cases where uniqueness is
    /// essential for security, this function should be called to constrain
    /// uniqueness.
    ///
    /// Note that this is done automatically in [PartialEq] and [Eq] implementations.
    ///
    /// ## Panics
    /// If assertion fails.
    fn assert_unique(&self) {
        // This must not be optimized out
        let _ = core::hint::black_box(PartialEq::eq(self, self));
    }

    /// This function is mostly for internal use in other internal implementations.
    /// Normal users are not advised to use it.
    ///
    /// If `self` was directly constructed from a raw representation
    /// and not in its canonical unique form (e.g., less than the modulus),
    /// this function will "reduce" `self` to its canonical form and also
    /// call `assert_unique`.
    fn reduce(&mut self) {
        self.add_assign(&Self::ZERO);
        self.assert_unique();
    }
}

// Ref: https://docs.rs/elliptic-curve/latest/elliptic_curve/ops/trait.Reduce.html
pub trait Reduce: Sized {
    /// Interpret the given bytes as an integer and perform a modular reduction.
    fn reduce_le_bytes(bytes: &[u8]) -> Self;
    fn reduce_be_bytes(bytes: &[u8]) -> Self {
        Self::reduce_le_bytes(&bytes.iter().rev().copied().collect::<Vec<_>>())
    }
}
