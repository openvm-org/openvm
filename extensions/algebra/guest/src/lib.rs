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
    HintNonQr,
    HintSqrt,
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
#[cfg(target_os = "zkvm")]
pub use openvm_custom_insn;
#[cfg(target_os = "zkvm")]
pub use openvm_rv32im_guest;
pub use serde_big_array::BigArray;
use strum_macros::FromRepr;

/// Implementation of this library's traits on halo2curves types.
/// Used for testing and also VM runtime execution.
/// These should **only** be importable on a host machine.
#[cfg(all(not(target_os = "zkvm"), feature = "halo2curves"))]
mod halo2curves;

/// Exponentiation by bytes
mod exp_bytes;
/// Field traits
pub mod field;
pub use exp_bytes::*;
pub use once_cell;

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
    /// `SelfRef<'a>` should almost always be `&'a Self`. This is a way to include implementations
    /// of binary operations where both sides are `&'a Self`.
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
    /// Does not enforce the integer value of `bytes` must be less than the modulus.
    fn from_repr(repr: Self::Repr) -> Self;

    /// Creates a new IntMod from an array of bytes, little endian.
    /// Returns `None` if the integer value of `bytes` is greater than or equal to the modulus.
    fn from_le_bytes(bytes: &[u8]) -> Option<Self>;

    /// Creates a new IntMod from an array of bytes, big endian.
    /// Returns `None` if the integer value of `bytes` is greater than or equal to the modulus.
    fn from_be_bytes(bytes: &[u8]) -> Option<Self>;

    /// Creates a new IntMod from an array of bytes, little endian.
    /// Does not enforce the integer value of `bytes` must be less than the modulus.
    fn from_le_bytes_unchecked(bytes: &[u8]) -> Self;

    /// Creates a new IntMod from an array of bytes, big endian.
    /// Does not enforce the integer value of `bytes` must be less than the modulus.
    fn from_be_bytes_unchecked(bytes: &[u8]) -> Self;

    /// Creates a new IntMod from a u8.
    /// Does not enforce the integer value of `bytes` must be less than the modulus.
    fn from_u8(val: u8) -> Self;

    /// Creates a new IntMod from a u32.
    /// Does not enforce the integer value of `bytes` must be less than the modulus.
    fn from_u32(val: u32) -> Self;

    /// Creates a new IntMod from a u64.
    /// Does not enforce the integer value of `bytes` must be less than the modulus.
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

    /// VM specific concept: during guest execution, it is not enforced that the representation
    /// of `Self` must be the unique integer less than the modulus. The guest code may sometimes
    /// want to enforce that the representation is the canonical one less than the modulus.
    /// the host to an honest host to provide the canonical representation less than the modulus.
    ///
    /// This function should enforce that guest execution proceeds **if and only if** `self`
    /// is in the unique representation less than the modulus.
    fn assert_reduced(&self);

    /// Is the integer representation of `self` less than the modulus?
    fn is_reduced(&self) -> bool;

    /// Calls any setup required for this modulus. The implementation should internally use
    /// `OnceBool` to ensure that setup is only called once.
    fn set_up_once();

    /// Returns whether the two integers are congrument modulo the modulus.
    ///
    /// # Safety
    /// - If `CHECK_SETUP` is true, checks if setup has been called for this curve and if not, calls
    ///   `Self::set_up_once()`. Only set `CHECK_SETUP` to `false` if you are sure that setup has
    ///   been called already.
    unsafe fn eq_impl<const CHECK_SETUP: bool>(&self, other: &Self) -> bool;

    /// Add two elements.
    ///
    /// # Safety
    /// - If `CHECK_SETUP` is true, checks if setup has been called for this curve and if not, calls
    ///   `Self::set_up_once()`. Only set `CHECK_SETUP` to `false` if you are sure that setup has
    ///   been called already.
    unsafe fn add_ref<const CHECK_SETUP: bool>(&self, other: &Self) -> Self;
}

// Ref: https://docs.rs/elliptic-curve/latest/elliptic_curve/ops/trait.Reduce.html
pub trait Reduce: Sized {
    /// Interpret the given bytes as an integer and perform a modular reduction.
    fn reduce_le_bytes(bytes: &[u8]) -> Self;
    fn reduce_be_bytes(bytes: &[u8]) -> Self {
        Self::reduce_le_bytes(&bytes.iter().rev().copied().collect::<Vec<_>>())
    }
}

// Note that we use a hint-based approach to prove whether the square root exists.
// This approach works for prime moduli, but not necessarily for composite moduli,
// which is why the Sqrt trait requires the Field trait, not just the IntMod trait.
pub trait Sqrt: Field {
    /// Returns a square root of `self` if it exists.
    fn sqrt(&self) -> Option<Self>;
}
