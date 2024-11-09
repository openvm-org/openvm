use core::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use axvm::intrinsics::IntMod;
#[cfg(target_os = "zkvm")]
use {
    axvm_platform::constants::{Custom1Funct3, ModArithBaseFunct7, SwBaseFunct7, CUSTOM_1},
    axvm_platform::custom_insn_r,
    core::mem::MaybeUninit,
};

axvm::moduli_setup! {
    IntModN = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F";
}

pub trait Group:
    Clone
    + Debug
    + Eq
    + Sized
    + Add<Output = Self>
    + Sub<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + AddAssign
    + SubAssign
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + Mul<Self::Scalar, Output = Self>
    + MulAssign<Self::Scalar>
    + for<'a> Mul<&'a Self::Scalar, Output = Self>
    + for<'a> MulAssign<&'a Self::Scalar>
{
    type Scalar: IntMod;
    type SelfRef<'a>: Add<&'a Self, Output = Self>
        + Sub<&'a Self, Output = Self>
        + Mul<&'a Self::Scalar, Output = Self>
    where
        Self: 'a;

    fn identity() -> Self;
    fn is_identity(&self) -> bool;
    fn generator() -> Self;

    fn double(&self) -> Self;
}

axvm::ec_setup! {
    EcPointN = IntModN;
}
