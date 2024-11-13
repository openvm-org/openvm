use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use axvm::intrinsics::{DivUnsafe, IntMod};
use elliptic_curve::{CurveArithmetic, FieldBytes, Scalar};
#[cfg(target_os = "zkvm")]
use {
    axvm_platform::constants::{Custom1Funct3, ModArithBaseFunct7, SwBaseFunct7, CUSTOM_1},
    axvm_platform::custom_insn_r,
    core::mem::MaybeUninit,
};

use super::group::Group;

// Secp256k1 modulus
// TODO[jpw] rename to Secp256k1Coord
axvm::moduli_setup! {
    IntModN = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F";
}

// TODO[jpw] rename to Secp256k1
axvm::sw_setup! {
    EcPointN = (
        IntModN,
        "0x79be667e f9dcbbac 55a06295 ce870b07 029bfcdb 2dce28d9 59f2815b 16f81798",
        "0x483ada77 26a3c465 5da4fbfc 0e1108a8 fd17b448 a6855419 9c47d08f fb10d4b8",
    );
}
