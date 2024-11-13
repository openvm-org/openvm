use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use axvm::intrinsics::{DivUnsafe, IntMod};
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
    EcPointN = IntModN;
}
