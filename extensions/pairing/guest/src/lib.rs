#![no_std]
use strum_macros::FromRepr;

/// This is custom-1 defined in RISC-V spec document
pub const OPCODE: u8 = 0x2b;
pub const PAIRING_FUNCT3: u8 = 0b011;

/// The funct7 field equals `pairing_idx * PAIRING_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum PairingBaseFunct7 {
    HintFinalExp = 0,
}

impl PairingBaseFunct7 {
    pub const PAIRING_MAX_KINDS: u8 = 16;
}

extern crate alloc;
extern crate self as openvm_ecc;

#[cfg(feature = "halo2curves")]
pub use halo2curves_axiom as halo2curves;
pub use openvm_algebra_guest as algebra;

/// Implementation of this library's traits on halo2curves types.
/// Used for testing and also VM runtime execution.
/// These should **only** be importable on a host machine.
#[cfg(all(feature = "halo2curves", not(target_os = "zkvm")))]
pub mod halo2curves_shims;

/// Traits for optimal Ate pairing check using intrinsic functions.
pub mod pairing;

/// Types for BLS12-381 curve with intrinsic functions.
#[cfg(feature = "bls12_381")]
pub mod bls12_381;
/// Types for BN254 curve with intrinsic functions.
#[cfg(feature = "bn254")]
pub mod bn254;
