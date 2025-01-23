#![no_std]
extern crate self as openvm_ecc_guest;
#[macro_use]
extern crate alloc;

#[cfg(feature = "halo2curves")]
pub use halo2curves_axiom as halo2curves;
pub use once_cell;
pub use openvm_algebra_guest as algebra;
pub use openvm_ecc_sw_macros as sw_macros;
pub use openvm_ecc_te_macros as te_macros;
use strum_macros::FromRepr;

mod affine_point;
pub use affine_point::*;
mod group;
pub use group::*;
mod msm;
pub use msm::*;

/// ECDSA
pub mod ecdsa;
/// Edwards curve traits
pub mod edwards;
/// Weierstrass curve traits
pub mod weierstrass;

/// Types for Secp256k1 curve with intrinsic functions. Implements traits necessary for ECDSA.
#[cfg(feature = "k256")]
pub mod k256;

/// a.k.a. Secp256r1
#[cfg(feature = "p256")]
pub mod p256;

#[cfg(all(test, feature = "k256", feature = "p256", not(target_os = "zkvm")))]
mod tests;

/// This is custom-1 defined in RISC-V spec document
pub const SW_OPCODE: u8 = 0x2b;
pub const SW_FUNCT3: u8 = 0b001;

/// Short Weierstrass curves are configurable.
/// The funct7 field equals `curve_idx * SHORT_WEIERSTRASS_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum SwBaseFunct7 {
    SwAddNe = 0,
    SwDouble,
    SwSetup,
    HintDecompress,
    HintNonQr,
}

impl SwBaseFunct7 {
    pub const SHORT_WEIERSTRASS_MAX_KINDS: u8 = 8;
}

/// This is custom-1 defined in RISC-V spec document
pub const TE_OPCODE: u8 = 0x2b;
pub const TE_FUNCT3: u8 = 0b100;

#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum TeBaseFunct7 {
    TeAdd = 0,
    TeSetup,
}

impl TeBaseFunct7 {
    pub const TWISTED_EDWARDS_MAX_KINDS: u8 = 8;
}
