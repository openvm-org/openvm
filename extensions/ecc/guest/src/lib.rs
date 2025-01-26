#![no_std]
extern crate self as openvm_ecc_guest;
#[macro_use]
extern crate alloc;

#[cfg(feature = "halo2curves")]
pub use halo2curves_axiom as halo2curves;
pub use openvm_algebra_guest as algebra;
pub use openvm_ecc_sw_macros as sw_macros;
use strum_macros::FromRepr;

mod affine_point;
pub use affine_point::*;
mod group;
pub use group::*;
mod msm;
pub use msm::*;

/// ECDSA
pub mod ecdsa;
/// Weierstrass curve traits
pub mod weierstrass;

/// Types for Secp256k1 curve with intrinsic functions. Implements traits necessary for ECDSA.
#[cfg(feature = "k256")]
pub mod k256;

/// a.k.a. Secp256r1
#[cfg(feature = "p256")]
pub mod p256;

/// This is custom-1 defined in RISC-V spec document
pub const OPCODE: u8 = 0x2b;
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
}

impl SwBaseFunct7 {
    pub const SHORT_WEIERSTRASS_MAX_KINDS: u8 = 8;
}
