#![no_std]
use strum_macros::FromRepr;
extern crate self as axvm_ecc_guest;

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
}

impl SwBaseFunct7 {
    pub const SHORT_WEIERSTRASS_MAX_KINDS: u8 = 8;
}

extern crate alloc;

pub use axvm_algebra_guest as algebra;
pub use axvm_ecc_sw_setup as sw_setup;
#[cfg(feature = "halo2curves")]
pub use halo2curves_axiom as halo2curves;

mod affine_point;
pub use affine_point::*;
mod group;
pub use group::*;
mod msm;
pub use msm::*;
mod ecdsa;
pub use ecdsa::*;

/// Weierstrass curve traits
pub mod sw;

/// Types for Secp256k1 curve with intrinsic functions. Implements traits necessary for ECDSA.
#[cfg(feature = "k256")]
pub mod k256;
