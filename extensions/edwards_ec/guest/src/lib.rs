#![no_std]
extern crate self as openvm_te_guest;
#[macro_use]
extern crate alloc;

pub use once_cell;
pub use openvm_algebra_guest as algebra;
pub use openvm_ecc_te_macros as te_macros;
use strum_macros::FromRepr;

mod affine_point;
pub use affine_point::*;
// Use the same group traits as the short Weierstrass ec extension
pub use openvm_ecc_guest::group::*;
// Use the same msm implementation as the short Weierstrass ec extension
pub use openvm_ecc_guest::msm::*;

/// Edwards curve traits
pub mod edwards;

#[cfg(feature = "ed25519")]
pub mod ed25519;

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

// Use the same traits as the short Weierstrass ec extension
pub use openvm_ecc_guest::{FromCompressed, IntrinsicCurve};
