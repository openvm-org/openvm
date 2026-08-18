#![no_std]
extern crate self as openvm_ecc_guest;
#[macro_use]
extern crate alloc;

pub use once_cell;
pub use openvm_algebra_guest as algebra;
pub use openvm_ecc_sw_macros as sw_macros;
use strum_macros::FromRepr;

mod affine_point;
pub use affine_point::*;
mod group;
pub use group::*;
mod msm;
pub use msm::*;

/// Optimized ECDSA implementation with the same functional interface as the `ecdsa` crate
pub mod ecdsa;
/// Weierstrass curve traits
pub mod weierstrass;

/// This is custom-1 defined in RISC-V spec document
pub const OPCODE: u8 = 0x2b;
pub const SW_FUNCT3: u8 = 0b001;

/// Short Weierstrass curves are configurable.
/// The funct7 field equals `curve_idx * SHORT_WEIERSTRASS_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum SwBaseFunct7 {
    /// Adds two affine points.
    ///
    /// Requirements:
    /// - Both points are not the identity.
    /// - The x-coordinates are different.
    SwAddNe = 0,
    /// Doubles an affine point.
    ///
    /// Requirements:
    /// - The point is not the identity.
    /// - The result is not the identity.
    SwDouble,
    SwSetup,
    /// Multiplies an affine point by a scalar.
    ///
    /// Requirements:
    /// - The point is not the identity.
    /// - The point is in the configured prime-order subgroup.
    /// - The scalar is odd and less than the subgroup order.
    /// - The subgroup order is equal to 1 modulo 4.
    SwMul,
    SwSetupMul,
}

impl SwBaseFunct7 {
    pub const SHORT_WEIERSTRASS_MAX_KINDS: u8 = 8;
}
