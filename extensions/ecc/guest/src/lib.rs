#![no_std]
extern crate self as openvm_ecc_guest;
#[macro_use]
extern crate alloc;

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

/// Optimized ECDSA implementation with the same functional interface as the `ecdsa` crate
pub mod ecdsa;
/// Edwards curve traits
pub mod edwards;
/// Weierstrass curve traits
pub mod weierstrass;

#[cfg(feature = "ed25519")]
pub mod ed25519;

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

/// A trait for elliptic curves that bridges the openvm types and external types with CurveArithmetic etc.
/// Implement this for external curves with corresponding openvm point and scalar types.
pub trait IntrinsicCurve {
    type Scalar: Clone;
    type Point: Clone;

    /// Multi-scalar multiplication.
    /// The implementation may be specialized to use properties of the curve
    /// (e.g., if the curve order is prime).
    fn msm(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point;
}
