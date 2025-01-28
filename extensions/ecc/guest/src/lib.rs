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

#[cfg(feature = "ed25519")]
pub mod ed25519;

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
    SwHintDecompress,
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
    TeHintDecompress,
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

pub trait FromCompressed<Coordinate> {
    /// Given `x`-coordinate,
    ///
    /// ## Panics
    /// If the input is not a valid compressed point.
    /// The zkVM panics instead of returning an [Option] because this function
    /// can only guarantee correct behavior when decompression is possible,
    /// but the function cannot compute the boolean equal to true if and only
    /// if decompression is possible.
    // This is because we rely on a hint for the correct decompressed value
    // and then constrain its correctness. A malicious prover could hint
    // incorrectly, so there is no way to use a hint to prove that the input
    // **cannot** be decompressed.
    fn decompress(x: Coordinate, rec_id: &u8) -> Self;

    /// If it exists, hints the unique `y` coordinate that is less than `Coordinate::MODULUS`
    /// such that `(x, y)` is a point on the curve and `y` has parity equal to `rec_id`.
    /// If such `y` does not exist, undefined behavior.
    ///
    /// This is only a hint, and the returned `y` does not guarantee any of the above properties.
    /// They must be checked separately. Normal users should use `decompress` directly.
    fn hint_decompress(x: &Coordinate, rec_id: &u8) -> Coordinate;
}
