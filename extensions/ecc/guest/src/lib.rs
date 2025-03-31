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
    SwHintNonQr,
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
    TeHintNonQr,
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

// Hint for a decompression
// For short Weierstrass curves,
// - if possible is true, then `sqrt` is the decompressed y-coordinate
// - if possible is false, then `sqrt` is such that `sqrt^2 = (x^3 + a * x + b) * non_qr`
// For twisted Edwards curves,
// - if possible is true, then `sqrt` is the decompressed x-coordinate
// - if possible is false, then `sqrt` is such that `(d * y^2 - a) * x^2 = (y^2 - 1) * non_qr`
pub struct DecompressionHint<T> {
    pub possible: bool,
    pub sqrt: T,
}

pub trait FromCompressed<Coordinate> {
    /// For short Weierstrass curves, first parameter is the `x`-coordinate, for twisted Edwards curves,
    /// it is the `y`-coordinate.
    ///
    /// Decompresses a point from its `x_or_y`-coordinate and a recovery identifier which indicates
    /// the parity of the other coordinate. Given the `x_or_y`-coordinate, this function attempts to find the
    /// corresponding `other_coordinate` that satisfies the elliptic curve equation. If successful, it
    /// returns the point as an instance of Self. If the point cannot be decompressed, it returns None.
    fn decompress(x_or_y: Coordinate, rec_id: &u8) -> Option<Self>
    where
        Self: core::marker::Sized;

    /// For short Weierstrass curves, first parameter is the `x`-coordinate, for twisted Edwards curves,
    /// it is the `y`-coordinate.
    ///
    /// If it exists, hints the unique other coordinate `y_or_x` that is less than `Coordinate::MODULUS`
    /// such that `x_or_y` along with `y_or_x` is a point on the curve and `y_or_x` has parity equal to `rec_id`.
    /// If such `y_or_x` does not exist:
    /// - for short Weierstrass curves, hints a coordinate `sqrt` such that `sqrt^2 = (x^3 + a * x + b) * non_qr`
    /// - for twisted Edwards curves, hints a coordinate `sqrt` such that `(d * y^2 - a) * x^2 = (y^2 - 1) * non_qr`
    ///
    /// where `non_qr` is the non-quadratic residue for this curve that was initialized in the setup function.
    ///
    /// This is only a hint, and the returned value does not guarantee any of the above properties.
    /// They must be checked separately. Normal users should use `decompress` directly.
    ///
    /// Returns None if the `DecompressionHint::possible` flag in the hint stream is not a boolean.
    fn hint_decompress(x_or_y: &Coordinate, rec_id: &u8) -> Option<DecompressionHint<Coordinate>>;
}
