use alloc::vec::Vec;

use elliptic_curve::subtle::{Choice, ConditionallySelectable, ConstantTimeEq};
use openvm_algebra_guest::IntMod;

use crate::internal::Secp256k1Coord;

// --- Implement elliptic_curve traits on Secp256k1Coord ---

impl Copy for Secp256k1Coord {}

impl Default for Secp256k1Coord {
    fn default() -> Self {
        <Self as IntMod>::ZERO
    }
}

impl ConditionallySelectable for Secp256k1Coord {
    fn conditional_select(
        a: &Secp256k1Coord,
        b: &Secp256k1Coord,
        choice: Choice,
    ) -> Secp256k1Coord {
        Secp256k1Coord::from_le_bytes_unchecked(
            &a.as_le_bytes_unchecked()
                .iter()
                .zip(b.as_le_bytes_unchecked().iter())
                .map(|(a, b)| u8::conditional_select(a, b, choice))
                .collect::<Vec<_>>(),
        )
    }
}

impl ConstantTimeEq for Secp256k1Coord {
    /// Compares field elements for equality. On the host, both operands must be reduced.
    fn ct_eq(&self, other: &Self) -> Choice {
        #[cfg(not(target_os = "zkvm"))]
        {
            self.as_le_bytes_unchecked()
                .ct_eq(other.as_le_bytes_unchecked())
        }
        #[cfg(target_os = "zkvm")]
        {
            // Compare the difference with zero to test equality modulo the modulus.
            Choice::from(((self - other) == <Self as IntMod>::ZERO) as u8)
        }
    }
}
