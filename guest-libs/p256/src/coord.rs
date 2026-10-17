use alloc::vec::Vec;

use elliptic_curve::subtle::{Choice, ConditionallySelectable, ConstantTimeEq};
use openvm_algebra_guest::IntMod;

use crate::internal::P256Coord;

// --- Implement elliptic_curve traits on P256Coord ---

impl Copy for P256Coord {}

impl Default for P256Coord {
    fn default() -> Self {
        <Self as IntMod>::ZERO
    }
}

impl ConditionallySelectable for P256Coord {
    fn conditional_select(a: &P256Coord, b: &P256Coord, choice: Choice) -> P256Coord {
        P256Coord::from_le_bytes_unchecked(
            &a.as_le_bytes_unchecked()
                .iter()
                .zip(b.as_le_bytes_unchecked().iter())
                .map(|(a, b)| u8::conditional_select(a, b, choice))
                .collect::<Vec<_>>(),
        )
    }
}

impl ConstantTimeEq for P256Coord {
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
