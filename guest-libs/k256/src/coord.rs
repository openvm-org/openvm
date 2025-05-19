use alloc::vec::Vec;

use elliptic_curve::subtle::{Choice, ConditionallySelectable, ConstantTimeEq};
use openvm_algebra_guest::IntMod;

use crate::internal::Secp256k1Coord;

// --- Implement elliptic_curve traits on Secp256k1Coord ---

impl Copy for Secp256k1Coord {}

impl ConditionallySelectable for Secp256k1Coord {
    fn conditional_select(
        a: &Secp256k1Coord,
        b: &Secp256k1Coord,
        choice: Choice,
    ) -> Secp256k1Coord {
        Secp256k1Coord::from_le_bytes(
            &a.as_le_bytes()
                .iter()
                .zip(b.as_le_bytes().iter())
                .map(|(a, b)| u8::conditional_select(a, b, choice))
                .collect::<Vec<_>>(),
        )
    }
}

// Requires canonical form
impl ConstantTimeEq for Secp256k1Coord {
    fn ct_eq(&self, other: &Secp256k1Coord) -> Choice {
        self.as_le_bytes().ct_eq(other.as_le_bytes())
    }
}
