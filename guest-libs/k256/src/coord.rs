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
            &a.as_le_bytes()
                .iter()
                .zip(b.as_le_bytes().iter())
                .map(|(a, b)| u8::conditional_select(a, b, choice))
                .collect::<Vec<_>>(),
        )
    }
}

impl ConstantTimeEq for Secp256k1Coord {
    fn ct_eq(&self, other: &Secp256k1Coord) -> Choice {
        #[cfg(not(target_os = "zkvm"))]
        {
            // Requires canonical form
            self.as_le_bytes().ct_eq(other.as_le_bytes())
        }
        #[cfg(target_os = "zkvm")]
        {
            // The zkVM implementation calls iseqmod opcode so it is constant time, _except_ a check
            // of whether the setup opcode has been called already
            Choice::from((self == other) as u8)
        }
    }
}
