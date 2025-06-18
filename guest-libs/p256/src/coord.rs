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
            &a.as_le_bytes()
                .iter()
                .zip(b.as_le_bytes().iter())
                .map(|(a, b)| u8::conditional_select(a, b, choice))
                .collect::<Vec<_>>(),
        )
    }
}

impl ConstantTimeEq for P256Coord {
    fn ct_eq(&self, other: &P256Coord) -> Choice {
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
