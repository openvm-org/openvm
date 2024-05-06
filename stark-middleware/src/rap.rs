//! # RAP (Randomized Air with Preprocessing)
//! See <https://hackmd.io/@aztec-network/plonk-arithmetiization-air> for formal definition.

use p3_air::{PairBuilder, PermutationAirBuilder};

/// An AIR that works with a particular `AirBuilder` which allows preprocessing
/// and injected randomness.
///
/// Currently this is not a fully general RAP. Only the following phases are allowed:
/// - Preprocessing
/// - Main trace generation and commitment
/// - Permutation trace generation and commitment
/// Randomness is drawn after the main trace commitment phase, and used in the permutation trace.
///
/// Does not inherit [Air](p3_air::Air) trait to allow overrides for technical reasons
/// around dynamic dispatch.
pub trait Rap<AB>
where
    AB: PairBuilder + PermutationAirBuilder,
{
    fn eval(&self, builder: &mut AB);
}

/// Permutation AIR builder that exposes certain values to both prover and verifier
/// _after_ the permutation challenges are drawn. These can be thought of as
/// "public values" known after the challenges are drawn.
///
/// Exposed values are used internally by the prover and verifier
/// in cross-table permutation arguments.
pub trait PermutationAirBuilderWithExposedValues: PermutationAirBuilder {
    fn permutation_exposed_values(&self) -> &[Self::EF];
}
