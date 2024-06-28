//! # RAP (Randomized Air with Preprocessing)
//! See <https://hackmd.io/@aztec-network/plonk-arithmetiization-air> for formal definition.

use p3_air::{BaseAir, PairBuilder, PermutationAirBuilder};
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::{
    air_builders::{
        debug::DebugConstraintBuilder, prover::ProverConstraintFolder,
        symbolic::SymbolicRapBuilder, verifier::VerifierConstraintFolder,
    },
    interaction::{AirBridge, InteractiveAir},
};

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
pub trait Rap<AB>: Sync
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
    fn permutation_exposed_values(&self) -> &[Self::VarEF];
}

/// RAP trait for all-purpose dynamic dispatch use.
/// This trait is auto-implemented if you implement `Air` and `Chip` traits.
pub trait AnyRap<SC: StarkGenericConfig>:
for<'a> InteractiveAir<ProverConstraintFolder<'a, SC>> // for prover permutation trace generation
    + for<'a> Rap<ProverConstraintFolder<'a, SC>> // for prover quotient polynomial calculation
    + for<'a> Rap<VerifierConstraintFolder<'a, SC>> // for verifier quotient polynomial calculation
    + for<'a> Rap<DebugConstraintBuilder<'a, SC>> // for debugging
    + BaseAir<Val<SC>> + AirBridge<Val<SC>> + Rap<SymbolicRapBuilder<Val<SC>>> // for keygen to extract fixed data about the RAP
{
}

impl<SC, T> AnyRap<SC> for T
where
    SC: StarkGenericConfig,
    T: for<'a> InteractiveAir<ProverConstraintFolder<'a, SC>>
        + for<'a> Rap<ProverConstraintFolder<'a, SC>>
        + for<'a> Rap<VerifierConstraintFolder<'a, SC>>
        + for<'a> Rap<DebugConstraintBuilder<'a, SC>>
        + BaseAir<Val<SC>>
        + AirBridge<Val<SC>>
        + Rap<SymbolicRapBuilder<Val<SC>>>,
{
}
