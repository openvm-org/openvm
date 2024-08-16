//! # RAP (Randomized Air with Preprocessing)
//! See <https://hackmd.io/@aztec-network/plonk-arithmetiization-air> for formal definition.

use std::any::{type_name, Any};

use p3_air::{BaseAir, PermutationAirBuilder};
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::air_builders::{
    debug::DebugConstraintBuilder, prover::ProverConstraintFolder, symbolic::SymbolicRapBuilder,
    verifier::VerifierConstraintFolder,
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
    AB: PermutationAirBuilder,
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
    Rap<SymbolicRapBuilder<Val<SC>>> // for keygen to extract fixed data about the RAP
    + for<'a> Rap<ProverConstraintFolder<'a, SC>> // for prover quotient polynomial calculation
    + for<'a> Rap<VerifierConstraintFolder<'a, SC>> // for verifier quotient polynomial calculation
    + for<'a> Rap<DebugConstraintBuilder<'a, SC>> // for debugging
    + BaseAir<Val<SC>>
{
    fn as_any(&self) -> &dyn Any;
    /// Name for display purposes
    fn name(&self) -> String;
}

impl<SC, T> AnyRap<SC> for T
where
    SC: StarkGenericConfig,
    T: Rap<SymbolicRapBuilder<Val<SC>>>
        + for<'a> Rap<ProverConstraintFolder<'a, SC>>
        + for<'a> Rap<VerifierConstraintFolder<'a, SC>>
        + for<'a> Rap<DebugConstraintBuilder<'a, SC>>
        + BaseAir<Val<SC>>
        + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> String {
        let full_name = type_name::<Self>().to_string();
        let base_name = full_name
            .split('<')
            .next()
            .unwrap_or(&full_name)
            .rsplit("::")
            .next()
            .unwrap_or(&full_name)
            .to_string();

        base_name
    }
}
