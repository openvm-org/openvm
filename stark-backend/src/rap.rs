//! # RAP (Randomized Air with Preprocessing)
//! See <https://hackmd.io/@aztec-network/plonk-arithmetiization-air> for formal definition.

use std::any::{type_name, Any};

use p3_air::{BaseAir, PermutationAirBuilder};
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::air_builders::{
    debug::DebugConstraintBuilder, prover::ProverConstraintFolder, symbolic::SymbolicRapBuilder,
};

/// An AIR with 0 or more public values.
/// This trait will be merged into Plonky3 in PR: https://github.com/Plonky3/Plonky3/pull/470
pub trait BaseAirWithPublicValues<F>: BaseAir<F> {
    fn num_public_values(&self) -> usize {
        0
    }
}

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
    + for<'a> Rap<DebugConstraintBuilder<'a, SC>> // for debugging
    + BaseAirWithPublicValues<Val<SC>>
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
        + for<'a> Rap<DebugConstraintBuilder<'a, SC>>
        + BaseAirWithPublicValues<Val<SC>>
        + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> String {
        let full_name = type_name::<Self>().to_string();
        // Split the input by the first '<' to separate the main type from its generics
        if let Some((main_part, generics_part)) = full_name.split_once('<') {
            // Extract the last segment of the main type
            let main_type = main_part.split("::").last().unwrap_or("");

            // Remove the trailing '>' from the generics part and split by ", " to handle multiple generics
            let generics: Vec<String> = generics_part
                .trim_end_matches('>')
                .split(", ")
                .map(|generic| {
                    // For each generic type, extract the last segment after "::"
                    generic.split("::").last().unwrap_or("").to_string()
                })
                .collect();

            // Join the simplified generics back together with ", " and format the result
            format!("{}<{}>", main_type, generics.join(", "))
        } else {
            // If there's no generic part, just return the last segment after "::"
            full_name.split("::").last().unwrap_or("").to_string()
        }
    }
}
