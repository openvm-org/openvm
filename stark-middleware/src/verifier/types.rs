use p3_air::Air;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::{
    air_builders::{symbolic::SymbolicAirBuilder, verifier::VerifierConstraintFolder},
    rap::Rap,
};

/// AIR trait for verifier use.
pub trait VerifierRap<SC: StarkGenericConfig>:
    for<'a> Rap<VerifierConstraintFolder<'a, SC>> + Air<SymbolicAirBuilder<Val<SC>>>
{
}

impl<SC: StarkGenericConfig, T> VerifierRap<SC> for T where
    T: for<'a> Rap<VerifierConstraintFolder<'a, SC>> + Air<SymbolicAirBuilder<Val<SC>>>
{
}
