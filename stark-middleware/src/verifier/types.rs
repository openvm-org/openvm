use p3_uni_stark::StarkGenericConfig;

use crate::{air_builders::verifier::VerifierConstraintFolder, rap::Rap};

/// AIR trait for verifier use.
pub trait VerifierRap<SC: StarkGenericConfig>:
    for<'a> Rap<VerifierConstraintFolder<'a, SC>>
{
}

impl<SC: StarkGenericConfig, T> VerifierRap<SC> for T where
    T: for<'a> Rap<VerifierConstraintFolder<'a, SC>>
{
}
