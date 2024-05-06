use p3_air::Air;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

use crate::{
    air_builders::{symbolic::SymbolicAirBuilder, verifier::VerifierConstraintFolder},
    rap::Rap,
};

// TODO: add this to verifying key
/// Verifier data for single RAP
/// The data is for the traces committed into a single commitment.
#[derive(Serialize, Deserialize)]
pub struct VerifierSingleRapMetadata {
    /// Height of trace matrix
    pub degree: usize,
    /// The factor to multiply trace matrix height by to get the quotient
    /// polynomial degree, equals the number of quotient chunks
    pub quotient_degree: usize,
    /// (i, j) where i is the index of the main trace commitment, and
    /// j is the index of the matrix in that commitment
    pub main_trace_ptr: (usize, usize),
    /// Index of the permutation trace matrix in the permutation trace
    /// commitment, if any
    pub perm_trace_index: Option<usize>,
    /// Global index, corresponds to the index of this RAP's quotient polynomial
    /// among all quotient polynomials
    pub index: usize,
}

/// AIR trait for verifier use.
pub trait VerifierRap<SC: StarkGenericConfig>:
    for<'a> Rap<VerifierConstraintFolder<'a, SC>> + Air<SymbolicAirBuilder<Val<SC>>>
{
}

impl<SC: StarkGenericConfig, T> VerifierRap<SC> for T where
    T: for<'a> Rap<VerifierConstraintFolder<'a, SC>> + Air<SymbolicAirBuilder<Val<SC>>>
{
}
