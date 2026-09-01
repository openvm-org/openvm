//! Verification with the certified Lean VM verifier extracted from its formalization.
//!
//! The internal wire encoders serialize the recursive VK, verification baseline, proof, per-AIR
//! public values, and user-public-values proof into the five self-describing blobs consumed by
//! Lean. The internal FFI harness invokes the linked Lean verifier on those exact inputs.
//!
//! The verifier itself is Lean compiled to C: the generated C sources use a lightweight link
//! closure and Lean toolchain `leanprover/lean4:v4.26.0`. They are vendored under `csrc/`, compiled
//! by this crate's `build.rs`, and linked directly into the Rust target.

use harness::{run_certified_verifier, verifier_error_from_exit_code};
use openvm_verify_stark_host::{vk::VmStarkVerifyingKey, VmStarkProof};
use proof::write_proof;
use public_values::write_public_values;
use vk::write_vk;
use vm::{write_user_public_values_proof, write_verification_baseline};

mod ffi;
mod harness;
mod magic;
mod primitives;
mod proof;
mod public_values;
mod symbolic;
mod vk;
mod vm;

#[cfg(test)]
mod tests;

/// Rejection reported by the certified Lean VM verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifierError {
    /// A wire parser rejected one of the five inputs.
    ParseError,
    /// `VerifierError.traceHeightsTooLarge`.
    TraceHeightsTooLarge,
    /// `VerifierError.preprocessedTraceHeightMismatch`.
    PreprocessedTraceHeightMismatch,
    /// `VerifierError.emptyTraces`.
    EmptyTraces,
    /// `VerifierError.proofShapeError`.
    ProofShapeError,
    /// `VerifierError.challengeDerivationError`.
    ChallengeDerivationError,
    /// `VerifierError.batchConstraintError`.
    BatchConstraintError,
    /// `VerifierError.stackedReductionError`.
    StackedReductionError,
    /// `VerifierError.invalidPrismPoint`.
    InvalidPrismPoint,
    /// `VerifierError.whirError`.
    WhirError,
    /// `VerifierError.systemParamsMismatch`.
    SystemParamsMismatch,
    /// VM host public-value validation failed after STARK acceptance.
    PublicValues,
    /// An exit code that does not appear in the documented table.
    Unknown(i32),
}

/// Failure of a [`verify_vm_stark_proof`] run.
#[derive(Debug, thiserror::Error)]
pub enum CertifiedVerifierError {
    /// Encoding the inputs or invoking the verifier runtime failed; no verdict
    /// on the proof was reached.
    #[error("failed to run the certified verifier: {0}")]
    Io(#[from] std::io::Error),
    /// The verifier ran and rejected the proof.
    #[error(
        "certified verifier rejected the proof: {error:?} (exit {exit_code}), stderr: {stderr}"
    )]
    Rejected {
        error: VerifierError,
        exit_code: i32,
        stderr: String,
    },
}

/// Verify a complete non-deferral VM STARK proof with the linked Lean verifier.
///
/// This verifies both the inner recursive STARK and the host-side VM public-value
/// conditions against `vk.baseline`.
pub fn verify_vm_stark_proof(
    vk: &VmStarkVerifyingKey,
    proof: &VmStarkProof,
) -> Result<(), CertifiedVerifierError> {
    if proof.deferral_merkle_proofs.is_some() {
        return Err(CertifiedVerifierError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "the certified VM verifier does not support deferral proofs",
        )));
    }

    let mut vk_bytes = Vec::new();
    write_vk(&mut vk_bytes, &vk.mvk)?;
    let mut baseline_bytes = Vec::new();
    write_verification_baseline(&mut baseline_bytes, &vk.baseline)?;
    let mut proof_bytes = Vec::new();
    write_proof(&mut proof_bytes, &proof.inner)?;
    let mut pv_bytes = Vec::new();
    write_public_values(&mut pv_bytes, &vk.mvk, &proof.inner.public_values)?;
    let mut user_pvs_bytes = Vec::new();
    write_user_public_values_proof(&mut user_pvs_bytes, &proof.user_pvs_proof)?;

    let outcome = run_certified_verifier(
        &vk_bytes,
        &baseline_bytes,
        &proof_bytes,
        &pv_bytes,
        &user_pvs_bytes,
    )?;
    match verifier_error_from_exit_code(outcome.exit_code) {
        None => Ok(()),
        Some(error) => Err(CertifiedVerifierError::Rejected {
            error,
            exit_code: outcome.exit_code,
            stderr: outcome.stderr,
        }),
    }
}
