//! In-process FFI harness for the Lean-compiled verifier.
//!
//! Provides crate-internal helpers:
//!
//! - [`run_certified_verifier`] invokes the verifier on a `(vk, proof, public values)` triple.
//! - [`verifier_error_from_exit_code`] — mirror of the Lean-side
//!   `Swirl.Protocol.Noninteractive.exitCode` table.
//!
//! This crate's `build.rs` compiles the vendored Lean-generated C (`csrc/`)
//! with `leanc`, archives it, and links it and the pinned Lean runtime into
//! the Rust target. The private FFI module calls a single OpenVM-owned C adapter.

use std::io;
#[cfg(test)]
use std::path::PathBuf;

use crate::VerifierError;

/// Resolve the `swirl_dump_proof` wire-format test utility compiled from the
/// vendored C sources by this crate's `build.rs`.
#[cfg(test)]
pub(crate) fn swirl_dump_proof_bin() -> PathBuf {
    PathBuf::from(env!("OUT_DIR")).join("swirl_dump_proof")
}

/// Outcome of an individual certified-verifier invocation.
#[derive(Debug)]
pub(crate) struct SwirlVerifyOutcome {
    pub(crate) exit_code: i32,
    pub(crate) stderr: String,
}

/// Run the linked Lean verifier on `(vk_bytes, proof_bytes, pv_bytes)` and
/// return its exit code and rendered error.
///
/// This raw-wire entry point is crate-private. External callers should use
/// [`crate::verify_stark_proof`], which constructs the three blobs from typed inputs.
pub(crate) fn run_certified_verifier(
    vk_bytes: &[u8],
    proof_bytes: &[u8],
    pv_bytes: &[u8],
) -> io::Result<SwirlVerifyOutcome> {
    let (exit_code, message) = crate::ffi::verify(vk_bytes, proof_bytes, pv_bytes)?;
    Ok(SwirlVerifyOutcome {
        exit_code,
        stderr: if message.is_empty() {
            String::new()
        } else {
            format!("swirl_verify: {message}\n")
        },
    })
}

/// Translate an exit code from `swirl_verify` to its corresponding
/// [`VerifierError`] variant. Returns `None` for exit 0 (accept).
///
/// The table mirrors `Swirl.Protocol.Noninteractive.exitCode` in
/// `Swirl/Protocol/Noninteractive/VerifierBabyBearPoseidon2.lean`.
/// Any deviation here breaks the FFI contract; keep the two in lockstep.
pub(crate) fn verifier_error_from_exit_code(code: i32) -> Option<VerifierError> {
    match code {
        0 => None,
        1 => Some(VerifierError::ParseError),
        2 => Some(VerifierError::TraceHeightsTooLarge),
        3 => Some(VerifierError::PreprocessedTraceHeightMismatch),
        4 => Some(VerifierError::EmptyTraces),
        5 => Some(VerifierError::ProofShapeError),
        6 => Some(VerifierError::ChallengeDerivationError),
        7 => Some(VerifierError::BatchConstraintError),
        8 => Some(VerifierError::StackedReductionError),
        9 => Some(VerifierError::InvalidPrismPoint),
        10 => Some(VerifierError::WhirError),
        other => Some(VerifierError::Unknown(other)),
    }
}
