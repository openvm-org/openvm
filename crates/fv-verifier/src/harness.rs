//! Subprocess shim for the Lean-compiled verifier.
//!
//! Exposes:
//!
//! - [`run_fv_verifier`] writes an already-framed byte stream to the verifier.
//! - [`run_swirl_verify`] frames and writes a `(vk, proof, public values)` triple.
//! - [`verifier_error_from_exit_code`] — mirror of the Lean-side
//!   `Swirl.Protocol.Noninteractive.exitCode` table.
//!
//! The upstream crate's `build.rs` runs `lake build swirl_verify`
//! against the Lean sources; here the crate's `build.rs` instead
//! compiles the vendored Lean-generated C (`csrc/`) with `leanc` and
//! bakes the resulting exe path in. A `SWIRL_VERIFY_BIN` env var
//! overrides at runtime (and skips the C build at build time).

use std::{
    io::{self, Write},
    path::PathBuf,
    process::{Command, Stdio},
};

/// Resolve the `swirl_verify` Lean executable: the `SWIRL_VERIFY_BIN`
/// env var if set, otherwise the exe compiled from the vendored C
/// sources by this crate's `build.rs`.
pub fn swirl_verify_bin() -> PathBuf {
    if let Ok(path) = std::env::var("SWIRL_VERIFY_BIN") {
        if !path.is_empty() {
            return PathBuf::from(path);
        }
    }
    PathBuf::from(env!("SWIRL_VERIFY_BIN"))
}

/// Spawn the Lean verifier exe, write `bytes` to its stdin, and return
/// the exit code.
///
/// Returns [`io::ErrorKind::Other`] when the child exited via a signal
/// (no numeric exit code available).
pub fn run_fv_verifier(bytes: &[u8]) -> io::Result<i32> {
    let bin = swirl_verify_bin();
    let bin = bin.as_path();
    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;
    {
        let mut stdin = child.stdin.take().expect("child stdin piped");
        stdin.write_all(bytes)?;
    }
    let status = child.wait()?;
    status
        .code()
        .ok_or_else(|| io::Error::other(format!("swirl_verify terminated by signal: {status:?}")))
}

/// Outcome of an individual `swirl_verify` invocation.
#[derive(Debug)]
pub struct SwirlVerifyOutcome {
    pub exit_code: i32,
    pub stderr: String,
}

/// Frame the three blobs per `Tools/SwirlVerifyMain.lean`:
///
/// ```text
/// u32 LE vk_len | vk_bytes | u32 LE proof_len | proof_bytes | u32 LE pv_len | pv_bytes
/// ```
fn frame_three_blobs(vk_bytes: &[u8], proof_bytes: &[u8], pv_bytes: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(12 + vk_bytes.len() + proof_bytes.len() + pv_bytes.len());
    buf.extend_from_slice(&(vk_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(vk_bytes);
    buf.extend_from_slice(&(proof_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(proof_bytes);
    buf.extend_from_slice(&(pv_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(pv_bytes);
    buf
}

/// Spawn `swirl_verify`, pipe `(vk_bytes, proof_bytes, pv_bytes)` to its
/// stdin (with the u32-LE length framing documented above), and return
/// the exit code + captured stderr.
pub fn run_swirl_verify(
    vk_bytes: &[u8],
    proof_bytes: &[u8],
    pv_bytes: &[u8],
) -> io::Result<SwirlVerifyOutcome> {
    let bin = swirl_verify_bin();
    let framed = frame_three_blobs(vk_bytes, proof_bytes, pv_bytes);
    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    {
        let mut stdin = child.stdin.take().expect("child stdin piped");
        stdin.write_all(&framed)?;
    }
    let output = child.wait_with_output()?;
    let exit_code = output.status.code().ok_or_else(|| {
        io::Error::other(format!(
            "swirl_verify terminated by signal: {:?}",
            output.status
        ))
    })?;
    Ok(SwirlVerifyOutcome {
        exit_code,
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

/// Mirror of the Lean-side `Swirl.Protocol.Noninteractive.exitCode`
/// table, plus the `swirl_verify` driver's framing error (exit code 20).
///
/// Keep this in sync with the doc-comment table at the top of
/// `Tools/SwirlVerifyMain.lean`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifierError {
    /// `MonoError.parse _` — Wire.Raw parser rejected the bytes.
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
    /// Stdin framing error (Lean driver could not read three blobs).
    StdinFraming,
    /// An exit code that does not appear in the documented table.
    Unknown(i32),
}

/// Translate an exit code from `swirl_verify` to its corresponding
/// [`VerifierError`] variant. Returns `None` for exit 0 (accept).
///
/// The table mirrors `Swirl.Protocol.Noninteractive.exitCode` in
/// `Swirl/Protocol/Noninteractive/VerifierBabyBearPoseidon2.lean`.
/// Any deviation here breaks the FFI contract; keep the two in lockstep.
pub fn verifier_error_from_exit_code(code: i32) -> Option<VerifierError> {
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
        20 => Some(VerifierError::StdinFraming),
        other => Some(VerifierError::Unknown(other)),
    }
}
