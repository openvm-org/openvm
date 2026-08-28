//! Verification with the certified Lean Swirl verifier extracted from its formalization.
//!
//! Vendored from the private `swirl-rbr-fv` repo (the
//! `proof-wire` crate and `lean-verifier-harness` lib of its
//! `verifier-ffi/` workspace, merged into one crate) — see `README.md`
//! in this crate's directory.
//!
//! Two halves:
//!
//! - The wire encoder ([`write_vk`], [`write_proof`], [`write_public_values`]): hand-rolled,
//!   serde-free serialization of `MultiStarkVerifyingKey<SC>` / `Proof<SC>` / public values into
//!   the self-describing byte format consumed by the Lean decoders in
//!   `swirl-rbr-fv:Swirl/Protocol/Noninteractive/Wire/`. The byte layout mirrors the *Lean* struct
//!   tree, not the Rust one.
//! - The FFI harness ([`run_certified_verifier`], [`verifier_error_from_exit_code`]): passes the
//!   three blobs to the linked Lean-compiled verifier and maps its exit code (a mirror of
//!   `swirl-rbr-fv:Tools/SwirlVerifyMain.lean`; keep the two tables in lockstep).
//!
//! The verifier itself is Lean compiled to C: the generated C
//! sources (24-module link closure, Lean toolchain `leanprover/lean4:v4.26.0`)
//! are vendored under `csrc/`, compiled by this crate's `build.rs` with
//! `leanc`, and linked into the Rust target. A small OpenVM-owned C adapter
//! contains the Lean object ABI boundary and exposes one raw-buffer function
//! to Rust.

mod ffi;
pub mod harness;
pub mod magic;
pub mod primitives;
pub mod proof;
pub mod public_values;
pub mod symbolic;
pub mod vk;

pub use harness::{
    run_certified_verifier, swirl_dump_proof_bin, verifier_error_from_exit_code,
    SwirlVerifyOutcome, VerifierError,
};
pub use magic::{MAGIC_PROOF, MAGIC_PUBLIC_VALUES, MAGIC_VK, WIRE_VERSION};
use openvm_stark_backend::{
    codec::EncodableConfig, keygen::types::MultiStarkVerifyingKey, p3_field::PrimeField32,
    proof::Proof,
};
pub use proof::write_proof;
pub use public_values::write_public_values;
pub use vk::write_vk;

/// Failure of a [`verify_stark_proof`] run.
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

/// Verify `(vk, proof)` with the linked Lean-compiled verifier: encode the vk,
/// the proof body, and the proof's per-AIR public values to the Lean wire
/// format, invoke it through FFI, and map its exit code.
///
/// `Ok(())` means the formally verified verifier accepted the proof.
pub fn verify_stark_proof<SC: EncodableConfig>(
    vk: &MultiStarkVerifyingKey<SC>,
    proof: &Proof<SC>,
) -> Result<(), CertifiedVerifierError>
where
    SC::F: PrimeField32,
{
    let mut vk_bytes = Vec::new();
    write_vk(&mut vk_bytes, vk)?;
    let mut proof_bytes = Vec::new();
    write_proof(&mut proof_bytes, proof)?;
    let mut pv_bytes = Vec::new();
    write_public_values(&mut pv_bytes, vk, &proof.public_values)?;

    let outcome = run_certified_verifier(&vk_bytes, &proof_bytes, &pv_bytes)?;
    match verifier_error_from_exit_code(outcome.exit_code) {
        None => Ok(()),
        Some(error) => Err(CertifiedVerifierError::Rejected {
            error,
            exit_code: outcome.exit_code,
            stderr: outcome.stderr,
        }),
    }
}
