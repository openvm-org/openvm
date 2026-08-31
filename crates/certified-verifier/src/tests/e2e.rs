//! Crate-local end-to-end tests for the Lean monomorphic verifier.
//!
//! Generates a real FibonacciAir / BabyBearPoseidon2 proof through this
//! OpenVM workspace's `openvm-stark-backend` revision (the same fixture
//! pipeline used by `wire_roundtrip.rs`), serializes `(vk, proof, pv)`
//! through `openvm-certified-verifier`, passes the three blobs to the linked
//! Lean verifier through FFI, and asserts on the resulting exit code.
//!
//! Regression cases include:
//!
//! 1. `green_fib_proof_accepted` — honest proof must verify (exit 0).
//! 2. Optional positive-arity AIRs accept exactly the trace/public-value combinations permitted by
//!    the backend proof shape.
//! 3. `tampered_batch_constraint_proof_rejected` — a single-byte mutation inside the encoded GKR /
//!    Batch boundary claim must survive wire decoding and be rejected by the algebraic verifier
//!    prefix.

use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    proof::Proof,
    test_utils::{FibFixture, TestFixture},
    verifier::{
        proof_shape::{ProofShapeError, ProofShapeVDataError},
        VerifierError as RustVerifierError,
    },
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config, BabyBearPoseidon2RefEngine, DuplexSponge,
};

use crate::{
    harness::{run_certified_verifier, verifier_error_from_exit_code},
    proof::write_proof,
    public_values::write_public_values,
    vk::write_vk,
    VerifierError,
};

/// Base-2 logarithm of the Fibonacci trace length. Five is the smallest
/// value supported by the stacking and WHIR test parameters.
const LOG_TRACE_DEGREE: usize = 5;

type SC = BabyBearPoseidon2Config;

struct Fixture {
    vk: MultiStarkVerifyingKey<SC>,
    proof: Proof<SC>,
}

/// Generate a fresh (vk, proof) pair from FibonacciAir at `n = 2 ^
/// LOG_TRACE_DEGREE` rows. Sanity-checks the upstream prover by also
/// running the Rust verifier before returning.
fn generate_fib_proof() -> Fixture {
    let params = SystemParams::new_for_testing(LOG_TRACE_DEGREE);
    let engine = BabyBearPoseidon2RefEngine::<DuplexSponge>::new(params);
    let n = 1usize << LOG_TRACE_DEGREE;
    let (vk, proof) = FibFixture::new(0, 1, n).keygen_and_prove(&engine);
    engine
        .verify(&vk, &proof)
        .expect("upstream FibonacciAir prove+verify must succeed");
    Fixture { vk, proof }
}

/// Encode a key and proof into three wire blobs via `openvm-certified-verifier`.
fn encode_fixture(
    vk: &MultiStarkVerifyingKey<SC>,
    proof: &Proof<SC>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut vk_bytes = Vec::new();
    write_vk(&mut vk_bytes, vk).expect("write_vk");
    let mut proof_bytes = Vec::new();
    write_proof(&mut proof_bytes, proof).expect("write_proof");
    let mut pv_bytes = Vec::new();
    write_public_values(&mut pv_bytes, vk, &proof.public_values).expect("write_public_values");
    (vk_bytes, proof_bytes, pv_bytes)
}

fn read_u32_le(bytes: &[u8], offset: usize) -> usize {
    let chunk: [u8; 4] = bytes[offset..offset + 4]
        .try_into()
        .expect("u32 chunk in proof bytes");
    u32::from_le_bytes(chunk) as usize
}

fn skip_trace_vdata(bytes: &[u8], mut offset: usize) -> usize {
    let count = read_u32_le(bytes, offset);
    offset += 4;
    for _ in 0..count {
        let tag = bytes[offset];
        offset += 1;
        if tag == 1 {
            offset += 4; // logHeight
            let cached_count = read_u32_le(bytes, offset);
            offset += 4 + cached_count * 32;
        } else {
            assert_eq!(tag, 0, "unexpected Option tag in traceVdata");
        }
    }
    offset
}

/// Locate the low byte of `gkrProof.q0Claim[0]`. The Fibonacci
/// fixture has no GKR layers, so Batch expects this boundary claim to
/// equal one; mutating the low byte keeps the FBB word canonical but
/// breaks that algebraic check.
fn locate_gkr_boundary_claim_byte(proof_bytes: &[u8]) -> usize {
    let mut offset = 8; // PROF header
    offset += 32; // commonMainCommit
    offset = skip_trace_vdata(proof_bytes, offset);
    offset += 4; // gkrProof.logupPowWitness
    offset // gkrProof.q0Claim coefficient 0, low byte
}

// =====================================================================
// Green path
// =====================================================================

#[test]
fn green_fib_proof_accepted() {
    let fixture = generate_fib_proof();
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture.vk, &fixture.proof);
    let outcome =
        run_certified_verifier(&vk_bytes, &proof_bytes, &pv_bytes).expect("invoke verifier");
    assert_eq!(
        outcome.exit_code, 0,
        "green FibonacciAir proof rejected by swirl_verify; stderr={:?}",
        outcome.stderr
    );
}

#[test]
fn optional_positive_arity_public_values_follow_trace_presence() {
    let params = SystemParams::new_for_testing(LOG_TRACE_DEGREE);
    let engine = BabyBearPoseidon2RefEngine::<DuplexSponge>::new(params);
    let n = 1usize << LOG_TRACE_DEGREE;
    let present_fixture = FibFixture::new_with_num_airs(2, 3, n, 2);
    let (pk, vk) = present_fixture.keygen(&engine);

    let present_proof = present_fixture.prove(&engine, &pk);
    assert_eq!(present_proof.public_values[0].len(), 3);
    assert_eq!(present_proof.public_values[1].len(), 3);
    engine
        .verify(&vk, &present_proof)
        .expect("Rust accepts present AIRs with nominal public values");
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&vk, &present_proof);
    let outcome =
        run_certified_verifier(&vk_bytes, &proof_bytes, &pv_bytes).expect("invoke verifier");
    assert_eq!(
        outcome.exit_code, 0,
        "certified verifier rejected present nominal public values; stderr={:?}",
        outcome.stderr
    );

    let absent_fixture = FibFixture::new_with_num_airs(2, 3, n, 2).with_empty_air_indices([1]);
    let absent_proof = absent_fixture.prove(&engine, &pk);
    assert!(absent_proof.trace_vdata[1].is_none());
    assert!(absent_proof.public_values[1].is_empty());
    let absent_arity = vk.inner.per_air[1].params.num_public_values;
    assert_eq!(absent_arity, 3);
    engine
        .verify(&vk, &absent_proof)
        .expect("Rust accepts an absent optional AIR with an empty public-value row");
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&vk, &absent_proof);
    let outcome =
        run_certified_verifier(&vk_bytes, &proof_bytes, &pv_bytes).expect("invoke verifier");
    assert_eq!(
        outcome.exit_code, 0,
        "certified verifier rejected absent AIR with empty public values; stderr={:?}",
        outcome.stderr
    );

    let mut absent_with_values = absent_proof.clone();
    let zero = absent_with_values.public_values[0][0] - absent_with_values.public_values[0][0];
    absent_with_values.public_values[1] = vec![zero; absent_arity];
    let rust_error = engine
        .verify(&vk, &absent_with_values)
        .expect_err("Rust must reject public values without trace vdata");
    assert!(matches!(
        rust_error,
        RustVerifierError::ProofShapeError(ProofShapeError::InvalidVData(
            ProofShapeVDataError::PublicValuesNoVData { air_idx: 1 }
        ))
    ));
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&vk, &absent_with_values);
    let outcome =
        run_certified_verifier(&vk_bytes, &proof_bytes, &pv_bytes).expect("invoke verifier");
    assert_eq!(
        verifier_error_from_exit_code(outcome.exit_code),
        Some(VerifierError::ProofShapeError),
        "certified verifier should reject values for an absent AIR; stderr={:?}",
        outcome.stderr
    );

    let mut present_without_values = present_proof.clone();
    present_without_values.public_values[1].clear();
    let rust_error = engine
        .verify(&vk, &present_without_values)
        .expect_err("Rust must reject missing public values for a present AIR");
    assert!(matches!(
        rust_error,
        RustVerifierError::ProofShapeError(ProofShapeError::InvalidVData(
            ProofShapeVDataError::InvalidPublicValues {
                air_idx: 1,
                expected: 3,
                actual: 0
            }
        ))
    ));
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&vk, &present_without_values);
    let outcome =
        run_certified_verifier(&vk_bytes, &proof_bytes, &pv_bytes).expect("invoke verifier");
    assert_eq!(
        verifier_error_from_exit_code(outcome.exit_code),
        Some(VerifierError::ProofShapeError),
        "certified verifier should reject an empty row for a present AIR; stderr={:?}",
        outcome.stderr
    );
}

// =====================================================================
// Tampered path
// =====================================================================

// Exercises the real (interleaved TranscriptM) verifier wired into swirl_verify.
#[test]
fn tampered_batch_constraint_proof_rejected() {
    let fixture = generate_fib_proof();
    let (vk_bytes, mut proof_bytes, pv_bytes) = encode_fixture(&fixture.vk, &fixture.proof);
    let idx = locate_gkr_boundary_claim_byte(&proof_bytes);
    proof_bytes[idx] ^= 0x01;
    let outcome =
        run_certified_verifier(&vk_bytes, &proof_bytes, &pv_bytes).expect("invoke verifier");
    assert_ne!(
        outcome.exit_code, 0,
        "tampered proof accepted by swirl_verify; stderr={:?}",
        outcome.stderr
    );
    assert_eq!(
        outcome.exit_code, 7,
        "tampered GKR boundary claim should hit batchConstraintError, got {}; stderr={:?}",
        outcome.exit_code, outcome.stderr
    );
    assert!(matches!(
        verifier_error_from_exit_code(outcome.exit_code),
        Some(VerifierError::BatchConstraintError)
    ));
    assert!(
        !outcome.stderr.is_empty(),
        "Lean rejection should include the rendered MonoError"
    );
}

#[test]
fn repeated_and_cross_thread_calls_are_accepted() {
    let fixture = generate_fib_proof();
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture.vk, &fixture.proof);

    for _ in 0..2 {
        let outcome = run_certified_verifier(&vk_bytes, &proof_bytes, &pv_bytes)
            .expect("repeat verifier call");
        assert_eq!(outcome.exit_code, 0);
    }

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    run_certified_verifier(&vk_bytes, &proof_bytes, &pv_bytes)
                        .expect("cross-thread verifier call")
                        .exit_code
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().expect("verifier thread panicked"), 0);
        }
    });
}
