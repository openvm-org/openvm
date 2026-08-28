//! End-to-end tests for the Lean monomorphic verifier.
//!
//! Generates a real FibonacciAir / BabyBearPoseidon2 proof through this
//! OpenVM workspace's `openvm-stark-backend` revision (the same fixture
//! pipeline used by `wire_roundtrip.rs`), serializes `(vk, proof, pv)`
//! through `openvm-certified-verifier`, passes the three blobs to the linked
//! Lean verifier through FFI, and asserts on the resulting exit code.
//!
//! Two cases:
//!
//! 1. `green_fib_proof_accepted` — honest proof must verify (exit 0).
//! 2. `tampered_batch_constraint_proof_rejected` — a single-byte mutation inside the encoded GKR /
//!    Batch boundary claim must survive wire decoding and be rejected by the algebraic verifier
//!    prefix.

use openvm_certified_verifier::{
    run_certified_verifier, verifier_error_from_exit_code, write_proof, write_public_values,
    write_vk, VerifierError,
};
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    proof::Proof,
    test_utils::{FibFixture, TestFixture},
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config, BabyBearPoseidon2RefEngine, DuplexSponge,
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

/// Encode the fixture into three wire blobs via `openvm-certified-verifier`.
fn encode_fixture(f: &Fixture) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut vk_bytes = Vec::new();
    write_vk(&mut vk_bytes, &f.vk).expect("write_vk");
    let mut proof_bytes = Vec::new();
    write_proof(&mut proof_bytes, &f.proof).expect("write_proof");
    let mut pv_bytes = Vec::new();
    write_public_values(&mut pv_bytes, &f.vk, &f.proof.public_values).expect("write_public_values");
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
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture);
    let outcome =
        run_certified_verifier(&vk_bytes, &proof_bytes, &pv_bytes).expect("invoke verifier");
    assert_eq!(
        outcome.exit_code, 0,
        "green FibonacciAir proof rejected by swirl_verify; stderr={:?}",
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
    let (vk_bytes, mut proof_bytes, pv_bytes) = encode_fixture(&fixture);
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
    let (vk_bytes, proof_bytes, pv_bytes) = encode_fixture(&fixture);

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
