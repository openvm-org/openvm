//! End-to-end tests for the Lean VM verifier.
//!
//! Each case generates a canonical OpenVM Fibonacci proof and then exercises
//! the five-input certified verifier.

use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::config::baby_bear_poseidon2::{EF, F};
use openvm_verify_stark_host::{vk::VmStarkVerifyingKey, VmStarkProof};

use super::fixtures::{fixture, optional_air_proofs, OPTIONAL_AIR_ARITY, OPTIONAL_AIR_ID};
use crate::{verify_vm_stark_proof, CertifiedVerifierError, VerifierError};

fn assert_rejects(
    vk: &VmStarkVerifyingKey,
    proof: &VmStarkProof,
    expected_error: VerifierError,
    case: &str,
) {
    match verify_vm_stark_proof(vk, proof) {
        Err(CertifiedVerifierError::Rejected {
            error,
            exit_code,
            stderr,
        }) => assert_eq!(
            error, expected_error,
            "certified verifier returned the wrong result for {case}; exit={exit_code}; stderr={stderr:?}"
        ),
        Err(error) => panic!("certified verifier failed to run for {case}: {error}"),
        Ok(()) => panic!("certified verifier accepted {case}"),
    }
}

#[test]
fn green_vm_proof_accepted() {
    let fixture = fixture();
    verify_vm_stark_proof(&fixture.vk, &fixture.proof)
        .expect("certified verifier accepts the green OpenVM proof");
}

#[test]
fn optional_air_public_values_follow_trace_presence() {
    let (vk, present_proof, absent_proof) = optional_air_proofs();
    verify_vm_stark_proof(&vk, &present_proof)
        .expect("certified verifier accepts a present optional positive-arity AIR");
    verify_vm_stark_proof(&vk, &absent_proof)
        .expect("certified verifier accepts an absent optional positive-arity AIR");

    let mut absent_with_values = absent_proof.clone();
    absent_with_values.inner.public_values[OPTIONAL_AIR_ID] = vec![F::ZERO; OPTIONAL_AIR_ARITY];
    assert_rejects(
        &vk,
        &absent_with_values,
        VerifierError::ProofShapeError,
        "public values for an absent positive-arity AIR",
    );

    let mut present_without_values = present_proof;
    present_without_values.inner.public_values[OPTIONAL_AIR_ID].clear();
    assert_rejects(
        &vk,
        &present_without_values,
        VerifierError::ProofShapeError,
        "missing public values for a present positive-arity AIR",
    );
}

#[test]
fn mismatched_executable_commitment_rejected() {
    let fixture = fixture();
    let mut invalid_vk = fixture.vk.clone();
    invalid_vk.baseline.app_exe_commit[0] += F::ONE;
    assert_rejects(
        &invalid_vk,
        &fixture.proof,
        VerifierError::PublicValues,
        "mismatched executable commitment",
    );
}

#[test]
fn truncated_final_polynomial_rejected() {
    let fixture = fixture();
    let mut invalid = fixture.proof.clone();
    invalid.inner.whir_proof.final_poly.pop();
    assert_rejects(
        &fixture.vk,
        &invalid,
        VerifierError::ProofShapeError,
        "truncated final polynomial",
    );
}

#[test]
fn tampered_common_main_commit_rejected() {
    let fixture = fixture();
    let mut invalid = fixture.proof.clone();
    invalid.inner.common_main_commit[0] += F::ONE;
    assert_rejects(
        &fixture.vk,
        &invalid,
        VerifierError::ChallengeDerivationError,
        "tampered common-main commitment",
    );
}

#[test]
fn tampered_batch_constraint_polynomial_rejected() {
    let fixture = fixture();
    let mut invalid = fixture.proof.clone();
    invalid.inner.batch_constraint_proof.univariate_round_coeffs[0] += EF::ONE;
    assert_rejects(
        &fixture.vk,
        &invalid,
        VerifierError::BatchConstraintError,
        "tampered batch-constraint polynomial",
    );
}

#[test]
fn tampered_gkr_proof_rejected() {
    let fixture = fixture();
    let mut invalid = fixture.proof.clone();
    invalid.inner.gkr_proof.q0_claim += EF::ONE;
    assert_rejects(
        &fixture.vk,
        &invalid,
        VerifierError::BatchConstraintError,
        "tampered GKR boundary claim",
    );
}

#[test]
fn tampered_stacked_reduction_polynomial_rejected() {
    let fixture = fixture();
    let mut invalid = fixture.proof.clone();
    invalid.inner.stacking_proof.univariate_round_coeffs[0] += EF::ONE;
    assert_rejects(
        &fixture.vk,
        &invalid,
        VerifierError::StackedReductionError,
        "tampered stacked-reduction polynomial",
    );
}

#[test]
fn tampered_whir_opening_rejected() {
    let fixture = fixture();
    let mut invalid = fixture.proof.clone();
    invalid.inner.whir_proof.initial_round_opened_rows[0][0][0][0] += F::ONE;
    assert_rejects(
        &fixture.vk,
        &invalid,
        VerifierError::WhirError,
        "tampered WHIR opening",
    );
}

#[test]
fn repeated_and_cross_thread_calls_are_accepted() {
    let fixture = fixture();
    let verify = || verify_vm_stark_proof(&fixture.vk, &fixture.proof);

    verify().expect("first verifier call succeeds");
    verify().expect("second verifier call succeeds");

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4).map(|_| scope.spawn(verify)).collect();
        for handle in handles {
            handle
                .join()
                .expect("verifier thread panicked")
                .expect("cross-thread verifier call succeeds");
        }
    });
}
