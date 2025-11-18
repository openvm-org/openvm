use std::borrow::Borrow;

use eyre::Result;
use openvm_circuit::{
    arch::{ExitCode, hasher::poseidon2::vm_poseidon2_hasher},
    system::{
        memory::merkle::public_values::UserPublicValuesProof, program::trace::compute_exe_commit,
    },
};
use p3_field::{FieldAlgebra, PrimeField32};
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2, DIGEST_SIZE, F, StarkEngineV2,
    codec::{Decode, Encode},
    poseidon2::sponge::DuplexSponge,
    proof::Proof,
};

use crate::{
    error::VerifyStarkError,
    pvs::{NonRootVerifierPvs, VERIFIER_PVS_AIR_ID},
    vk::NonRootStarkVerifyingKey,
};

pub mod error;
pub mod pvs;
pub mod vk;

// Final internal recursive STARK proof to be verified against the baseline
#[derive(Clone, Debug, Encode, Decode)]
pub struct NonRootStarkProof {
    pub inner: Proof,
    pub user_pvs_proof: UserPublicValuesProof<DIGEST_SIZE, F>,
}

/// Verifies a non-root VM STARK proof (as a byte stream) given the internal-recursive
/// layer verifying key and VM- and exe-specific baseline artifacts.
pub fn verify_vm_stark_proof(
    vk: &NonRootStarkVerifyingKey,
    encoded_proof: &[u8],
) -> Result<(), VerifyStarkError> {
    let decompressed = zstd::decode_all(encoded_proof)?;
    verify_vm_stark_proof_decoded(vk, &NonRootStarkProof::decode_from_bytes(&decompressed)?)
}

/// Verifies a non-root VM STARK proof given the internal-recursive layer verifying
/// key and VM- and exe-specific baseline artifacts.
pub fn verify_vm_stark_proof_decoded(
    vk: &NonRootStarkVerifyingKey,
    proof: &NonRootStarkProof,
) -> Result<(), VerifyStarkError> {
    // Verify the STARK proof.
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(vk.mvk.inner.params);
    engine.verify(&vk.mvk, &proof.inner)?;

    let &NonRootVerifierPvs::<F> {
        user_pv_commit,
        program_commit,
        initial_pc,
        exit_code,
        is_terminate,
        initial_root,
        final_root,
        internal_flag,
        leaf_commit,
        internal_for_leaf_commit,
        internal_recursive_commit,
        ..
    } = proof.inner.public_values[VERIFIER_PVS_AIR_ID]
        .as_slice()
        .borrow();
    let hasher = vm_poseidon2_hasher();

    // Verify the merkle root proof against final_root.
    proof
        .user_pvs_proof
        .verify(&hasher, vk.baseline.memory_dimensions, final_root)?;

    // Check that user_pv_commit is equal to the commit given in the merkle proof
    if user_pv_commit != proof.user_pvs_proof.public_values_commit {
        return Err(VerifyStarkError::UserPvCommitMismatch {
            expected: proof.user_pvs_proof.public_values_commit,
            actual: user_pv_commit,
        });
    }

    // Check that the app_commit is as expected.
    let claimed_app_exe_commit =
        compute_exe_commit(&hasher, &program_commit, &initial_root, initial_pc);
    if claimed_app_exe_commit != vk.baseline.app_exe_commit {
        return Err(VerifyStarkError::AppExeCommitMismatch {
            expected: vk.baseline.app_exe_commit,
            actual: claimed_app_exe_commit,
        });
    }

    // Check that the program terminated with a successful exit code.
    if exit_code.as_canonical_u32() != ExitCode::Success as u32 || is_terminate != F::ONE {
        return Err(VerifyStarkError::ExecutionUnsuccessful(exit_code));
    }

    // Check that the final proof is computed by the internal recursive prover, i.e.
    // that internal_flag is 2.
    if internal_flag != F::TWO {
        return Err(VerifyStarkError::InvalidInternalFlag(internal_flag));
    }

    // Check leaf_commit against expected_commits.
    if leaf_commit != vk.baseline.leaf_commit {
        return Err(VerifyStarkError::LeafCommitMismatch {
            expected: vk.baseline.leaf_commit,
            actual: leaf_commit,
        });
    }

    // Check internal_for_leaf_commit against expected_commits.
    if internal_for_leaf_commit != vk.baseline.internal_for_leaf_commit {
        return Err(VerifyStarkError::InternalForLeafCommitMismatch {
            expected: vk.baseline.internal_for_leaf_commit,
            actual: internal_for_leaf_commit,
        });
    }

    // Check internal_for_leaf_commit against expected_commits.
    if internal_recursive_commit != vk.baseline.internal_recursive_commit {
        return Err(VerifyStarkError::InternalRecursiveMismatch {
            expected: vk.baseline.internal_recursive_commit,
            actual: internal_recursive_commit,
        });
    }
    Ok(())
}
