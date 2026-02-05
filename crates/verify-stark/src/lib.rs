use std::borrow::Borrow;

use eyre::Result;
use openvm_circuit::{
    arch::{ExitCode, hasher::poseidon2::vm_poseidon2_hasher},
    system::{
        memory::merkle::public_values::UserPublicValuesProof, program::trace::compute_exe_commit,
    },
};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
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
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(vk.mvk.inner.params.clone());
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
        app_vk_commit,
        leaf_vk_commit,
        internal_for_leaf_vk_commit,
        recursion_flag,
        internal_recursive_vk_commit,
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

    // Check that the final proof is computed by the internal recursive (or compression)
    // prover, i.e. that internal_flag is 2.
    if internal_flag != F::TWO {
        return Err(VerifyStarkError::InvalidInternalFlag(internal_flag));
    }

    // Check app_vk_commit against expected_commits.
    if app_vk_commit != vk.baseline.app_vk_commit {
        return Err(VerifyStarkError::AppVkCommitMismatch {
            expected: vk.baseline.app_vk_commit,
            actual: app_vk_commit,
        });
    }

    // Check leaf_vk_commit against expected_commits.
    if leaf_vk_commit != vk.baseline.leaf_vk_commit {
        return Err(VerifyStarkError::LeafVkCommitMismatch {
            expected: vk.baseline.leaf_vk_commit,
            actual: leaf_vk_commit,
        });
    }

    // Check internal_for_leaf_vk_commit against expected_commits.
    if internal_for_leaf_vk_commit != vk.baseline.internal_for_leaf_vk_commit {
        return Err(VerifyStarkError::InternalForLeafVkCommitMismatch {
            expected: vk.baseline.internal_for_leaf_vk_commit,
            actual: internal_for_leaf_vk_commit,
        });
    }

    // Check that recursion_flag is 2, i.e. that the penultimate layer is internal
    // recursive.
    if recursion_flag != F::TWO {
        return Err(VerifyStarkError::InvalidRecursionFlag(recursion_flag));
    }

    // Check internal_recursive_vk_commit against expected_commits.
    if internal_recursive_vk_commit != vk.baseline.internal_recursive_vk_commit {
        return Err(VerifyStarkError::InternalRecursiveVkCommitMismatch {
            expected: vk.baseline.internal_recursive_vk_commit,
            actual: internal_recursive_vk_commit,
        });
    }

    // Check that the public values of the last AIR matches up with the expected
    // compression_commit if compression is enabled, else ensure the last AIR has
    // no public values.
    let compression_commit_pvs = proof.inner.public_values.last().unwrap().clone();
    if let Some(expected_compression_commit) = vk.baseline.compression_commit.as_ref() {
        let expected_expression_commit = expected_compression_commit.to_vec();
        if compression_commit_pvs != expected_expression_commit {
            return Err(VerifyStarkError::CompressionCommitMismatch {
                expected: expected_expression_commit,
                actual: compression_commit_pvs,
            });
        }
    } else if !compression_commit_pvs.is_empty() {
        return Err(VerifyStarkError::CompressionCommitDefined {
            actual: compression_commit_pvs,
        });
    }

    Ok(())
}
