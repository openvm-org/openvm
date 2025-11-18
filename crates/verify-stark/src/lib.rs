use std::borrow::Borrow;

use continuations_v2::{
    aggregation::NonRootStarkProof,
    public_values::{NonRootVerifierPvs, verifier::VERIFIER_PVS_AIR_ID},
};
use eyre::Result;
use openvm_circuit::{
    arch::{ExitCode, hasher::poseidon2::vm_poseidon2_hasher},
    system::{memory::dimensions::MemoryDimensions, program::trace::compute_exe_commit},
};
use p3_field::{FieldAlgebra, PrimeField32};
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2, Digest, F, StarkEngineV2,
    keygen::types::MultiStarkVerifyingKeyV2, poseidon2::sponge::DuplexSponge,
};

use crate::error::VerifyStarkError;

pub mod error;

/// Baseline artifacts for a specific VM and fixed executable that are used to verify a final
/// (i.e. internal-recursive) VM STARK proof
pub struct VerificationBaseline {
    /// Commit to the app exe (i.e. hash of the program commit, initial memory merkle root,
    /// and initial program counter)
    pub app_exe_commit: Digest,
    /// VM memory metadata used to verify the user public values merkle proof
    pub memory_dimensions: MemoryDimensions,
    /// Cached trace commit of the leaf verifier circuit's SymbolicExpresionAir, which is
    /// derived from the app_vk
    pub leaf_commit: Digest,
    /// Cached trace commit of the internal-for-leaf verifier circuit's SymbolicExpresionAir,
    /// which derived from the leaf_vk
    pub internal_for_leaf_commit: Digest,
    /// Cached trace commit of the internal-recursive verifier circuit's SymbolicExpresionAir,
    /// which derived from the internal_for_leaf_vk
    pub internal_recursive_commit: Digest,
}

/// Verifies a non-root VM STARK proof given the internal-recursive layer verifying
/// key and VM- and exe-specific baseline artifacts.
pub fn verify_vm_stark_proof(
    vk: &MultiStarkVerifyingKeyV2,
    baseline: VerificationBaseline,
    proof: &NonRootStarkProof,
) -> Result<(), VerifyStarkError> {
    // Verify the STARK proof.
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(vk.inner.params);
    engine.verify(vk, &proof.inner)?;

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
        .verify(&hasher, baseline.memory_dimensions, final_root)?;

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
    if claimed_app_exe_commit != baseline.app_exe_commit {
        return Err(VerifyStarkError::AppExeCommitMismatch {
            expected: baseline.app_exe_commit,
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
    if leaf_commit != baseline.leaf_commit {
        return Err(VerifyStarkError::LeafCommitMismatch {
            expected: baseline.leaf_commit,
            actual: leaf_commit,
        });
    }

    // Check internal_for_leaf_commit against expected_commits.
    if internal_for_leaf_commit != baseline.internal_for_leaf_commit {
        return Err(VerifyStarkError::InternalForLeafCommitMismatch {
            expected: baseline.internal_for_leaf_commit,
            actual: internal_for_leaf_commit,
        });
    }

    // Check internal_for_leaf_commit against expected_commits.
    if internal_recursive_commit != baseline.internal_recursive_commit {
        return Err(VerifyStarkError::InternalRecursiveMismatch {
            expected: baseline.internal_recursive_commit,
            actual: internal_recursive_commit,
        });
    }
    Ok(())
}
