use std::borrow::Borrow;

use eyre::Result;
use openvm_circuit::{
    arch::{hasher::poseidon2::vm_poseidon2_hasher, ExitCode},
    system::{
        memory::merkle::public_values::UserPublicValuesProof, program::trace::compute_exe_commit,
    },
};
use openvm_stark_backend::{
    codec::{Decode, Encode},
    proof::Proof,
    StarkEngine,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as SC, BabyBearPoseidon2CpuEngine, DuplexSponge, DIGEST_SIZE, F,
};
use p3_field::{PrimeCharacteristicRing, PrimeField32};

use crate::{
    error::VerifyStarkError,
    pvs::{NonRootVerifierPvs, VERIFIER_PVS_AIR_ID},
    vk::NonRootStarkVerifyingKey,
};

pub mod error;
pub mod pvs;
pub mod vk;

// Final internal recursive STARK proof to be verified against the baseline
#[derive(Clone, Debug)]
pub struct NonRootStarkProof {
    pub inner: Proof<SC>,
    pub user_pvs_proof: UserPublicValuesProof<DIGEST_SIZE, F>,
}

impl Encode for NonRootStarkProof {
    fn encode<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.inner.encode(writer)?;
        self.user_pvs_proof.encode::<SC, _>(writer)?;
        Ok(())
    }
}

impl Decode for NonRootStarkProof {
    fn decode<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let inner = Proof::<SC>::decode(reader)?;
        let user_pvs_proof = UserPublicValuesProof::decode::<SC, _>(reader)?;
        Ok(Self {
            inner,
            user_pvs_proof,
        })
    }
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
    let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(vk.mvk.inner.params.clone());
    engine.verify(&vk.mvk, &proof.inner)?;

    let &NonRootVerifierPvs::<F> {
        program_commit,
        initial_pc,
        exit_code,
        is_terminate,
        initial_root,
        final_root,
        internal_flag,
        app_dag_commit,
        leaf_dag_commit,
        internal_for_leaf_dag_commit,
        recursion_flag,
        internal_recursive_dag_commit,
        ..
    } = proof.inner.public_values[VERIFIER_PVS_AIR_ID]
        .as_slice()
        .borrow();
    let hasher = vm_poseidon2_hasher();

    // Verify the merkle root proof against final_root.
    proof
        .user_pvs_proof
        .verify(&hasher, vk.baseline.memory_dimensions, final_root)?;

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

    // Check app_dag_commit against expected_commits.
    if app_dag_commit != vk.baseline.app_dag_commit {
        return Err(VerifyStarkError::AppDagCommitMismatch {
            expected: vk.baseline.app_dag_commit,
            actual: app_dag_commit,
        });
    }

    // Check leaf_dag_commit against expected_commits.
    if leaf_dag_commit != vk.baseline.leaf_dag_commit {
        return Err(VerifyStarkError::LeafDagCommitMismatch {
            expected: vk.baseline.leaf_dag_commit,
            actual: leaf_dag_commit,
        });
    }

    // Check internal_for_leaf_dag_commit against expected_commits.
    if internal_for_leaf_dag_commit != vk.baseline.internal_for_leaf_dag_commit {
        return Err(VerifyStarkError::InternalForLeafDagCommitMismatch {
            expected: vk.baseline.internal_for_leaf_dag_commit,
            actual: internal_for_leaf_dag_commit,
        });
    }

    // Check that recursion_flag is 2, i.e. that the penultimate layer is internal
    // recursive.
    if recursion_flag != F::TWO {
        return Err(VerifyStarkError::InvalidRecursionFlag(recursion_flag));
    }

    // Check internal_recursive_dag_commit against expected_commits.
    if internal_recursive_dag_commit != vk.baseline.internal_recursive_dag_commit {
        return Err(VerifyStarkError::InternalRecursiveDagCommitMismatch {
            expected: vk.baseline.internal_recursive_dag_commit,
            actual: internal_recursive_dag_commit,
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
