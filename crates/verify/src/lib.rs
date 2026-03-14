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
    deferral::DeferralMerkleProofs,
    error::VerifyStarkError,
    pvs::{
        DeferralPvs, VerifierBasePvs, VerifierDefPvs, VmPvs, DEF_PVS_AIR_ID, VERIFIER_PVS_AIR_ID,
        VM_PVS_AIR_ID,
    },
    vk::NonRootStarkVerifyingKey,
};

pub mod deferral;
pub mod error;
pub mod pvs;
pub mod vk;

pub(crate) type DagCommit = pvs::DagCommit<F>;

// Final internal recursive STARK proof to be verified against the baseline
#[derive(Clone, Debug)]
pub struct NonRootStarkProof {
    pub inner: Proof<SC>,
    pub user_pvs_proof: UserPublicValuesProof<DIGEST_SIZE, F>,
    pub deferral_merkle_proofs: Option<DeferralMerkleProofs<F>>,
}

impl Encode for NonRootStarkProof {
    fn encode<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.inner.encode(writer)?;
        self.user_pvs_proof.encode::<SC, _>(writer)?;
        (self.deferral_merkle_proofs.is_some() as u8).encode(writer)?;
        if let Some(ref proofs) = self.deferral_merkle_proofs {
            proofs.encode(writer)?;
        }
        Ok(())
    }
}

impl Decode for NonRootStarkProof {
    fn decode<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let inner = Proof::<SC>::decode(reader)?;
        let user_pvs_proof = UserPublicValuesProof::decode::<SC, _>(reader)?;
        let deferral_merkle_proofs = if u8::decode(reader)? != 0 {
            Some(DeferralMerkleProofs::decode(reader)?)
        } else {
            None
        };
        Ok(Self {
            inner,
            user_pvs_proof,
            deferral_merkle_proofs,
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

    let (verifier_base_pvs_slice, verifier_def_pvs_slice) = proof.inner.public_values
        [VERIFIER_PVS_AIR_ID]
        .as_slice()
        .split_at(VerifierBasePvs::<u8>::width());

    let &VerifierBasePvs::<F> {
        internal_flag,
        app_dag_commit,
        leaf_dag_commit,
        internal_for_leaf_dag_commit,
        recursion_flag,
        internal_recursive_dag_commit,
    } = verifier_base_pvs_slice.borrow();

    let &VmPvs::<F> {
        program_commit,
        initial_pc,
        exit_code,
        is_terminate,
        initial_root,
        final_root,
        ..
    } = proof.inner.public_values[VM_PVS_AIR_ID].as_slice().borrow();

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

    // Check that the final proof is computed by the internal recursive prover, i.e.
    // that internal_flag is 2.
    if internal_flag != F::TWO {
        return Err(VerifyStarkError::InvalidInternalFlag(internal_flag));
    }

    // Check app_dag_commit against expected_commits.
    if app_dag_commit.cached_commit != vk.baseline.app_dag_commit.cached_commit {
        return Err(VerifyStarkError::AppDagCachedCommitMismatch {
            expected: vk.baseline.app_dag_commit.cached_commit,
            actual: app_dag_commit.cached_commit,
        });
    }
    if app_dag_commit.vk_pre_hash != vk.baseline.app_dag_commit.vk_pre_hash {
        return Err(VerifyStarkError::AppDagPreHashMismatch {
            expected: vk.baseline.app_dag_commit.vk_pre_hash,
            actual: app_dag_commit.vk_pre_hash,
        });
    }

    // Check leaf_dag_commit against expected_commits.
    if leaf_dag_commit.cached_commit != vk.baseline.leaf_dag_commit.cached_commit {
        return Err(VerifyStarkError::LeafDagCachedCommitMismatch {
            expected: vk.baseline.leaf_dag_commit.cached_commit,
            actual: leaf_dag_commit.cached_commit,
        });
    }
    if leaf_dag_commit.vk_pre_hash != vk.baseline.leaf_dag_commit.vk_pre_hash {
        return Err(VerifyStarkError::LeafDagPreHashMismatch {
            expected: vk.baseline.leaf_dag_commit.vk_pre_hash,
            actual: leaf_dag_commit.vk_pre_hash,
        });
    }

    // Check internal_for_leaf_dag_commit against expected_commits.
    if internal_for_leaf_dag_commit.cached_commit
        != vk.baseline.internal_for_leaf_dag_commit.cached_commit
    {
        return Err(VerifyStarkError::InternalForLeafDagCachedCommitMismatch {
            expected: vk.baseline.internal_for_leaf_dag_commit.cached_commit,
            actual: internal_for_leaf_dag_commit.cached_commit,
        });
    }
    if internal_for_leaf_dag_commit.vk_pre_hash
        != vk.baseline.internal_for_leaf_dag_commit.vk_pre_hash
    {
        return Err(VerifyStarkError::InternalForLeafDagPreHashMismatch {
            expected: vk.baseline.internal_for_leaf_dag_commit.vk_pre_hash,
            actual: internal_for_leaf_dag_commit.vk_pre_hash,
        });
    }

    // Check that recursion_flag is 2, i.e. that the penultimate layer is internal
    // recursive.
    if recursion_flag != F::TWO {
        return Err(VerifyStarkError::InvalidRecursionFlag(recursion_flag));
    }

    // Check internal_recursive_dag_commit against expected_commits.
    if internal_recursive_dag_commit.cached_commit
        != vk.baseline.internal_recursive_dag_commit.cached_commit
    {
        return Err(VerifyStarkError::InternalRecursiveDagCachedCommitMismatch {
            expected: vk.baseline.internal_recursive_dag_commit.cached_commit,
            actual: internal_recursive_dag_commit.cached_commit,
        });
    }
    if internal_recursive_dag_commit.vk_pre_hash
        != vk.baseline.internal_recursive_dag_commit.vk_pre_hash
    {
        return Err(VerifyStarkError::InternalRecursiveDagPreHashMismatch {
            expected: vk.baseline.internal_recursive_dag_commit.vk_pre_hash,
            actual: internal_recursive_dag_commit.vk_pre_hash,
        });
    }

    // Deferral verification
    if let Some(expected_def_vk_commit) = vk.baseline.expected_def_vk_commit {
        let &VerifierDefPvs {
            deferral_flag,
            def_hook_vk_commit,
        } = verifier_def_pvs_slice.borrow();

        let flag = deferral_flag.as_canonical_u32();
        if flag != 0 && flag != 2 {
            return Err(VerifyStarkError::InvalidDeferralFlag(deferral_flag));
        }
        if flag == 0 {
            return Err(VerifyStarkError::DeferralFlagNotSet);
        }

        if def_hook_vk_commit != expected_def_vk_commit {
            return Err(VerifyStarkError::DefHookVkCommitMismatch {
                expected: expected_def_vk_commit,
                actual: def_hook_vk_commit,
            });
        }

        let deferral_merkle_proofs = proof
            .deferral_merkle_proofs
            .as_ref()
            .ok_or(VerifyStarkError::MissingDeferralMerkleProofs)?;

        let &DeferralPvs {
            initial_acc_hash,
            final_acc_hash,
            depth,
        } = proof.inner.public_values[DEF_PVS_AIR_ID]
            .as_slice()
            .borrow();

        deferral_merkle_proofs.verify(
            vk.baseline.memory_dimensions,
            initial_root,
            final_root,
            initial_acc_hash,
            final_acc_hash,
            depth.as_canonical_u32() as usize,
        )?;
    } else if !verifier_def_pvs_slice.is_empty()
        || !proof.inner.public_values[DEF_PVS_AIR_ID].is_empty()
        || proof.deferral_merkle_proofs.is_some()
    {
        return Err(VerifyStarkError::UnexpectedDeferral);
    }

    Ok(())
}
