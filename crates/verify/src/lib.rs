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
        DeferralPvs, VerifierBasePvs, VerifierDefPvs, VmPvs, CONSTRAINT_EVAL_AIR_ID,
        CONSTRAINT_EVAL_CACHED_INDEX, DEF_PVS_AIR_ID, VERIFIER_PVS_AIR_ID, VM_PVS_AIR_ID,
    },
    vk::VmStarkVerifyingKey,
};

pub mod deferral;
pub mod error;
pub mod pvs;
pub mod vk;

pub(crate) type VkCommit = pvs::VkCommit<F>;

// Final internal recursive STARK proof to be verified against the baseline
#[derive(Clone, Debug)]
pub struct VmStarkProof {
    pub inner: Proof<SC>,
    pub user_pvs_proof: UserPublicValuesProof<DIGEST_SIZE, F>,
    pub deferral_merkle_proofs: Option<DeferralMerkleProofs<F>>,
}

impl Encode for VmStarkProof {
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

impl Decode for VmStarkProof {
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
    vk: &VmStarkVerifyingKey,
    encoded_proof: &[u8],
) -> Result<(), VerifyStarkError> {
    let decompressed = zstd::decode_all(encoded_proof)?;
    verify_vm_stark_proof_decoded(vk, &VmStarkProof::decode_from_bytes(&decompressed)?)
}

/// Verifies a non-root VM STARK proof given the internal-recursive layer verifying
/// key and VM- and exe-specific baseline artifacts.
pub fn verify_vm_stark_proof_decoded(
    vk: &VmStarkVerifyingKey,
    proof: &VmStarkProof,
) -> Result<(), VerifyStarkError> {
    // Verify the STARK proof.
    let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(vk.mvk.inner.params.clone());
    engine.verify(&vk.mvk, &proof.inner)?;
    verify_vm_stark_proof_pvs(vk, proof)
}

pub fn verify_vm_stark_proof_pvs(
    vk: &VmStarkVerifyingKey,
    proof: &VmStarkProof,
) -> Result<(), VerifyStarkError> {
    let (verifier_base_pvs_slice, verifier_def_pvs_slice) = proof.inner.public_values
        [VERIFIER_PVS_AIR_ID]
        .as_slice()
        .split_at(VerifierBasePvs::<u8>::width());

    let &VerifierBasePvs::<F> {
        internal_flag,
        app_vk_commit,
        leaf_vk_commit,
        internal_for_leaf_vk_commit,
        recursion_flag,
        internal_recursive_vk_commit,
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

    // Check app_vk_commit against expected_commits.
    if app_vk_commit.cached_commit != vk.baseline.app_vk_commit.cached_commit {
        return Err(VerifyStarkError::AppVkCachedCommitMismatch {
            expected: vk.baseline.app_vk_commit.cached_commit,
            actual: app_vk_commit.cached_commit,
        });
    }
    if app_vk_commit.vk_pre_hash != vk.baseline.app_vk_commit.vk_pre_hash {
        return Err(VerifyStarkError::AppVkPreHashMismatch {
            expected: vk.baseline.app_vk_commit.vk_pre_hash,
            actual: app_vk_commit.vk_pre_hash,
        });
    }

    // Check leaf_vk_commit against expected_commits.
    if leaf_vk_commit.cached_commit != vk.baseline.leaf_vk_commit.cached_commit {
        return Err(VerifyStarkError::LeafVkCachedCommitMismatch {
            expected: vk.baseline.leaf_vk_commit.cached_commit,
            actual: leaf_vk_commit.cached_commit,
        });
    }
    if leaf_vk_commit.vk_pre_hash != vk.baseline.leaf_vk_commit.vk_pre_hash {
        return Err(VerifyStarkError::LeafVkPreHashMismatch {
            expected: vk.baseline.leaf_vk_commit.vk_pre_hash,
            actual: leaf_vk_commit.vk_pre_hash,
        });
    }

    // Check internal_for_leaf_vk_commit against expected_commits.
    if internal_for_leaf_vk_commit.cached_commit
        != vk.baseline.internal_for_leaf_vk_commit.cached_commit
    {
        return Err(VerifyStarkError::InternalForLeafVkCachedCommitMismatch {
            expected: vk.baseline.internal_for_leaf_vk_commit.cached_commit,
            actual: internal_for_leaf_vk_commit.cached_commit,
        });
    }
    if internal_for_leaf_vk_commit.vk_pre_hash
        != vk.baseline.internal_for_leaf_vk_commit.vk_pre_hash
    {
        return Err(VerifyStarkError::InternalForLeafVkPreHashMismatch {
            expected: vk.baseline.internal_for_leaf_vk_commit.vk_pre_hash,
            actual: internal_for_leaf_vk_commit.vk_pre_hash,
        });
    }

    // Check that SymbolicExpressionAir's cached trace exists and extract it.
    let proof_cached_commit =
        if let Some(trace_vdata) = proof.inner.trace_vdata[CONSTRAINT_EVAL_AIR_ID].as_ref() {
            if let Some(proof_cached_commit) = trace_vdata
                .cached_commitments
                .get(CONSTRAINT_EVAL_CACHED_INDEX)
            {
                *proof_cached_commit
            } else {
                return Err(VerifyStarkError::MissingConstraintEvalCachedTrace {
                    air_idx: CONSTRAINT_EVAL_AIR_ID,
                    cached_idx: CONSTRAINT_EVAL_CACHED_INDEX,
                });
            }
        } else {
            return Err(VerifyStarkError::MissingConstraintEvalTraceVdata {
                air_idx: CONSTRAINT_EVAL_AIR_ID,
            });
        };

    // Check that recursion_flag is 1 or 2, i.e. that the penultimate layer is
    // internal-for-leaf or internal-recursive.
    if recursion_flag != F::ONE && recursion_flag != F::TWO {
        return Err(VerifyStarkError::InvalidRecursionFlag(recursion_flag));
    }

    // Check internal_recursive_vk_commit against expected_commits if recursion_flag
    // is 2, and check it is unset if recursion_flag is 1.
    if recursion_flag == F::TWO {
        if internal_recursive_vk_commit.cached_commit
            != vk.baseline.internal_recursive_vk_commit.cached_commit
        {
            return Err(VerifyStarkError::InternalRecursiveVkCachedCommitMismatch {
                expected: vk.baseline.internal_recursive_vk_commit.cached_commit,
                actual: internal_recursive_vk_commit.cached_commit,
            });
        }
        if internal_recursive_vk_commit.vk_pre_hash
            != vk.baseline.internal_recursive_vk_commit.vk_pre_hash
        {
            return Err(VerifyStarkError::InternalRecursiveVkPreHashMismatch {
                expected: vk.baseline.internal_recursive_vk_commit.vk_pre_hash,
                actual: internal_recursive_vk_commit.vk_pre_hash,
            });
        }
        if proof_cached_commit != vk.baseline.internal_recursive_vk_commit.cached_commit {
            return Err(VerifyStarkError::ProofCachedCommitMismatch {
                expected: vk.baseline.internal_recursive_vk_commit.cached_commit,
                actual: proof_cached_commit,
            });
        }
    } else {
        if !is_unset(&internal_recursive_vk_commit.cached_commit) {
            return Err(VerifyStarkError::InternalRecursiveVkCachedCommitSet {
                actual: internal_recursive_vk_commit.cached_commit,
            });
        }
        if !is_unset(&internal_recursive_vk_commit.vk_pre_hash) {
            return Err(VerifyStarkError::InternalRecursiveVkPreHashSet {
                actual: internal_recursive_vk_commit.vk_pre_hash,
            });
        }
        if proof_cached_commit != vk.baseline.internal_for_leaf_vk_commit.cached_commit {
            return Err(VerifyStarkError::ProofCachedCommitMismatch {
                expected: vk.baseline.internal_for_leaf_vk_commit.cached_commit,
                actual: proof_cached_commit,
            });
        }
    }

    // Deferral verification
    if let Some(expected_def_hook_commit) = vk.baseline.expected_def_hook_commit {
        let &VerifierDefPvs {
            deferral_flag,
            def_hook_commit,
        } = verifier_def_pvs_slice.borrow();

        let &DeferralPvs {
            initial_acc_hash,
            final_acc_hash,
            depth,
        } = proof.inner.public_values[DEF_PVS_AIR_ID]
            .as_slice()
            .borrow();

        if deferral_flag == F::ZERO {
            if !is_unset(&def_hook_commit) {
                return Err(VerifyStarkError::DefHookCommitSet {
                    actual: def_hook_commit,
                });
            } else if !is_unset(&initial_acc_hash) {
                return Err(VerifyStarkError::DefInitialAccHashCommitSet {
                    actual: initial_acc_hash,
                });
            } else if !is_unset(&final_acc_hash) {
                return Err(VerifyStarkError::DefFinalAccHashCommitSet {
                    actual: final_acc_hash,
                });
            } else if depth != F::ZERO {
                return Err(VerifyStarkError::DefDepthSet { actual: depth });
            }
        } else if deferral_flag == F::TWO {
            if def_hook_commit != expected_def_hook_commit {
                return Err(VerifyStarkError::DefHookCommitMismatch {
                    expected: expected_def_hook_commit,
                    actual: def_hook_commit,
                });
            }
            let deferral_merkle_proofs = proof
                .deferral_merkle_proofs
                .as_ref()
                .ok_or(VerifyStarkError::MissingDeferralMerkleProofs)?;
            deferral_merkle_proofs.verify(
                vk.baseline.memory_dimensions,
                initial_root,
                final_root,
                initial_acc_hash,
                final_acc_hash,
                depth.as_canonical_u32() as usize,
            )?;
        } else {
            return Err(VerifyStarkError::InvalidDeferralFlag(deferral_flag));
        }
    } else if !verifier_def_pvs_slice.is_empty()
        || !proof.inner.public_values[DEF_PVS_AIR_ID].is_empty()
        || proof.deferral_merkle_proofs.is_some()
    {
        return Err(VerifyStarkError::UnexpectedDeferralDisabled);
    }

    Ok(())
}

fn is_unset(slice: &[F]) -> bool {
    slice.iter().all(|&f| f == F::ZERO)
}
