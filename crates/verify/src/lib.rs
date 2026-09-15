use std::{
    borrow::Borrow,
    io::{self, Read},
};

use eyre::Result;
use openvm_circuit::{
    arch::{hasher::poseidon2::vm_poseidon2_hasher, ExitCode},
    system::memory::merkle::public_values::UserPublicValuesProof,
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
        CONSTRAINT_EVAL_CACHED_INDEX, DEF_PVS_AIR_ID, MAX_RECURSION_DEPTH, VERIFIER_PVS_AIR_ID,
        VM_PVS_AIR_ID,
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
    let proof = decode_zstd(encoded_proof)?;
    verify_vm_stark_proof_decoded(vk, &proof)
}

fn decode_zstd<T: Decode>(encoded: &[u8]) -> io::Result<T> {
    let mut decoder = zstd::Decoder::new(encoded)?;
    decode_exact(&mut decoder)
}

fn decode_exact<T: Decode>(reader: &mut impl Read) -> io::Result<T> {
    let value = T::decode(reader)?;

    // Preserve `Decode::decode_from_bytes`'s canonical-encoding check without first collecting the
    // entire decompressed stream. In particular, malformed compressed inputs should be rejected as
    // soon as proof decoding fails instead of being fully inflated in memory.
    if reader.read(&mut [0])? != 0 {
        return Err(io::Error::other("trailing bytes after decoded value"));
    }

    Ok(value)
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
        recursion_depth,
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

    // Check that user_pvs_proof has the correct number of public values.
    if proof.user_pvs_proof.public_values.len() != vk.baseline.num_user_pvs {
        return Err(VerifyStarkError::UserPvsLengthMismatch {
            expected: vk.baseline.num_user_pvs,
            actual: proof.user_pvs_proof.public_values.len(),
        });
    }

    // Check the executable's program commitment and initial state directly.
    if program_commit != vk.baseline.program_commit {
        return Err(VerifyStarkError::ProgramCommitMismatch {
            expected: vk.baseline.program_commit,
            actual: program_commit,
        });
    }
    if initial_root != vk.baseline.initial_state {
        return Err(VerifyStarkError::InitialStateMismatch {
            expected: vk.baseline.initial_state,
            actual: initial_root,
        });
    }
    if initial_pc != vk.baseline.initial_pc {
        return Err(VerifyStarkError::InitialPcMismatch {
            expected: vk.baseline.initial_pc,
            actual: initial_pc,
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

    // Check that recursion_depth is within [1, MAX_RECURSION_DEPTH]. If
    // recursion_depth == 1 then the penultimate layer is internal-for-leaf,
    // else it is internal-recursive.
    let recursion_depth_u32 = recursion_depth.as_canonical_u32();
    if recursion_depth_u32 == 0 || recursion_depth_u32 > MAX_RECURSION_DEPTH {
        return Err(VerifyStarkError::InvalidRecursionDepth {
            actual: recursion_depth,
            max: MAX_RECURSION_DEPTH,
        });
    }

    // Check that internal_recursive_vk_commit is unset if recursion_depth == 1,
    // and against expected_commits otherwise.
    if recursion_depth == F::ONE {
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
    } else {
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
            node_idx,
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
        } else {
            return Err(VerifyStarkError::InvalidDeferralFlag(deferral_flag));
        }

        if node_idx != F::ZERO {
            return Err(VerifyStarkError::DefNodeIdxNonZero { actual: node_idx });
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

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};

    use openvm_stark_backend::codec::Encode;

    use super::{decode_exact, decode_zstd, VmStarkProof};

    struct CountingReader<R> {
        inner: R,
        bytes_read: usize,
    }

    impl<R: Read> Read for CountingReader<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let read = self.inner.read(buf)?;
            self.bytes_read += read;
            Ok(read)
        }
    }

    #[test]
    fn decode_zstd_accepts_exactly_one_value() {
        let encoded = 42u32.encode_to_vec().unwrap();
        let compressed = zstd::encode_all(encoded.as_slice(), 0).unwrap();

        assert_eq!(decode_zstd::<u32>(&compressed).unwrap(), 42);
    }

    #[test]
    fn decode_zstd_rejects_trailing_decompressed_bytes() {
        let mut encoded = 42u32.encode_to_vec().unwrap();
        encoded.push(0);
        let compressed = zstd::encode_all(encoded.as_slice(), 0).unwrap();

        let err = decode_zstd::<u32>(&compressed).unwrap_err();
        assert_eq!(err.to_string(), "trailing bytes after decoded value");
    }

    #[test]
    fn malformed_proof_stops_before_expanding_zstd_tail() {
        const EXPANDED_SIZE: usize = 512 * 1024 * 1024;
        static CHUNK: [u8; 64 * 1024] = [0; 64 * 1024];

        // A zero codec version makes this an invalid proof after its first four decompressed bytes.
        // Stream the input into the encoder so constructing the test case itself does not allocate
        // the 512 MiB decompressed payload.
        let mut encoder = zstd::Encoder::new(Vec::new(), 0).unwrap();
        for _ in 0..EXPANDED_SIZE / CHUNK.len() {
            encoder.write_all(&CHUNK).unwrap();
        }
        let compressed = encoder.finish().unwrap();
        assert!(compressed.len() < 64 * 1024);

        let decoder = zstd::Decoder::new(compressed.as_slice()).unwrap();
        let mut reader = CountingReader {
            inner: decoder,
            bytes_read: 0,
        };
        let err = decode_exact::<VmStarkProof>(&mut reader).unwrap_err();

        assert!(err.to_string().contains("CODEC_VERSION mismatch"));
        assert_eq!(reader.bytes_read, size_of::<u32>());
    }
}
