use openvm_circuit::{
    arch::instructions::DEFERRAL_AS, system::memory::dimensions::MemoryDimensions,
};
use openvm_stark_backend::codec::{DecodableConfig, EncodableConfig};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config as SC, DIGEST_SIZE, F,
};
use p3_field::PrimeCharacteristicRing;

use crate::error::VerifyStarkError;

/// Deferral Merkle proofs connecting `initial_acc_hash` and `final_acc_hash` to the
/// initial and final memory roots. Both proofs have length `overall_height()`.
/// When `depth > 0`, the first `depth` entries are zeros (skipped levels covered by
/// the deferral subtree).
#[derive(Clone, Debug)]
pub struct DeferralMerkleProofs<F> {
    pub initial_merkle_proof: Vec<[F; DIGEST_SIZE]>,
    pub final_merkle_proof: Vec<[F; DIGEST_SIZE]>,
}

impl DeferralMerkleProofs<F> {
    pub fn encode<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        SC::encode_digest_slice(&self.initial_merkle_proof, writer)?;
        SC::encode_digest_slice(&self.final_merkle_proof, writer)?;
        Ok(())
    }

    pub fn decode<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let initial_merkle_proof = SC::decode_digest_vec(reader)?;
        let final_merkle_proof = SC::decode_digest_vec(reader)?;
        Ok(Self {
            initial_merkle_proof,
            final_merkle_proof,
        })
    }

    /// Verify that `initial_acc_hash` and `final_acc_hash` are committed into `initial_root`
    /// and `final_root` respectively via the deferral address space Merkle path.
    ///
    /// `depth` is the deferral subtree depth: when `depth == 0` the acc hashes are "unset"
    /// (the leaf is `compress(acc_hash, [0; DIGEST_SIZE])`), otherwise the leaf is `acc_hash`
    /// directly and the first `depth` proof siblings are skipped.
    pub fn verify(
        &self,
        memory_dimensions: MemoryDimensions,
        initial_root: [F; DIGEST_SIZE],
        final_root: [F; DIGEST_SIZE],
        initial_acc_hash: [F; DIGEST_SIZE],
        final_acc_hash: [F; DIGEST_SIZE],
        depth: usize,
    ) -> Result<(), VerifyStarkError> {
        let overall_height = memory_dimensions.overall_height();
        if self.initial_merkle_proof.len() != overall_height {
            return Err(VerifyStarkError::DeferralMerkleProofLengthMismatch {
                expected: overall_height,
                actual: self.initial_merkle_proof.len(),
            });
        }
        if self.final_merkle_proof.len() != overall_height {
            return Err(VerifyStarkError::DeferralMerkleProofLengthMismatch {
                expected: overall_height,
                actual: self.final_merkle_proof.len(),
            });
        }

        let is_unset = depth == 0;
        let skip_depth = if is_unset { 0 } else { depth };
        let idx_prefix =
            usize::try_from(memory_dimensions.label_to_index((DEFERRAL_AS, 0))).unwrap();

        // When unset, the leaf is compress(acc_hash, zeros); otherwise it's acc_hash directly.
        let initial_leaf = if is_unset {
            poseidon2_compress_with_capacity(initial_acc_hash, [F::ZERO; DIGEST_SIZE]).0
        } else {
            initial_acc_hash
        };
        let final_leaf = if is_unset {
            poseidon2_compress_with_capacity(final_acc_hash, [F::ZERO; DIGEST_SIZE]).0
        } else {
            final_acc_hash
        };

        let computed_initial_root = merkle_path_root(
            initial_leaf,
            &self.initial_merkle_proof,
            idx_prefix,
            skip_depth,
        );
        if computed_initial_root != initial_root {
            return Err(VerifyStarkError::DeferralInitialRootMismatch {
                expected: initial_root,
                actual: computed_initial_root,
            });
        }

        let computed_final_root =
            merkle_path_root(final_leaf, &self.final_merkle_proof, idx_prefix, skip_depth);
        if computed_final_root != final_root {
            return Err(VerifyStarkError::DeferralFinalRootMismatch {
                expected: final_root,
                actual: computed_final_root,
            });
        }

        Ok(())
    }
}

/// Walk a Merkle path from `leaf` to root, skipping the first `skip_depth` levels.
/// `idx_prefix` determines which side the node is on at each level (bit i => right child).
fn merkle_path_root(
    leaf: [F; DIGEST_SIZE],
    proof: &[[F; DIGEST_SIZE]],
    idx_prefix: usize,
    skip_depth: usize,
) -> [F; DIGEST_SIZE] {
    let mut node = leaf;
    for (i, sibling) in proof.iter().enumerate() {
        if i < skip_depth {
            continue;
        }
        let is_right = (idx_prefix >> i) & 1 == 1;
        node = if is_right {
            poseidon2_compress_with_capacity(*sibling, node).0
        } else {
            poseidon2_compress_with_capacity(node, *sibling).0
        };
    }
    node
}
