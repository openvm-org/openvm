use openvm_circuit::{
    arch::instructions::DEFERRAL_AS,
    system::memory::{dimensions::MemoryDimensions, merkle::MerkleTree},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, F};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;

use openvm_verify_stark_host::deferral::DeferralMerkleProofs;

/// Compute deferral merkle proofs from the initial and final memory merkle trees.
///
/// Proofs have length `overall_height()`. When `depth > 0`, the first `depth` entries
/// are zeros (skipped levels covered by the deferral subtree).
pub fn compute_deferral_merkle_proofs(
    memory_dimensions: MemoryDimensions,
    initial_merkle_tree: &MerkleTree<F, DIGEST_SIZE>,
    final_merkle_tree: &MerkleTree<F, DIGEST_SIZE>,
    depth: usize,
) -> DeferralMerkleProofs<F> {
    let initial_merkle_proof =
        deferral_merkle_proof_from_tree(memory_dimensions, initial_merkle_tree, depth);
    let final_merkle_proof =
        deferral_merkle_proof_from_tree(memory_dimensions, final_merkle_tree, depth);
    DeferralMerkleProofs {
        initial_merkle_proof,
        final_merkle_proof,
    }
}

/// Extract one side of the deferral merkle proof from a memory merkle tree.
///
/// Returns a full-length proof (`overall_height()` entries). The first `depth` entries
/// are zeros; the remaining entries are siblings from the tree.
fn deferral_merkle_proof_from_tree(
    memory_dimensions: MemoryDimensions,
    merkle_tree: &MerkleTree<F, DIGEST_SIZE>,
    depth: usize,
) -> Vec<[F; DIGEST_SIZE]> {
    let overall_height = memory_dimensions.overall_height();

    // Leaf index for DEFERRAL_AS, block_id=0 in the full tree (1-indexed).
    let leaf_idx = (1u64 << overall_height) + memory_dimensions.label_to_index((DEFERRAL_AS, 0));

    // Start at level `depth` above the leaf.
    let mut node_idx = leaf_idx >> depth;

    // Pad the first `depth` entries with zeros (skipped levels).
    let mut proof = vec![[F::ZERO; DIGEST_SIZE]; depth];

    // Collect siblings from depth up to the root.
    while node_idx > 1 {
        let sibling_idx = node_idx ^ 1;
        proof.push(merkle_tree.get_node(sibling_idx));
        node_idx >>= 1;
    }

    assert_eq!(proof.len(), overall_height);
    proof
}
