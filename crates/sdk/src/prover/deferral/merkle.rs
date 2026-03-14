use openvm_circuit::{
    arch::instructions::DEFERRAL_AS,
    system::memory::{dimensions::MemoryDimensions, merkle::MerkleTree},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, F};
use openvm_verify_stark_host::deferral::DeferralMerkleProofs;

/// Compute deferral merkle proofs from the initial and final memory merkle trees.
///
/// When `depth == 0` (unset), the full `overall_height()` path is returned.
/// When `depth > 0`, only the upper `overall_height() - depth` siblings are returned
/// (the lower `depth` levels are covered by the deferral subtree).
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
/// Walks from the DEFERRAL_AS node at level `depth` up to the root, collecting siblings.
fn deferral_merkle_proof_from_tree(
    memory_dimensions: MemoryDimensions,
    merkle_tree: &MerkleTree<F, DIGEST_SIZE>,
    depth: usize,
) -> Vec<[F; DIGEST_SIZE]> {
    let overall_height = memory_dimensions.overall_height();
    let proof_len = overall_height - depth;

    // Leaf index for DEFERRAL_AS, block_id=0 in the full tree (1-indexed).
    let leaf_idx = (1u64 << overall_height) + memory_dimensions.label_to_index((DEFERRAL_AS, 0));

    // Start at level `depth` above the leaf.
    let mut node_idx = leaf_idx >> depth;

    let mut proof = Vec::with_capacity(proof_len);
    while node_idx > 1 {
        let sibling_idx = node_idx ^ 1;
        proof.push(merkle_tree.get_node(sibling_idx));
        node_idx >>= 1;
    }

    assert_eq!(proof.len(), proof_len);
    proof
}
