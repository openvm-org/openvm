use afs_test_utils::{
    config::poseidon2::StarkConfigPoseidon2,
    utils::{run_simple_test, ProverVerifierRap},
};
use p3_keccak::KeccakF;
use p3_symmetric::{PseudoCompressionFunction, TruncatedPermutation};

use afs_chips::merkle_tree::{columns::MERKLE_TREE_DEPTH, MerkleTreeChip};

fn generate_digests(leaf_hashes: Vec<[u8; 32]>) -> Vec<Vec<[u8; 32]>> {
    let keccak = TruncatedPermutation::new(KeccakF {});
    let mut digests = vec![leaf_hashes];

    while let Some(last_level) = digests.last().cloned() {
        if last_level.len() == 1 {
            break;
        }

        let next_level = last_level
            .chunks_exact(2)
            .map(|chunk| keccak.compress([chunk[0], chunk[1]]))
            .collect();

        digests.push(next_level);
    }

    digests
}

#[test]
fn test_merkle_tree_prove() {
    let leaf_hashes: Vec<[u8; 32]> = (0..2u64.pow(MERKLE_TREE_DEPTH as u32))
        .map(|_| [0; 32])
        .collect();

    let digests = generate_digests(leaf_hashes);

    let leaf_index = 0;
    let leaf = digests[0][leaf_index];

    let siblings = (0..MERKLE_TREE_DEPTH)
        .map(|i| digests[i][(leaf_index >> i) ^ 1])
        .collect::<Vec<[u8; 32]>>()
        .try_into()
        .unwrap();

    let merkle_tree_air = MerkleTreeChip {
        leaves: vec![leaf],
        leaf_indices: vec![leaf_index],
        siblings: vec![siblings],
    };

    let trace = merkle_tree_air.generate_trace();

    let chips: Vec<&dyn ProverVerifierRap<StarkConfigPoseidon2>> = vec![&merkle_tree_air];
    let traces = vec![trace];

    run_simple_test(chips, traces).expect("Verification failed");
}
