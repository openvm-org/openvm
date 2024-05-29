use afs_stark_backend::rap::AnyRap;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_keccak::KeccakF;
use p3_symmetric::{PseudoCompressionFunction, TruncatedPermutation};

use afs_chips::{
    keccak_permute::KeccakPermuteChip,
    merkle_tree::{columns::MERKLE_TREE_DEPTH, MerkleTreeChip},
};

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

    let height = digests.len() - 1;
    let keccak_inputs = (0..height)
        .map(|i| {
            let index = leaf_index >> i;
            let parity = index & 1;
            let (left, right) = if parity == 0 {
                (digests[i][index], digests[i][index ^ 1])
            } else {
                (digests[i][index ^ 1], digests[i][index])
            };
            let mut input = [0; 25];
            input[0..4].copy_from_slice(
                left.chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            input[4..8].copy_from_slice(
                right
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            input
        })
        .collect::<Vec<_>>();

    let merkle_tree_air = MerkleTreeChip {
        bus_hash_input: 0,
        bus_hash_output: 1,
        leaves: vec![leaf],
        leaf_indices: vec![leaf_index],
        siblings: vec![siblings],
    };

    let keccak_permute_air = KeccakPermuteChip {
        bus_input: 0,
        bus_output: 1, // Not needed, change to option
        bus_output_digest: 1,
        inputs_digest: keccak_inputs,
        inputs_sponge: vec![],
    };

    let merkle_tree_trace = merkle_tree_air.generate_trace();
    let keccak_permute_trace = keccak_permute_air.generate_trace();

    let chips = vec![
        &merkle_tree_air as &dyn AnyRap<_>,
        &keccak_permute_air as &dyn AnyRap<_>,
    ];
    let traces = vec![merkle_tree_trace, keccak_permute_trace];

    run_simple_test_no_pis(chips, traces).expect("Verification failed");
}
