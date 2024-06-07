use afs_stark_backend::rap::AnyRap;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use itertools::Itertools;
use p3_keccak::Keccak256Hash;
use p3_symmetric::{CompressionFunction, CompressionFunctionFromHasher};
use rand::{rngs::StdRng, Rng, SeedableRng};

use afs_chips::{
    keccak_permute::KeccakPermuteChip,
    keccak_sponge::{
        columns::{KECCAK_RATE_BYTES, KECCAK_WIDTH_BYTES},
        KeccakSpongeAir, KeccakSpongeOp,
    },
    merkle_proof::{MerkleProofChip, MerkleProofOp},
};

fn generate_digests<Compress: CompressionFunction<[u8; 32], 2>>(
    leaf_hashes: Vec<[u8; 32]>,
    hasher: &Compress,
) -> Vec<Vec<[u8; 32]>> {
    let mut digests = vec![leaf_hashes];

    while let Some(last_level) = digests.last().cloned() {
        if last_level.len() == 1 {
            break;
        }

        let next_level = last_level
            .chunks_exact(2)
            .map(|chunk| hasher.compress([chunk[0], chunk[1]]))
            .collect();

        digests.push(next_level);
    }

    digests
}

#[test]
fn test_merkle_proof_prove() {
    const RANDOM_SEED: u64 = 0;
    let mut seeded_rng = StdRng::seed_from_u64(RANDOM_SEED);

    const HEIGHT: usize = 8;
    const NUM_LEAVES: usize = 1 << HEIGHT;
    const DIGEST_WIDTH: usize = 32;

    let hasher = CompressionFunctionFromHasher::new(Keccak256Hash);

    let leaf_hashes = (0..NUM_LEAVES).map(|_| seeded_rng.gen()).collect_vec();
    let digests = generate_digests(leaf_hashes, &hasher);

    let leaf_index = seeded_rng.gen_range(0..NUM_LEAVES);
    let leaf_hash = digests[0][leaf_index];

    let siblings: [[u8; 32]; HEIGHT] = (0..HEIGHT)
        .map(|i| digests[i][(leaf_index >> i) ^ 1])
        .collect::<Vec<[u8; 32]>>()
        .try_into()
        .unwrap();
    let op = MerkleProofOp {
        leaf_index,
        leaf_hash,
        siblings,
    };

    let keccak_inputs = (0..HEIGHT)
        .map(|i| {
            let index = leaf_index >> i;
            let parity = index & 1;
            let (left, right) = if parity == 0 {
                (digests[i][index], digests[i][index ^ 1])
            } else {
                (digests[i][index ^ 1], digests[i][index])
            };
            let input = left.into_iter().chain(right).collect_vec();
            KeccakSpongeOp {
                timestamp: 0,
                addr: 0,
                input,
            }
        })
        .collect_vec();
    // let mut xor_inputs = Vec::new();
    let permute_inputs = keccak_inputs
        .iter()
        .map(|op| {
            let mut bytes_input = [0; KECCAK_WIDTH_BYTES];
            bytes_input[0..2 * DIGEST_WIDTH].copy_from_slice(&op.input);
            bytes_input[2 * DIGEST_WIDTH] = 1;
            bytes_input[KECCAK_RATE_BYTES - 1] |= 0b10000000;

            // bytes_input[0..KECCAK_RATE_BYTES].chunks(4).for_each(|val| {
            //     xor_inputs.push((val.try_into().unwrap(), [0; 4]));
            // });
            let input = bytes_input
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect_vec()
                .try_into()
                .unwrap();
            input
        })
        .collect_vec();

    let merkle_proof_air = MerkleProofChip {
        bus_hash_input: 0,
        bus_hash_output: 1,
    };
    let keccak_sponge_air = KeccakSpongeAir {
        bus_input: 0,
        bus_output: 1,

        bus_permute_input: 2,
        bus_permute_output: 3,

        bus_xor_input: 4,
        bus_xor_output: 5,
    };
    let keccak_permute_air = KeccakPermuteChip {
        bus_input: 2,
        bus_output: 3,
    };

    let merkle_proof_trace = merkle_proof_air.generate_trace(vec![op], &hasher);
    let keccak_sponge_trace = keccak_sponge_air.generate_trace(keccak_inputs);
    let keccak_permute_trace = keccak_permute_air.generate_trace(permute_inputs);

    let chips = vec![
        &merkle_proof_air as &dyn AnyRap<_>,
        &keccak_sponge_air as &dyn AnyRap<_>,
        &keccak_permute_air as &dyn AnyRap<_>,
    ];
    let traces = vec![
        merkle_proof_trace,
        keccak_sponge_trace,
        keccak_permute_trace,
    ];

    run_simple_test_no_pis(chips, traces).expect("Verification failed");
}
