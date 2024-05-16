use afs_middleware::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, types::ProverRap, MultiTraceStarkProver},
    verifier::{types::VerifierRap, MultiTraceStarkVerifier},
};
use p3_keccak::KeccakF;
use p3_symmetric::{PseudoCompressionFunction, TruncatedPermutation};
use p3_uni_stark::StarkGenericConfig;

mod config;

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

    let log_trace_degree_max: usize = MERKLE_TREE_DEPTH;

    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, log_trace_degree_max);

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
    keygen_builder.add_air(&merkle_tree_air, MERKLE_TREE_DEPTH, 0);

    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let prover = MultiTraceStarkProver::new(config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.config.pcs());
    let trace = merkle_tree_air.generate_trace();
    trace_builder.load_trace(trace);
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&vk, vec![&merkle_tree_air as &dyn ProverRap<_>]);

    let pis = vec![vec![]; vk.per_air.len()];

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pis);

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier
        .verify(
            &mut challenger,
            vk,
            vec![&merkle_tree_air as &dyn VerifierRap<_>],
            proof,
            &pis,
        )
        .expect("Verification failed");
}
