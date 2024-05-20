use afs_stark_backend::keygen::types::SymbolicRap;
use afs_stark_backend::keygen::MultiStarkKeygenBuilder;
use afs_stark_backend::prover::trace::TraceCommitmentBuilder;
use afs_stark_backend::prover::types::ProverRap;
use afs_stark_backend::prover::MultiTraceStarkProver;
use afs_stark_backend::verifier::types::VerifierRap;
use afs_stark_backend::verifier::{MultiTraceStarkVerifier, VerificationError};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

use crate::config::{self, poseidon2::StarkConfigPoseidon2};

use crate::utils::ProverVerifierRap;

pub mod dummy_interaction_air;

type Val = BabyBear;

pub fn verify_interactions(
    traces: Vec<RowMajorMatrix<Val>>,
    airs: Vec<&dyn ProverVerifierRap<StarkConfigPoseidon2>>,
    pis: Vec<Vec<Val>>,
) -> Result<(), VerificationError> {
    let log_trace_degree = 3;
    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, log_trace_degree);

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
    for ((air, trace), pis) in airs.iter().zip_eq(&traces).zip_eq(&pis) {
        let height = trace.height();
        keygen_builder.add_air(*air as &dyn SymbolicRap<_>, height, pis.len());
    }
    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let prover = MultiTraceStarkProver::new(config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    for trace in traces {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(
        &vk,
        airs.iter().map(|&air| air as &dyn ProverRap<_>).collect(),
    );

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pis);

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier.verify(
        &mut challenger,
        vk,
        airs.iter().map(|&air| air as &dyn VerifierRap<_>).collect(),
        proof,
        &pis,
    )
}
