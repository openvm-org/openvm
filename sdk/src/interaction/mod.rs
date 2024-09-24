use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
    rap::AnyRap,
    verifier::{MultiTraceStarkVerifier, VerificationError},
};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;

use crate::config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config};

pub mod dummy_interaction_air;

type Val = BabyBear;

pub fn verify_interactions(
    traces: Vec<RowMajorMatrix<Val>>,
    airs: Vec<&dyn AnyRap<BabyBearPoseidon2Config>>,
    pis: Vec<Vec<Val>>,
) -> Result<(), VerificationError> {
    let log_trace_degree = 3;
    let perm = config::baby_bear_poseidon2::random_perm();
    let config = config::baby_bear_poseidon2::default_config(&perm, log_trace_degree);

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
    for (air, pis) in airs.iter().zip_eq(&pis) {
        keygen_builder.add_air(*air, pis.len());
    }
    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let prover = MultiTraceStarkProver::new(&config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    for trace in traces {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&vk, airs.clone());

    let mut challenger = config::baby_bear_poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pis);

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::baby_bear_poseidon2::Challenger::new(perm.clone());
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier.verify(&mut challenger, &vk, &proof)
}
