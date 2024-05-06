use afs_middleware::{
    prover::{trace::TraceCommitter, types::ProvenMultiMatrixAirTrace, PartitionProver},
    verifier::PartitionVerifier,
};
use fib_air::trace::generate_trace_rows;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_uni_stark::StarkGenericConfig;
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

use crate::config::poseidon2::StarkConfigPoseidon2;
use crate::fib_air::air::FibonacciAir;

mod config;
mod fib_air;

#[test]
fn test_single_fib_stark() {
    // Set up tracing:
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    let log_trace_degree = 3;
    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, log_trace_degree);

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let pis = [a, b, get_fib_number(n)].map(BabyBear::from_canonical_u32);

    let trace = generate_trace_rows::<Val>(a, b, n);
    let trace_committer = TraceCommitter::<StarkConfigPoseidon2>::new(config.pcs());
    let proven_trace = trace_committer.commit(vec![trace]);
    let proven = ProvenMultiMatrixAirTrace {
        trace_data: &proven_trace,
        airs: vec![&FibonacciAir],
    };

    let prover = PartitionProver::new(config);
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, vec![proven], &pis);

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = PartitionVerifier::new(prover.config);
    verifier
        .verify(&mut challenger, vec![&FibonacciAir], proof, &pis)
        .expect("Verification failed");
}

fn get_fib_number(n: usize) -> u32 {
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n {
        let c = a + b;
        a = b;
        b = c;
    }
    a
}
