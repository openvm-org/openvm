use afs_middleware::{
    prover::{trace::TraceCommitter, types::ProvenMultiMatrixAirTrace, PartitionProver},
    setup::PartitionSetup,
    verifier::PartitionVerifier,
};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_uni_stark::StarkGenericConfig;
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

use crate::config::poseidon2::StarkConfigPoseidon2;

mod config;
mod fib_air;
mod fib_selector_air;

#[test]
fn test_single_fib_stark() {
    use fib_air::air::FibonacciAir;
    use fib_air::trace::generate_trace_rows;

    // Set up tracing:
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    let _ = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .try_init();

    let log_trace_degree = 3;
    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, log_trace_degree);

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let pis = [a, b, get_fib_number(n)].map(BabyBear::from_canonical_u32);

    let air = FibonacciAir {};

    let prep_trace = air.preprocessed_trace();
    let setup = PartitionSetup::new(&config);
    let (pk, vk) = setup.setup(vec![prep_trace]);

    let trace = generate_trace_rows::<Val>(a, b, n);
    let trace_committer = TraceCommitter::<StarkConfigPoseidon2>::new(config.pcs());
    let proven_trace = trace_committer.commit(vec![trace]);
    let proven = ProvenMultiMatrixAirTrace {
        trace_data: &proven_trace,
        airs: vec![&air],
    };

    let prover = PartitionProver::new(config);
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, vec![proven], &pis);

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = PartitionVerifier::new(prover.config);
    verifier
        .verify(&mut challenger, vk, vec![&air], proof, &pis)
        .expect("Verification failed");
}

#[test]
fn test_single_fib_selector_stark() {
    use fib_selector_air::air::FibonacciSelectorAir;
    use fib_selector_air::trace::generate_trace_rows;

    // Set up tracing:
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    let _ = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .try_init();

    let log_trace_degree = 3;
    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, log_trace_degree);

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let sels: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let pis = [a, b, get_conditional_fib_number(&sels)].map(BabyBear::from_canonical_u32);

    let air = FibonacciSelectorAir { sels };

    let prep_trace = air.preprocessed_trace();
    let setup = PartitionSetup::new(&config);
    let (pk, vk) = setup.setup(vec![prep_trace]);

    let trace = generate_trace_rows::<Val>(a, b, &air.sels);
    let trace_committer = TraceCommitter::<StarkConfigPoseidon2>::new(config.pcs());
    let proven_trace = trace_committer.commit(vec![trace]);
    let proven = ProvenMultiMatrixAirTrace {
        trace_data: &proven_trace,
        airs: vec![&air],
    };

    let prover = PartitionProver::new(config);
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, vec![proven], &pis);

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = PartitionVerifier::new(prover.config);
    verifier
        .verify(&mut challenger, vk, vec![&air], proof, &pis)
        .expect("Verification failed");
}

#[test]
fn test_double_fib_starks() {
    use fib_air::air::FibonacciAir;
    use fib_selector_air::air::FibonacciSelectorAir;

    // Set up tracing:
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    let _ = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .try_init();

    let log_trace_degree = 3;
    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, log_trace_degree);

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let sels: Vec<bool> = (0..n).map(|_| true).collect(); // All 1s
    let pis = [a, b, get_fib_number(n)].map(BabyBear::from_canonical_u32);

    let air1 = FibonacciAir {};
    let air2 = FibonacciSelectorAir { sels };

    let prep_trace1 = air1.preprocessed_trace();
    let prep_trace2 = air2.preprocessed_trace();
    let setup = PartitionSetup::new(&config);
    let (pk, vk) = setup.setup(vec![prep_trace1, prep_trace2]);

    let trace1 = fib_air::trace::generate_trace_rows::<Val>(a, b, n);
    let trace2 = fib_selector_air::trace::generate_trace_rows::<Val>(a, b, &air2.sels);
    let trace_committer = TraceCommitter::<StarkConfigPoseidon2>::new(config.pcs());
    let proven_trace = trace_committer.commit(vec![trace1, trace2]);
    let proven = ProvenMultiMatrixAirTrace {
        trace_data: &proven_trace,
        airs: vec![&air1, &air2],
    };

    let prover = PartitionProver::new(config);
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, vec![proven], &pis);

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = PartitionVerifier::new(prover.config);
    verifier
        .verify(&mut challenger, vk, vec![&air1, &air2], proof, &pis)
        .expect("Verification failed");
}

fn get_fib_number(n: usize) -> u32 {
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n - 1 {
        let c = a + b;
        a = b;
        b = c;
    }
    b
}

fn get_conditional_fib_number(sels: &[bool]) -> u32 {
    let mut a = 0;
    let mut b = 1;
    for &s in sels[0..sels.len() - 1].iter() {
        if s {
            let c = a + b;
            a = b;
            b = c;
        }
    }
    b
}
