#![feature(trait_upcasting)]
#![allow(incomplete_features)]

use afs_stark_backend::keygen::MultiStarkKeygenBuilder;
use afs_stark_backend::prover::trace::TraceCommitmentBuilder;
use afs_stark_backend::prover::MultiTraceStarkProver;
use afs_stark_backend::verifier::MultiTraceStarkVerifier;
/// Test utils
use afs_test_utils::{config, utils};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_uni_stark::StarkGenericConfig;

mod cached_lookup;
mod fib_air;
mod fib_selector_air;
mod fib_triples_air;
pub mod interaction;
mod partitioned_sum_air;

#[test]
fn test_single_fib_stark() {
    use fib_air::air::FibonacciAir;
    use fib_air::trace::generate_trace_rows;

    let log_trace_degree = 3;
    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, log_trace_degree);

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let pis = [a, b, get_fib_number(n)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();
    let air = FibonacciAir;

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
    keygen_builder.add_air(&air, n, pis.len());
    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let trace = generate_trace_rows::<Val>(a, b, n);

    let prover = MultiTraceStarkProver::new(config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.config.pcs());
    trace_builder.load_trace(trace);
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&vk, vec![&air]);

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &[pis.clone()]);

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier
        .verify(&mut challenger, vk, vec![&air], proof, &[pis])
        .expect("Verification failed");
}

#[test]
fn test_single_fib_triples_stark() {
    use fib_triples_air::air::FibonacciAir;
    use fib_triples_air::trace::generate_trace_rows;

    let log_trace_degree = 3;
    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, log_trace_degree);

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let pis = [a, b, get_fib_number(n + 1)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();

    let air = FibonacciAir;

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
    keygen_builder.add_air(&air, n, pis.len());
    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let trace = generate_trace_rows::<Val>(a, b, n);

    let prover = MultiTraceStarkProver::new(config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.config.pcs());
    trace_builder.load_trace(trace);
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&vk, vec![&air]);

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &[pis.clone()]);

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier
        .verify(&mut challenger, vk, vec![&air], proof, &[pis])
        .expect("Verification failed");
}

#[test]
fn test_single_fib_selector_stark() {
    use fib_selector_air::air::FibonacciSelectorAir;
    use fib_selector_air::trace::generate_trace_rows;

    let log_trace_degree = 3;
    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, log_trace_degree);

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let sels: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let pis = [a, b, get_conditional_fib_number(&sels)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();

    let air = FibonacciSelectorAir::new(sels, false);

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
    keygen_builder.add_air(&air, n, pis.len());
    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let trace = generate_trace_rows::<Val>(a, b, air.sels());

    let prover = MultiTraceStarkProver::new(config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.config.pcs());
    trace_builder.load_trace(trace);
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&vk, vec![&air]);

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &[pis.clone()]);

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier
        .verify(&mut challenger, vk, vec![&air], proof, &[pis])
        .expect("Verification failed");
}

#[test]
fn test_double_fib_starks() {
    use fib_air::air::FibonacciAir;
    use fib_selector_air::air::FibonacciSelectorAir;

    let log_n1 = 3;
    let log_n2 = 5;
    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, log_n1.max(log_n2));

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n1 = 1usize << log_n1;
    let n2 = 1usize << log_n2;

    type Val = BabyBear;
    let sels: Vec<bool> = (0..n2).map(|i| i % 2 == 0).collect(); // Evens
    let pis1 = [a, b, get_fib_number(n1)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();
    let pis2 = [a, b, get_conditional_fib_number(&sels)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();

    let air1 = FibonacciAir;
    let air2 = FibonacciSelectorAir::new(sels, false);

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
    keygen_builder.add_air(&air1, n1, pis1.len());
    keygen_builder.add_air(&air2, n2, pis2.len());
    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let trace1 = fib_air::trace::generate_trace_rows::<Val>(a, b, n1);
    let trace2 = fib_selector_air::trace::generate_trace_rows::<Val>(a, b, air2.sels());

    let prover = MultiTraceStarkProver::new(config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.config.pcs());
    trace_builder.load_trace(trace1);
    trace_builder.load_trace(trace2);
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&vk, vec![&air1, &air2]);
    let pis_all = [pis1, pis2];

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pis_all);

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier
        .verify(&mut challenger, vk, vec![&air1, &air2], proof, &pis_all)
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
