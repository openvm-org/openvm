use afs_middleware::{
    prover::{trace::TraceCommitter, types::ProvenMultiMatrixAirTrace, PartitionProver},
    setup::PartitionSetup,
    verifier::{PartitionVerifier, VerificationError},
};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::StarkGenericConfig;
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

use crate::{
    config::{self, poseidon2::StarkConfigPoseidon2},
    fib_selector_air, get_conditional_fib_number, sender_air,
};

type Val = BabyBear;

fn verify_fib_interaction(
    sender_trace_gen: fn(a: u32, b: u32, n: usize, sels: Vec<bool>) -> RowMajorMatrix<Val>,
) -> Result<(), VerificationError> {
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

    let sels: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let fib_res = get_conditional_fib_number(&sels);
    let pis = [a, b, fib_res].map(Val::from_canonical_u32);

    let air = FibonacciSelectorAir {
        sels: sels.clone(),
        enable_interactions: true,
    };

    let sender_air = sender_air::SenderAir {};

    let prep_trace = air.preprocessed_trace();
    let sender_prep_trace = sender_air.preprocessed_trace();
    let setup = PartitionSetup::new(&config);
    let (pk, vk) = setup.setup(vec![prep_trace, sender_prep_trace]);

    let trace = generate_trace_rows::<Val>(a, b, &air.sels);
    let sender_trace = sender_trace_gen(a, b, n, sels);

    let trace_committer = TraceCommitter::<StarkConfigPoseidon2>::new(config.pcs());
    let proven_trace = trace_committer.commit(vec![trace, sender_trace]);
    let proven = ProvenMultiMatrixAirTrace {
        trace_data: &proven_trace,
        airs: vec![&air, &sender_air],
    };

    let prover = PartitionProver::new(config);
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, vec![proven], &pis);

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = PartitionVerifier::new(prover.config);
    verifier.verify(&mut challenger, vk, vec![&air, &sender_air], proof, &pis)
}

#[test]
fn test_interaction_stark_happy_path() {
    verify_fib_interaction(|a, b, _n, sels| {
        let mut curr_a = a;
        let mut curr_b = b;
        let mut vals = vec![];
        for sel in sels {
            vals.push(Val::from_bool(sel));
            if sel {
                let c = curr_a + curr_b;
                curr_a = curr_b;
                curr_b = c;
            }
            vals.push(Val::from_canonical_u32(curr_b));
        }
        RowMajorMatrix::new(vals, 2)
    })
    .expect("Verification failed");
}

#[test]
fn test_interaction_stark_neg() {
    let res = verify_fib_interaction(|a, b, _n, sels| {
        let mut curr_a = a;
        let mut curr_b = b;
        let mut vals = vec![];
        for sel in sels {
            vals.push(Val::from_bool(sel));
            if sel {
                let c = curr_a + curr_b;
                curr_a = curr_b;
                curr_b = c;
            }
            vals.push(Val::from_canonical_u32(curr_b + 1));
        }
        RowMajorMatrix::new(vals, 2)
    });
    assert_eq!(res, Err(VerificationError::NonZeroCumulativeSum));
}
