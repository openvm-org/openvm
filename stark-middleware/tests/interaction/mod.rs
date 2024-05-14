use afs_middleware::{
    prover::{
        trace::TraceCommitter,
        types::{ProvenMultiMatrixAirTrace, ProverRap},
        PartitionProver,
    },
    setup::PartitionSetup,
    verifier::{types::VerifierRap, PartitionVerifier, VerificationError},
};
use itertools::Itertools;
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
    fib_selector_air::{air::FibonacciSelectorAir, trace::generate_trace_rows},
    get_conditional_fib_number, ProverVerifierRap,
};

mod dummy_interaction_air;

type Val = BabyBear;

fn verify_interactions(
    traces: Vec<RowMajorMatrix<Val>>,
    airs: Vec<&dyn ProverVerifierRap<StarkConfigPoseidon2>>,
    pis: Vec<Val>,
) -> Result<(), VerificationError> {
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

    let setup = PartitionSetup::new(&config);
    let (pk, vk) = setup.setup(airs.iter().map(|air| air.preprocessed_trace()).collect());

    let trace_committer = TraceCommitter::<StarkConfigPoseidon2>::new(config.pcs());
    let proven_trace = trace_committer.commit(traces);

    let proven = ProvenMultiMatrixAirTrace {
        trace_data: &proven_trace,
        airs: airs
            .iter()
            .map(|&air| air as &dyn ProverRap<StarkConfigPoseidon2>)
            .collect(),
    };

    let prover = PartitionProver::new(config);
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, vec![proven], &pis);

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = PartitionVerifier::new(prover.config);
    verifier.verify(
        &mut challenger,
        vk,
        airs.iter()
            .map(|&air| air as &dyn VerifierRap<StarkConfigPoseidon2>)
            .collect(),
        proof,
        &pis,
    )
}

#[test]
fn test_interaction_fib_selector_happy_path() {
    let log_trace_degree = 3;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    let sels: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let fib_res = get_conditional_fib_number(&sels);
    let pis = vec![a, b, fib_res]
        .into_iter()
        .map(Val::from_canonical_u32)
        .collect_vec();

    let air = FibonacciSelectorAir {
        sels: sels.clone(),
        enable_interactions: true,
    };
    let trace = generate_trace_rows::<Val>(a, b, &sels);

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
    let sender_trace = RowMajorMatrix::new(vals, 2);
    let sender_air = dummy_interaction_air::DummyInteractionAir { is_send: true };
    verify_interactions(vec![trace, sender_trace], vec![&air, &sender_air], pis)
        .expect("Verification failed");
}

fn to_field_vec(v: Vec<u32>) -> Vec<Val> {
    v.into_iter().map(Val::from_canonical_u32).collect()
}

#[test]
fn test_interaction_stark_multi_rows_happy_path() {
    // Mul  Val
    //   0    1
    //   7    4
    //   3    5
    // 546  889
    let sender_trace = RowMajorMatrix::new(to_field_vec(vec![0, 1, 3, 5, 7, 4, 546, 889]), 2);
    let sender_air = dummy_interaction_air::DummyInteractionAir { is_send: true };

    // Mul  Val
    //   1    5
    //   3    4
    //   4    4
    //   2    5
    //   0  123
    // 545  889
    //   1  889
    //   0  456
    let receiver_trace = RowMajorMatrix::new(
        to_field_vec(vec![
            1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 1, 889, 0, 456,
        ]),
        2,
    );
    let receiver_air = dummy_interaction_air::DummyInteractionAir { is_send: false };
    verify_interactions(
        vec![sender_trace, receiver_trace],
        vec![&sender_air, &receiver_air],
        vec![],
    )
    .expect("Verification failed");
}

#[test]
fn test_interaction_stark_multi_rows_neg() {
    // Mul  Val
    //   0    1
    //   3    5
    //   7    4
    // 546    0
    let sender_trace = RowMajorMatrix::new(to_field_vec(vec![0, 1, 3, 5, 7, 4, 546, 0]), 2);
    let sender_air = dummy_interaction_air::DummyInteractionAir { is_send: true };

    // count of 0 is 545 != 546 in send.
    // Mul  Val
    //   1    5
    //   3    4
    //   4    4
    //   2    5
    //   0  123
    // 545    0
    //   0    0
    //   0  456
    let receiver_trace = RowMajorMatrix::new(
        to_field_vec(vec![1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 0, 0, 0, 0, 456]),
        2,
    );
    let receiver_air = dummy_interaction_air::DummyInteractionAir { is_send: false };
    let res = verify_interactions(
        vec![sender_trace, receiver_trace],
        vec![&sender_air, &receiver_air],
        vec![],
    );
    assert_eq!(res, Err(VerificationError::NonZeroCumulativeSum));
}

#[test]
fn test_interaction_stark_all_0_sender_happy_path() {
    // Mul  Val
    //   0    1
    //   0  646
    //   0    0
    //   0  589
    let sender_trace = RowMajorMatrix::new(to_field_vec(vec![0, 1, 0, 5, 0, 4, 0, 889]), 2);
    let sender_air = dummy_interaction_air::DummyInteractionAir { is_send: true };
    verify_interactions(vec![sender_trace], vec![&sender_air], vec![])
        .expect("Verification failed");
}

#[test]
fn test_interaction_stark_multi_senders_happy_path() {
    // Mul  Val
    //   0    1
    //   6    4
    //   3    5
    // 333  889
    let sender_trace1 = RowMajorMatrix::new(to_field_vec(vec![0, 1, 3, 5, 6, 4, 333, 889]), 2);
    // Mul  Val
    //   1    4
    // 213  889
    let sender_trace2 = RowMajorMatrix::new(to_field_vec(vec![1, 4, 213, 889]), 2);

    let sender_air = dummy_interaction_air::DummyInteractionAir { is_send: true };

    // Mul  Val
    //   1    5
    //   3    4
    //   4    4
    //   2    5
    //   0  123
    // 545  889
    //   1  889
    //   0  456
    let receiver_trace = RowMajorMatrix::new(
        to_field_vec(vec![
            1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 1, 889, 0, 456,
        ]),
        2,
    );
    let receiver_air = dummy_interaction_air::DummyInteractionAir { is_send: false };
    verify_interactions(
        vec![sender_trace1, sender_trace2, receiver_trace],
        vec![&sender_air, &sender_air, &receiver_air],
        vec![],
    )
    .expect("Verification failed");
}

#[test]
fn test_interaction_stark_multi_senders_neg() {
    // Mul  Val
    //   0    1
    //   5    4
    //   3    5
    // 333  889
    let sender_trace1 = RowMajorMatrix::new(to_field_vec(vec![0, 1, 3, 5, 5, 4, 333, 889]), 2);
    // Mul  Val
    //   1    4
    // 213  889
    let sender_trace2 = RowMajorMatrix::new(to_field_vec(vec![1, 4, 213, 889]), 2);

    let sender_air = dummy_interaction_air::DummyInteractionAir { is_send: true };

    // Mul  Val
    //   1    5
    //   3    4
    //   4    4
    //   2    5
    //   0  123
    // 545  889
    //   1  889
    //   0  456
    let receiver_trace = RowMajorMatrix::new(
        to_field_vec(vec![
            1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 1, 889, 0, 456,
        ]),
        2,
    );
    let receiver_air = dummy_interaction_air::DummyInteractionAir { is_send: false };
    let res = verify_interactions(
        vec![sender_trace1, sender_trace2, receiver_trace],
        vec![&sender_air, &sender_air, &receiver_air],
        vec![],
    );
    assert_eq!(res, Err(VerificationError::NonZeroCumulativeSum));
}

#[test]
fn test_interaction_stark_multi_sender_receiver_happy_path() {
    // Mul  Val
    //   0    1
    //   6    4
    //   3    5
    // 333  889
    let sender_trace1 = RowMajorMatrix::new(to_field_vec(vec![0, 1, 3, 5, 6, 4, 333, 889]), 2);
    // Mul  Val
    //   1    4
    // 213  889
    let sender_trace2 = RowMajorMatrix::new(to_field_vec(vec![1, 4, 213, 889]), 2);

    let sender_air = dummy_interaction_air::DummyInteractionAir { is_send: true };

    // Mul  Val
    //   1    5
    //   3    4
    //   4    4
    //   2    5
    //   0  123
    // 545  889
    //   0  289
    //   0  456
    let receiver_trace1 = RowMajorMatrix::new(
        to_field_vec(vec![
            1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 0, 289, 0, 456,
        ]),
        2,
    );

    // Mul  Val
    //   1  889
    let receiver_trace2 = RowMajorMatrix::new(to_field_vec(vec![1, 889]), 2);
    let receiver_air = dummy_interaction_air::DummyInteractionAir { is_send: false };
    verify_interactions(
        vec![
            sender_trace1,
            sender_trace2,
            receiver_trace1,
            receiver_trace2,
        ],
        vec![&sender_air, &sender_air, &receiver_air, &receiver_air],
        vec![],
    )
    .expect("Verification failed");
}
