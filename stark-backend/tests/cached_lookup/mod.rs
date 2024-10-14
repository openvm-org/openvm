use std::{iter, sync::Arc};

use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{
        types::{AirProofInput, CommittedTraceData, ProofInput, TraceCommitter},
        MultiTraceStarkProver, USE_DEBUG_BUILDER,
    },
    verifier::{MultiTraceStarkVerifier, VerificationError},
};
use ax_sdk::dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::StarkGenericConfig;
use p3_util::log2_ceil_usize;

use crate::config;

mod instrumented;
pub mod prove;

type Val = BabyBear;

// Lookup table is cached, everything else (including counts) is committed together
pub fn prove_and_verify_indexless_lookups(
    sender: Vec<(u32, Vec<u32>)>,
    receiver: Vec<(u32, Vec<u32>)>,
) -> Result<(), VerificationError> {
    let sender_degree = sender.len();
    let receiver_degree = receiver.len();
    let [sender_log_degree, receiver_log_degree] =
        [sender_degree, receiver_degree].map(log2_ceil_usize);

    let perm = config::baby_bear_poseidon2::random_perm();
    let config = config::baby_bear_poseidon2::default_config(
        &perm,
        sender_log_degree.max(receiver_log_degree),
    );

    let sender_air = DummyInteractionAir::new(sender[0].1.len(), true, 0);
    let receiver_air = DummyInteractionAir::new(receiver[0].1.len(), false, 0).partition();

    // Single row major matrix for |count|fields[..]|
    let sender_trace = RowMajorMatrix::new(
        sender
            .into_iter()
            .flat_map(|(count, fields)| {
                assert_eq!(fields.len(), sender_air.field_width());
                iter::once(count).chain(fields)
            })
            .map(Val::from_wrapped_u32)
            .collect(),
        sender_air.field_width() + 1,
    );
    let (recv_count, recv_fields): (Vec<_>, Vec<_>) = receiver.into_iter().unzip();
    let recv_count_trace = RowMajorMatrix::new(
        recv_count.into_iter().map(Val::from_wrapped_u32).collect(),
        1,
    );
    let recv_fields_trace = RowMajorMatrix::new(
        recv_fields
            .into_iter()
            .flat_map(|fields| {
                assert_eq!(fields.len(), receiver_air.field_width());
                fields
            })
            .map(Val::from_wrapped_u32)
            .collect(),
        receiver_air.field_width(),
    );
    {
        let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
        let receiver_air_id = keygen_builder.add_air(Arc::new(receiver_air));
        // Auto-adds sender matrix
        let sender_air_id = keygen_builder.add_air(Arc::new(sender_air));
        let pk = keygen_builder.generate_pk();
        let committer = TraceCommitter::new(config.pcs());
        let cached_trace_data = committer.commit(vec![recv_fields_trace.clone()]);
        let proof_input = ProofInput {
            per_air: vec![
                (
                    receiver_air_id,
                    AirProofInput {
                        air: Arc::new(receiver_air),
                        cached_mains: vec![CommittedTraceData {
                            raw_data: recv_fields_trace,
                            prover_data: cached_trace_data,
                        }],
                        common_main: Some(recv_count_trace),
                        public_values: vec![],
                    },
                ),
                (
                    sender_air_id,
                    AirProofInput {
                        air: Arc::new(sender_air),
                        cached_mains: vec![],
                        common_main: Some(sender_trace),
                        public_values: vec![],
                    },
                ),
            ],
        };

        let prover = MultiTraceStarkProver::new(&config);

        let mut challenger = config::baby_bear_poseidon2::Challenger::new(perm.clone());
        let proof = prover.prove(&mut challenger, &pk, proof_input);

        // Verify the proof:
        // Start from clean challenger
        let mut challenger = config::baby_bear_poseidon2::Challenger::new(perm.clone());
        let verifier = MultiTraceStarkVerifier::new(prover.config);
        verifier.verify(&mut challenger, &pk.get_vk(), &proof)
    }
}

/// tests for cached_lookup
#[test]
fn test_interaction_cached_trace_happy_path() {
    // count fields
    //   0    1 1
    //   7    4 2
    //   3    5 1
    // 546  889 4
    let sender = vec![
        (0, vec![1, 1]),
        (7, vec![4, 2]),
        (3, vec![5, 1]),
        (546, vec![889, 4]),
    ];

    // count fields
    //   1    5 1
    //   3    4 2
    //   4    4 2
    //   2    5 1
    //   0  123 3
    // 545  889 4
    //   1  889 4
    //   0  456 5
    let receiver = vec![
        (1, vec![5, 1]),
        (3, vec![4, 2]),
        (4, vec![4, 2]),
        (2, vec![5, 1]),
        (0, vec![123, 3]),
        (545, vec![889, 4]),
        (1, vec![889, 4]),
        (0, vec![456, 5]),
    ];

    prove_and_verify_indexless_lookups(sender, receiver).expect("Verification failed");
}

#[test]
fn test_interaction_cached_trace_neg() {
    // count fields
    //   0    1 1
    //   7    4 2
    //   3    5 1
    // 546  889 4
    let sender = vec![
        (0, vec![1, 1]),
        (7, vec![4, 2]),
        (3, vec![5, 1]),
        (546, vec![889, 4]),
    ];

    // field [889, 4] has count 545 != 546 in sender
    // count fields
    //   1    5 1
    //   3    4 2
    //   4    4 2
    //   2    5 1
    //   0  123 3
    // 545  889 4
    //   1  889 10
    //   0  456 5
    let receiver = vec![
        (1, vec![5, 1]),
        (3, vec![4, 2]),
        (4, vec![4, 2]),
        (2, vec![5, 1]),
        (0, vec![123, 3]),
        (545, vec![889, 4]),
        (1, vec![889, 10]),
        (0, vec![456, 5]),
    ];

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        prove_and_verify_indexless_lookups(sender, receiver),
        Err(VerificationError::NonZeroCumulativeSum)
    );
}
