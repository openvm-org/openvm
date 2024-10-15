use afs_stark_backend::{
    engine::StarkEngine, prover::USE_DEBUG_BUILDER, verifier::VerificationError, Chip,
};
use ax_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    dummy_airs::interaction::dummy_interaction_air::{DummyInteractionChip, DummyInteractionData},
    engine::StarkFriEngine,
};
use p3_uni_stark::StarkGenericConfig;

mod instrumented;
pub mod prove;

// Lookup table is cached, everything else (including counts) is committed together
pub fn prove_and_verify_indexless_lookups(
    sender: Vec<(u32, Vec<u32>)>,
    receiver: Vec<(u32, Vec<u32>)>,
) -> Result<(), VerificationError> {
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());

    let mut sender_chip = DummyInteractionChip::new_without_partition(sender[0].1.len(), true, 0);
    let mut receiver_chip = DummyInteractionChip::new_with_partition(
        engine.config().pcs(),
        receiver[0].1.len(),
        false,
        0,
    );
    {
        let (count, fields): (Vec<_>, Vec<_>) = sender.into_iter().unzip();
        sender_chip.load_data(DummyInteractionData { count, fields });
    }
    {
        let (count, fields): (Vec<_>, Vec<_>) = receiver.into_iter().unzip();
        receiver_chip.load_data(DummyInteractionData { count, fields });
    }
    engine
        .run_test(vec![
            receiver_chip.generate_air_proof_input(),
            sender_chip.generate_air_proof_input(),
        ])
        .map(|_| ())
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
