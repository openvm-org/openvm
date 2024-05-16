use afs_stark_backend::verifier::VerificationError;

use test_utils::cached_lookup::prove_and_verify_indexless_lookups;

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

    assert_eq!(
        prove_and_verify_indexless_lookups(sender, receiver),
        Err(VerificationError::NonZeroCumulativeSum)
    );
}
