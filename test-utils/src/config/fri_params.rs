use std::env;

use super::FriParameters;

pub fn default_fri_params() -> FriParameters {
    // blowup factor = 4
    if matches!(env::var("AXIOM_FAST_TEST"), Ok(x) if &x == "1") {
        fri_params_fast_testing()[2]
    } else {
        fri_params_with_80_bits_of_security()[2]
    }
}

/// Query phase security, ignores commit phase security which depends on field size
pub fn fri_params_with_80_bits_of_security() -> Vec<FriParameters> {
    vec![
        FriParameters {
            log_blowup: 4,
            num_queries: 45,
            proof_of_work_bits: 0,
        },
        FriParameters {
            log_blowup: 3,
            num_queries: 65,
            proof_of_work_bits: 0,
        },
        FriParameters {
            log_blowup: 2,
            num_queries: 103,
            proof_of_work_bits: 0,
        },
    ]
}

/// Query phase security, ignores commit phase security which depends on field size
pub fn fri_params_with_100_bits_of_security() -> Vec<FriParameters> {
    vec![
        FriParameters {
            log_blowup: 4,
            num_queries: 57,
            proof_of_work_bits: 0,
        },
        FriParameters {
            log_blowup: 3,
            num_queries: 80,
            proof_of_work_bits: 0,
        },
    ]
}

/// FRI parameters for fast testing. NOT secure in bits of security.
pub fn fri_params_fast_testing() -> Vec<FriParameters> {
    vec![
        FriParameters {
            log_blowup: 4,
            num_queries: 2,
            proof_of_work_bits: 0,
        },
        FriParameters {
            log_blowup: 3,
            num_queries: 2,
            proof_of_work_bits: 0,
        },
        FriParameters {
            log_blowup: 2,
            num_queries: 2,
            proof_of_work_bits: 0,
        },
    ]
}
