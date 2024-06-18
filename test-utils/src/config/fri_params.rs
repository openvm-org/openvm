use super::FriParameters;

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
