use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FriParameters {
    pub log_blowup: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
}

impl FriParameters {
    /// Conjectured bits of security.
    /// See ethSTARK paper (https://eprint.iacr.org/2021/582.pdf) section 5.10.1 equation (19)
    ///
    /// `challenge_field_bits` is the number of bits in the challenge field (extension field) of the STARK config.
    pub fn get_conjectured_security_bits(&self, challenge_field_bits: usize) -> usize {
        let fri_query_security_bits = self.num_queries * self.log_blowup + self.proof_of_work_bits;
        // The paper says min(fri_field_bits, fri_query_security_bits) - 1 but plonky2 (https://github.com/0xPolygonZero/plonky2/blob/41dc325e61ab8d4c0491e68e667c35a4e8173ffa/starky/src/config.rs#L86C1-L87C1) omits the -1
        challenge_field_bits.min(fri_query_security_bits)
    }

    pub fn standard_fast() -> FriParameters {
        standard_fri_params_with_100_bits_conjectured_security(1)
    }

    pub fn standard_with_100_bits_conjectured_security(log_blowup: usize) -> FriParameters {
        standard_fri_params_with_100_bits_conjectured_security(log_blowup)
    }
}

/// Pre-defined FRI parameters with 100 bits of conjectured security.
/// Security bits calculated following ethSTARK (https://eprint.iacr.org/2021/582.pdf) 5.10.1 eq (19)
///
/// Assumes that the challenge field used as more than 100 bits.
pub fn standard_fri_params_with_100_bits_conjectured_security(log_blowup: usize) -> FriParameters {
    if let Ok("1") = std::env::var("AXIOM_FAST_TEST").as_deref() {
        return FriParameters {
            log_blowup,
            num_queries: 2,
            proof_of_work_bits: 0,
        };
    }
    let fri_params = match log_blowup {
        // plonky2 standard fast config uses num_queries=84: https://github.com/0xPolygonZero/plonky2/blob/41dc325e61ab8d4c0491e68e667c35a4e8173ffa/starky/src/config.rs#L49
        // plonky3's default is num_queries=100, so we will use that. See https://github.com/Plonky3/Plonky3/issues/380 for related security discussion.
        1 => FriParameters {
            log_blowup,
            num_queries: 100,
            proof_of_work_bits: 16,
        },
        2 => FriParameters {
            log_blowup,
            num_queries: 42,
            proof_of_work_bits: 16,
        },
        // plonky2 standard recursion config: https://github.com/0xPolygonZero/plonky2/blob/41dc325e61ab8d4c0491e68e667c35a4e8173ffa/plonky2/src/plonk/circuit_data.rs#L101
        3 => FriParameters {
            log_blowup,
            num_queries: 28,
            proof_of_work_bits: 16,
        },
        4 => FriParameters {
            log_blowup,
            num_queries: 21,
            proof_of_work_bits: 16,
        },
        _ => todo!("No standard FRI params defined for log blowup {log_blowup}",),
    };
    assert!(fri_params.get_conjectured_security_bits(100) >= 100);
    tracing::info!("FRI parameters | log_blowup: {log_blowup:<2} | num_queries: {:<2} | proof_of_work_bits: {:<2}", fri_params.num_queries, fri_params.proof_of_work_bits);
    fri_params
}
