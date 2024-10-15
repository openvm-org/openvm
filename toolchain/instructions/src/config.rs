use serde::{Deserialize, Serialize};

/// Struct to store the configuration parameters for custom enabled opcodes.
/// These parameters are supplied by the front-end user **before** the program is compiled.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CustomOpConfig {
    pub intrinsics: IntrinsicsOpConfig,
}

/// Configuration parameters for the intrinsics opcodes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntrinsicsOpConfig {
    pub field_arithmetic: FieldArithmeticOpConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldArithmeticOpConfig {
    /// **Ordered** list of enabled prime moduli.
    pub primes: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_toml() {
        let config = CustomOpConfig {
            intrinsics: IntrinsicsOpConfig {
                field_arithmetic: FieldArithmeticOpConfig {
                    primes: vec![
                        "0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC"
                            .to_string(),
                    ],
                },
            },
        };
        println!("{}", toml::to_string(&config).unwrap());
    }
}
