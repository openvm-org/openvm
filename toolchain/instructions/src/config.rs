use num_bigint::BigUint;
use num_traits::Num;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_with::{serde_as, DeserializeAs, SerializeAs};

/// Struct to store the configuration parameters for custom enabled opcodes.
/// These parameters are supplied by the front-end user **before** the program is compiled.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CustomOpConfig {
    /// Configuration parameters for custom opcodes used in intrinsics.
    pub intrinsics: IntrinsicsOpConfig,
    // In the future, we will add config for kernel opcodes.
}

/// Configuration parameters for the intrinsics opcodes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntrinsicsOpConfig {
    pub field_arithmetic: FieldArithmeticOpConfig,
}

#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldArithmeticOpConfig {
    /// **Ordered** list of enabled prime moduli.
    #[serde_as(as = "Vec<HumanReadableBigUint>")]
    pub primes: Vec<BigUint>,
}

struct HumanReadableBigUint;

impl SerializeAs<BigUint> for HumanReadableBigUint {
    fn serialize_as<S>(value: &BigUint, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_str_radix(10))
    }
}

impl<'de> DeserializeAs<'de, BigUint> for HumanReadableBigUint {
    fn deserialize_as<D>(deserializer: D) -> Result<BigUint, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        parse_biguint_auto(&s)
            .ok_or_else(|| serde::de::Error::custom("Failed to parse BigUint from hex or decimal"))
    }
}

pub fn parse_biguint_auto(s: &str) -> Option<BigUint> {
    let s = s.trim();
    if s.starts_with("0x") || s.starts_with("0X") {
        BigUint::from_str_radix(&s[2..], 16).ok()
    } else if s.starts_with("0b") || s.starts_with("0B") {
        BigUint::from_str_radix(&s[2..], 2).ok()
    } else {
        BigUint::from_str_radix(s, 10).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_toml() {
        let config = CustomOpConfig {
            intrinsics: IntrinsicsOpConfig {
                field_arithmetic: FieldArithmeticOpConfig {
                    primes: vec![parse_biguint_auto(
                        "0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC",
                    )
                    .unwrap()],
                },
            },
        };
        println!("{}", toml::to_string(&config).unwrap());
    }
}
