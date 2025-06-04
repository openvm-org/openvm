// re-export types that are visible in the p256 crate for API compatibility

// Use these types instead of unpatched p256::ecdsa::{Signature, VerifyingKey}
// because those are type aliases that use non-zkvm implementations

pub use ecdsa_core::signature::{self, Error};

use super::NistP256;

/// ECDSA/secp256k1 signature (fixed-size)
pub type Signature = ecdsa_core::Signature<NistP256>;

/// ECDSA/secp256k1 signing key
#[cfg(feature = "ecdsa")]
pub type SigningKey = ecdsa_core::SigningKey<NistP256>;

/// ECDSA/secp256k1 verification key (i.e. public key)
#[cfg(feature = "ecdsa")]
pub type VerifyingKey = openvm_ecc_guest::ecdsa::VerifyingKey<NistP256>;
