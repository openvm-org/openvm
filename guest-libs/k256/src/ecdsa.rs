// re-export types that are visible in the k256 crate for API compatibility

// Use these types instead of k256::ecdsa::{Signature, VerifyingKey}
// because those are type aliases that use non-zkvm implementations

pub use k256::ecdsa::{Error, RecoveryId};

/// ECDSA/secp256k1 signature (fixed-size)
pub type Signature = ecdsa::Signature<crate::Secp256k1>;

/// ECDSA/secp256k1 signing key
#[cfg(feature = "ecdsa")]
pub type SigningKey = ecdsa::SigningKey<crate::Secp256k1>;

/// ECDSA/secp256k1 verification key (i.e. public key)
#[cfg(feature = "ecdsa")]
pub type VerifyingKey = openvm_ecc_guest::ecdsa::VerifyingKey<crate::Secp256k1>;
