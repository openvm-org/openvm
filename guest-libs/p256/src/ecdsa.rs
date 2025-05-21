// re-export types that are visible in the p256 crate for API compatibility

pub use p256::ecdsa::Error;

/// ECDSA/secp256k1 signature (fixed-size)
pub type Signature = ecdsa::Signature<crate::P256>;

/// ECDSA/secp256k1 signing key
#[cfg(feature = "ecdsa")]
pub type SigningKey = ecdsa::SigningKey<crate::P256>;

/// ECDSA/secp256k1 verification key (i.e. public key)
#[cfg(feature = "ecdsa")]
pub type VerifyingKey = ecdsa::VerifyingKey<crate::P256>;
