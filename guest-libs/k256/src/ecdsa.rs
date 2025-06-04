// re-export types that are visible in the k256 crate for API compatibility

// Use these types instead of unpatched k256::ecdsa::{Signature, VerifyingKey}
// because those are type aliases that use non-zkvm implementations

#[cfg(any(feature = "ecdsa", feature = "sha256"))]
pub use ecdsa_core::hazmat;
pub use ecdsa_core::{
    signature::{self, Error},
    RecoveryId,
};

/// ECDSA/secp256k1 signature (fixed-size)
pub type Signature = ecdsa_core::Signature<crate::Secp256k1>;

/// ECDSA/secp256k1 signing key
#[cfg(feature = "ecdsa")]
pub type SigningKey = ecdsa_core::SigningKey<crate::Secp256k1>;

/// ECDSA/secp256k1 verification key (i.e. public key)
#[cfg(feature = "ecdsa")]
pub type VerifyingKey = openvm_ecc_guest::ecdsa::VerifyingKey<crate::Secp256k1>;
