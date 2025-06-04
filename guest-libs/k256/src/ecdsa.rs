// re-export types that are visible in the k256 crate for API compatibility

// Use these types instead of unpatched k256::ecdsa::{Signature, VerifyingKey}
// because those are type aliases that use non-zkvm implementations

#[cfg(any(feature = "ecdsa", feature = "sha256"))]
pub use ecdsa_core::hazmat;
pub use ecdsa_core::{
    signature::{self, Error},
    RecoveryId,
};
#[cfg(feature = "ecdsa")]
use {super::Secp256k1Point, ecdsa_core::hazmat::VerifyPrimitive};

use super::Secp256k1;

/// ECDSA/secp256k1 signature (fixed-size)
pub type Signature = ecdsa_core::Signature<Secp256k1>;

/// ECDSA/secp256k1 signing key
#[cfg(feature = "ecdsa")]
pub type SigningKey = ecdsa_core::SigningKey<Secp256k1>;

/// ECDSA/secp256k1 verification key (i.e. public key)
#[cfg(feature = "ecdsa")]
pub type VerifyingKey = openvm_ecc_guest::ecdsa::VerifyingKey<Secp256k1>;

#[cfg(feature = "ecdsa")]
impl VerifyPrimitive<Secp256k1> for Secp256k1Point {
    fn verify_prehashed(
        &self,
        z: &crate::point::FieldBytes,
        sig: &Signature,
    ) -> Result<(), ecdsa_core::Error> {
        openvm_ecc_guest::ecdsa::verify_prehashed::<Secp256k1>(
            *self,
            z.as_slice(),
            sig.to_bytes().as_slice(),
        )
        .map_err(|_| ecdsa_core::Error::new())
    }
}
