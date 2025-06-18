// re-export types that are visible in the p256 crate for API compatibility

// Use these types instead of unpatched p256::ecdsa::{Signature, VerifyingKey}
// because those are type aliases that use non-zkvm implementations

pub use ecdsa_core::signature::{self, Error};
#[cfg(feature = "ecdsa")]
use openvm_ecc_guest::ecdsa::VerifyCustomHook;
#[cfg(feature = "ecdsa")]
use {super::P256Point, ecdsa_core::hazmat::VerifyPrimitive};

use super::NistP256;

/// ECDSA/secp256k1 signature (fixed-size)
pub type Signature = ecdsa_core::Signature<NistP256>;

/// ECDSA/secp256k1 signing key
#[cfg(feature = "ecdsa")]
pub type SigningKey = ecdsa_core::SigningKey<NistP256>;

/// ECDSA/secp256k1 verification key (i.e. public key)
#[cfg(feature = "ecdsa")]
pub type VerifyingKey = openvm_ecc_guest::ecdsa::VerifyingKey<NistP256>;

// No custom hook
#[cfg(feature = "ecdsa")]
impl VerifyCustomHook<NistP256> for P256Point {}

#[cfg(feature = "ecdsa")]
impl VerifyPrimitive<NistP256> for P256Point {
    fn verify_prehashed(
        &self,
        z: &crate::point::FieldBytes,
        sig: &Signature,
    ) -> Result<(), ecdsa_core::Error> {
        openvm_ecc_guest::ecdsa::verify_prehashed::<NistP256>(
            *self,
            z.as_slice(),
            sig.to_bytes().as_slice(),
        )
        .map_err(|_| ecdsa_core::Error::new())
    }
}
