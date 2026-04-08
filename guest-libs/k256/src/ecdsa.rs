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
use openvm_ecc_guest::ecdsa::VerifyCustomHook;
#[cfg(feature = "ecdsa")]
use {
    super::{Scalar, Secp256k1Point},
    ecdsa_core::hazmat::{SignPrimitive, VerifyPrimitive},
    elliptic_curve::{ops::Invert, scalar::IsHigh, subtle::CtOption},
};

use super::Secp256k1;

/// ECDSA/secp256k1 signature (fixed-size)
pub type Signature = ecdsa_core::Signature<Secp256k1>;

/// ECDSA/secp256k1 signing key
#[cfg(feature = "ecdsa")]
pub type SigningKey = openvm_ecc_guest::ecdsa::SigningKey<Secp256k1>;

/// ECDSA/secp256k1 verification key (i.e. public key)
#[cfg(feature = "ecdsa")]
pub type VerifyingKey = openvm_ecc_guest::ecdsa::VerifyingKey<Secp256k1>;

// We implement the trait so that patched libraries can compile when they only need ECDSA
// verification and not signing
#[cfg(feature = "ecdsa")]
impl SignPrimitive<Secp256k1> for Scalar {
    fn try_sign_prehashed<K>(
        &self,
        _k: K,
        _z: &elliptic_curve::FieldBytes<Secp256k1>,
    ) -> signature::Result<(Signature, Option<RecoveryId>)>
    where
        K: AsRef<Self> + Invert<Output = CtOption<Self>>,
    {
        todo!("ECDSA signing from private key is not yet implemented")
    }
}

#[cfg(feature = "ecdsa")]
impl VerifyCustomHook<Secp256k1> for Secp256k1Point {
    #[inline]
    fn verify_hook(&self, _z: &[u8], sig: &Signature) -> signature::Result<()> {
        if sig.s().is_high().into() {
            return Err(Error::new());
        }
        Ok(())
    }
}

#[cfg(feature = "ecdsa")]
impl VerifyPrimitive<Secp256k1> for Secp256k1Point {
    fn verify_prehashed(
        &self,
        z: &crate::point::FieldBytes,
        sig: &Signature,
    ) -> Result<(), ecdsa_core::Error> {
        self.verify_hook(z, sig)?;

        openvm_ecc_guest::ecdsa::verify_prehashed::<Secp256k1>(
            *self,
            z.as_slice(),
            sig.to_bytes().as_slice(),
        )
        .map_err(|_| ecdsa_core::Error::new())
    }
}
