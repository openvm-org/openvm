// This library current uses std::io::Read for transcript.

use halo2curves_axiom::bn256::G1Affine as Halo2G1Affine;
use loader::OpenVmLoader;
use openvm_pairing_guest::bn254::Bn254Scalar;
use serde::{Deserialize, Serialize};
use snark_verifier_sdk::snark_verifier::{
    self, halo2_base::halo2_proofs::halo2curves::bn256::Bn256 as Halo2Bn254,
    pcs::kzg::KzgDecidingKey, verifier::plonk::PlonkProtocol,
};

pub mod loader;
pub mod traits;
pub mod transcript;

/// The context necessary to verify a PLONKish SNARK proof using KZG
/// as the polynomial commitment scheme over the BN254 elliptic curve.
/// Includes the protocol, derived from the verifying key, as well as
/// the proof to verify and the public values.
pub struct PlonkVerifierContext {
    /// KZG Deciding Key, obtained from trusted setup
    pub dk: KzgDecidingKey<Halo2Bn254>,
    pub protocol: PlonkProtocol<Halo2G1Affine, OpenVmLoader>,
    pub proof: Vec<u8>,
    pub public_values: Vec<Bn254Scalar>,
    pub kzg_as: KzgAccumulationScheme,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(u8)]
pub enum KzgAccumulationScheme {
    SHPLONK,
    GWC,
}

impl PlonkVerifierContext {
    pub fn verify(self) -> Result<(), snark_verifier::Error> {
        Ok(())
    }
}
