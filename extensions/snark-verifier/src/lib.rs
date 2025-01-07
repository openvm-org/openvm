// This library current uses std::io::Read for transcript.

use std::mem::transmute;

use halo2curves_axiom::bn256::{
    Bn256 as Halo2Bn254, Fr as Halo2Bn254Fr, G1Affine as Halo2G1Affine,
};
use loader::OpenVmLoader;
use openvm_pairing_guest::bn254::Bn254Scalar;
use serde::{Deserialize, Serialize};
use snark_verifier_sdk::{
    snark_verifier::{
        pcs::kzg::KzgDecidingKey,
        verifier::{plonk::PlonkProtocol, SnarkVerifier},
        Error,
    },
    PlonkVerifier, GWC, SHPLONK,
};
use traits::OpenVmScalar;
use transcript::OpenVmTranscript;

pub mod loader;
pub mod traits;
pub mod transcript;

/// The context necessary to verify a PLONKish SNARK proof using KZG
/// as the polynomial commitment scheme over the BN254 elliptic curve.
/// Includes the protocol, derived from the verifying key, as well as
/// the proof to verify and the public values.
#[derive(Clone, Serialize, Deserialize)]
pub struct PlonkVerifierContext {
    /// KZG Deciding Key, obtained from trusted setup
    pub dk: KzgDecidingKey<Halo2Bn254>,
    pub protocol: PlonkProtocol<Halo2G1Affine, OpenVmLoader>,
    pub proof: Vec<u8>,
    pub public_values: Vec<Vec<Bn254Scalar>>,
    pub kzg_as: KzgAccumulationScheme,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(u8)]
pub enum KzgAccumulationScheme {
    SHPLONK,
    GWC,
}

impl PlonkVerifierContext {
    pub fn verify(self) -> Result<(), Error> {
        let Self {
            dk,
            protocol,
            proof,
            public_values,
            kzg_as,
        } = self;
        let mut transcript = OpenVmTranscript::new(proof.as_slice());
        // SAFETY: OpenVmScalar is a repr(transparent) around Bn254Scalar
        let instances: Vec<Vec<OpenVmScalar<Halo2Bn254Fr, Bn254Scalar>>> =
            unsafe { transmute(public_values) };
        match kzg_as {
            KzgAccumulationScheme::SHPLONK => {
                let loaded_proof = PlonkVerifier::<SHPLONK>::read_proof(
                    &dk,
                    &protocol,
                    &instances[..],
                    &mut transcript,
                )?;
                // verify calls decide_all on accumulators
                PlonkVerifier::<SHPLONK>::verify(&dk, &protocol, &instances[..], &loaded_proof)?;
            }
            KzgAccumulationScheme::GWC => {
                let loaded_proof = PlonkVerifier::<GWC>::read_proof(
                    &dk,
                    &protocol,
                    &instances[..],
                    &mut transcript,
                )?;
                // verify calls decide_all on accumulators
                PlonkVerifier::<GWC>::verify(&dk, &protocol, &instances[..], &loaded_proof)?;
            }
        }
        Ok(())
    }
}
