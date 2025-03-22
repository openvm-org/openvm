use std::io::{self, Result};

use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_native_recursion::hints::{
    InnerBatchOpening, InnerFriProof, InnerQueryProof, InnerValMmcs,
};
use openvm_stark_backend::{
    config::{Com, PcsProof, StarkGenericConfig},
    interaction::fri_log_up::FriLogUpPartialProof,
    p3_commit::OpenedValues,
    p3_field::{extension::BinomialExtensionField, FieldExtensionAlgebra, PrimeField32},
    proof::{AdjacentOpenedValues, AirProofData, OpeningProof, Proof},
};

use super::{F, SC}; // BabyBearPoseidon2Config

type Challenge = BinomialExtensionField<F, 4>;

/// Hardware and language independent encoding.
/// Vector lengths must be unsigned integers at most `u32::MAX`.
// @dev Private trait right now just for implementation sanity
trait Encode {
    fn encode(&self) -> Result<Vec<u8>>;
}

// We need to know:
// - Pcs is TwoAdicFriPcs
// - Com<SC>: Into<[F; 8]>
// For simplicity, we only implement for fixed `BabyBearPoseidon2Config`
pub fn encode_proof(proof: &Proof<SC>) -> Result<Vec<u8>> {
    let mut encoded = Vec::new();

    // Encode commitments
    encoded.extend(encode_commitments(&proof.commitments.main_trace)?);
    encoded.extend(encode_commitments(&proof.commitments.after_challenge)?);
    let quotient_commit: [F; DIGEST_SIZE] = proof.commitments.quotient.into();
    encoded.extend(quotient_commit.encode()?);

    // Encode OpeningProof
    encoded.extend(encode_opening_proof(&proof.opening)?);

    // Encode per_air data
    encode_slice(&proof.per_air)?;
    // Encode logup witness
    encoded.extend(proof.rap_phase_seq_proof.encode()?);

    Ok(encoded)
}

// Helper function to encode OpeningProof
fn encode_opening_proof(opening: &OpeningProof<PcsProof<SC>, Challenge>) -> Result<Vec<u8>> {
    let mut encoded = Vec::new();

    // Encode PCS proof
    encoded.extend(encode_pcs_proof(&opening.proof)?);

    // Encode OpenedValues
    encoded.extend(opening.values.encode()?);

    Ok(encoded)
}

impl Encode for OpenedValues<Challenge> {
    fn encode(&self) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();

        // Encode preprocessed values
        encoded.extend(encode_slice(&values.preprocessed)?);

        // Encode main values
        let main_len: u32 = values.main.len().try_into().map_err(io::Error::other)?;
        encoded.extend(main_len.to_le_bytes().to_vec());
        for matrices in &values.main {
            encoded.extend(encode_slice(matrices)?);
        }

        // Encode after_challenge values
        let after_challenge_len: u32 = values
            .after_challenge
            .len()
            .try_into()
            .map_err(io::Error::other)?;
        encoded.extend(after_challenge_len.to_le_bytes().to_vec());
        for matrices in &values.after_challenge {
            encoded.extend(encode_slice(matrices)?);
        }

        // Encode quotient values
        let quotient_len: u32 = values.quotient.len().try_into().map_err(io::Error::other)?;
        encoded.extend(quotient_len.to_le_bytes().to_vec());
        for rap in &values.quotient {
            let rap_len: u32 = rap.len().try_into().map_err(io::Error::other)?;
            encoded.extend(rap_len.to_le_bytes().to_vec());
            for chunk in rap {
                encoded.extend(encode_slice(chunk)?);
            }
        }

        Ok(encoded)
    }
}

impl Encode for AdjacentOpenedValues<Challenge> {
    fn encode(&self) -> Result<Vec<u8>> {
        let local = encode_slice(&self.local)?;
        let next = encode_slice(&self.next)?;
        Ok([local, next].concat())
    }
}

impl Encode for AirProofData<F, Challenge> {
    fn encode(&self) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();
        encoded.extend(self.air_id.encode()?);
        encoded.extend(self.degree.encode()?);
        encoded.extend(self.exposed_values_after_challenge.len().encode()?);
        for exposed_vals in &self.exposed_values_after_challenge {
            encode_slice(exposed_vals)?;
        }
        encode_slice(&self.public_values)?;
        Ok(encoded)
    }
}

// PcsProof<SC> = InnerFriProof where Pcs = TwoAdicFriPcs
impl Encode for InnerFriProof {
    fn encode(&self) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();
        encoded.extend(encode_commitments(&self.commit_phase_commits)?);
        encode_slice(&self.query_proofs)?;
        encode_slice(&self.final_poly)?;
        encoded.extend(self.pow_witness.encode()?);

        Ok(encoded)
    }
}

impl Encode for InnerQueryProof {
    fn encode(&self) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();
        encoded.extend(encode_slice(&self.input_proof)?);
        encoded.extend(encode_slice(&self.commit_phase_openings)?);
        Ok(encoded)
    }
}

impl Encode for InnerBatchOpening {
    fn encode(&self) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();
        encoded.extend(self.opened_values.len().encode()?);
        for vals in &self.opened_values {
            encoded.extend(encode_slice(vals)?);
        }
        // Opening proof is just a vector of siblings
        encoded.extend(encode_slice(&self.opening_proof)?);
        Ok(encoded)
    }
}

impl Encode for Option<FriLogUpPartialProof<F>> {
    fn encode(&self) -> Result<Vec<u8>> {
        match self {
            // If exists, `F` will be < MODULUS < 2^31 so it will
            // never collide with u32::MAX
            Some(FriLogUpPartialProof { logup_pow_witness }) => logup_pow_witness.encode(),
            None => Ok(u32::MAX.to_le_bytes().to_vec()),
        }
    }
}

impl Encode for Challenge {
    fn encode(&self) -> Result<Vec<u8>> {
        let base_slice: &[F] = self.as_base_slice();
        encode_slice(base_slice)
    }
}

/// Encodes length of slice and then each commitment
fn encode_commitments(commitments: &[Com<SC>]) -> Result<Vec<u8>> {
    let coms: Vec<[F; DIGEST_SIZE]> = commitments.iter().copied().map(Into::into).collect();
    encode_slice(&coms)
}

// Can't implement Encode on Com<SC> because Rust complains about associated trait types when you don't own the trait (in this case SC)
impl Encode for [F; DIGEST_SIZE] {
    fn encode(&self) -> Result<Vec<u8>> {
        encode_slice(self)
    }
}

fn encode_slice<T: Encode>(slice: &[T]) -> Result<Vec<u8>> {
    let mut encoded = slice.len().encode()?;
    for elt in slice {
        encoded.extend(elt.encode()?);
    }
    Ok(encoded)
}

impl Encode for F {
    fn encode(&self) -> Result<Vec<u8>> {
        Ok(self.as_canonical_u32().to_le_bytes().to_vec())
    }
}

impl Encode for usize {
    fn encode(&self) -> Result<Vec<u8>> {
        let x: u32 = (*self).try_into().map_err(io::Error::other)?;
        Ok(x.to_le_bytes().to_vec())
    }
}
