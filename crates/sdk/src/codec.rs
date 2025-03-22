use std::io::{self, Result, Write};

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
/// Uses the Writer pattern for more efficient encoding without intermediate buffers.
// @dev Private trait right now just for implementation sanity
trait Encode {
    /// Writes the encoded representation of `self` to the given writer.
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()>;

    /// Convenience method to encode into a Vec<u8>
    fn encode_to_vec(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        self.encode(&mut buffer)?;
        Ok(buffer)
    }
}

// We need to know:
// - Pcs is TwoAdicFriPcs
// - Com<SC>: Into<[F; 8]>
// For simplicity, we only implement for fixed `BabyBearPoseidon2Config`
pub fn encode_proof<W: Write>(proof: &Proof<SC>, writer: &mut W) -> Result<()> {
    // Encode commitments
    encode_commitments(&proof.commitments.main_trace, writer)?;
    encode_commitments(&proof.commitments.after_challenge, writer)?;
    let quotient_commit: [F; DIGEST_SIZE] = proof.commitments.quotient.into();
    quotient_commit.encode(writer)?;

    // Encode OpeningProof
    encode_opening_proof(&proof.opening, writer)?;

    // Encode per_air data
    encode_slice(&proof.per_air, writer)?;

    // Encode logup witness
    proof.rap_phase_seq_proof.encode(writer)?;

    Ok(())
}

// Helper function to encode OpeningProof
fn encode_opening_proof<W: Write>(
    opening: &OpeningProof<PcsProof<SC>, Challenge>,
    writer: &mut W,
) -> Result<()> {
    // Encode PCS proof
    encode_pcs_proof(&opening.proof, writer)?;

    // Encode OpenedValues
    opening.values.encode(writer)?;

    Ok(())
}

impl Encode for OpenedValues<Challenge> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Encode preprocessed values
        encode_slice(&self.preprocessed, writer)?;

        // Encode main values
        (self.main.len() as u32).encode(writer)?;
        for matrices in &self.main {
            encode_slice(matrices, writer)?;
        }

        // Encode after_challenge values
        self.after_challenge.len().encode(writer)?;
        for matrices in &self.after_challenge {
            encode_slice(matrices, writer)?;
        }

        // Encode quotient values
        self.quotient.len().encode(writer)?;
        for rap in &self.quotient {
            (rap.len() as u32).encode(writer)?;
            for chunk in rap {
                encode_slice(chunk, writer)?;
            }
        }

        Ok(())
    }
}

impl Encode for AdjacentOpenedValues<Challenge> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        encode_slice(&self.local, writer)?;
        encode_slice(&self.next, writer)?;
        Ok(())
    }
}

impl Encode for AirProofData<F, Challenge> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.air_id.encode(writer)?;
        self.degree.encode(writer)?;
        self.exposed_values_after_challenge.len().encode(writer)?;
        for exposed_vals in &self.exposed_values_after_challenge {
            encode_slice(exposed_vals, writer)?;
        }
        encode_slice(&self.public_values, writer)?;
        Ok(())
    }
}

// PcsProof<SC> = InnerFriProof where Pcs = TwoAdicFriPcs
impl Encode for InnerFriProof {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        encode_commitments(&self.commit_phase_commits, writer)?;
        encode_slice(&self.query_proofs, writer)?;
        encode_slice(&self.final_poly, writer)?;
        self.pow_witness.encode(writer)?;
        Ok(())
    }
}

impl Encode for InnerQueryProof {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        encode_slice(&self.input_proof, writer)?;
        encode_slice(&self.commit_phase_openings, writer)?;
        Ok(())
    }
}

impl Encode for InnerBatchOpening {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.opened_values.len().encode(writer)?;
        for vals in &self.opened_values {
            encode_slice(vals, writer)?;
        }
        // Opening proof is just a vector of siblings
        encode_slice(&self.opening_proof, writer)?;
        Ok(())
    }
}

impl Encode for Option<FriLogUpPartialProof<F>> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        match self {
            // If exists, `F` will be < MODULUS < 2^31 so it will
            // never collide with u32::MAX
            Some(FriLogUpPartialProof { logup_pow_witness }) => logup_pow_witness.encode(writer),
            None => writer.write_all(&u32::MAX.to_le_bytes()),
        }
    }
}

impl Encode for Challenge {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        let base_slice: &[F] = self.as_base_slice();
        // Fixed length slice, so don't encode length
        for val in base_slice {
            val.encode(writer)?;
        }
        Ok(())
    }
}

/// Encodes length of slice and then each commitment
fn encode_commitments<W: Write>(commitments: &[Com<SC>], writer: &mut W) -> Result<()> {
    let coms: Vec<[F; DIGEST_SIZE]> = commitments.iter().copied().map(Into::into).collect();
    encode_slice(&coms, writer)
}

// Can't implement Encode on Com<SC> because Rust complains about associated trait types when you don't own the trait (in this case SC)
impl Encode for [F; DIGEST_SIZE] {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        for val in self {
            val.encode(writer)?;
        }
        Ok(())
    }
}

/// Encodes length of slice and then each element
fn encode_slice<T: Encode, W: Write>(slice: &[T], writer: &mut W) -> Result<()> {
    slice.len().encode(writer)?;
    for elt in slice {
        elt.encode(writer)?;
    }
    Ok(())
}

impl Encode for F {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.as_canonical_u32().to_le_bytes())
    }
}

impl Encode for usize {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        let x: u32 = (*self).try_into().map_err(io::Error::other)?;
        writer.write_all(&x.to_le_bytes())
    }
}
