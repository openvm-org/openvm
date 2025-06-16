use std::io::{self, Cursor, Read, Result, Write};

use openvm_circuit::{
    arch::ContinuationVmProof, system::memory::tree::public_values::UserPublicValuesProof,
};
use openvm_continuations::verifier::{
    internal::types::VmStarkProof, root::types::RootVmVerifierInput,
};
use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_native_recursion::hints::{InnerBatchOpening, InnerFriProof, InnerQueryProof};
use openvm_stark_backend::{
    config::{Com, PcsProof},
    interaction::{fri_log_up::FriLogUpPartialProof, RapPhaseSeqKind},
    p3_field::{
        extension::BinomialExtensionField, FieldAlgebra, FieldExtensionAlgebra, PrimeField32,
    },
    proof::{AdjacentOpenedValues, AirProofData, Commitments, OpenedValues, OpeningProof, Proof},
};
use p3_fri::CommitPhaseProofStep;

use super::{F, SC};

type Challenge = BinomialExtensionField<F, 4>;

/// Codec version should change only when proof system or proof format changes.
/// It does correspond to the main openvm version (which may change more frequently).
const CODEC_VERSION: u32 = 1;

/// Hardware and language independent encoding.
/// Uses the Writer pattern for more efficient encoding without intermediate buffers.
// @dev Trait just for implementation sanity
pub trait Encode {
    /// Writes the encoded representation of `self` to the given writer.
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()>;

    /// Convenience method to encode into a `Vec<u8>`
    fn encode_to_vec(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        self.encode(&mut buffer)?;
        Ok(buffer)
    }
}

/// Hardware and language independent decoding.
/// Uses the Reader pattern for efficient decoding.
pub trait Decode: Sized {
    /// Reads and decodes a value from the given reader.
    fn decode<R: Read>(reader: &mut R) -> Result<Self>;
    fn decode_from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = Cursor::new(bytes);
        Self::decode(&mut reader)
    }
}

// ==================== Encode implementation ====================

impl Encode for ContinuationVmProof<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        encode_slice(&self.per_segment, writer)?;
        self.user_public_values.encode(writer)
    }
}

impl Encode for VmStarkProof<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.proof.encode(writer)?;
        encode_slice(&self.user_public_values, writer)
    }
}

impl Encode for UserPublicValuesProof<DIGEST_SIZE, F> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        encode_slice(&self.proof, writer)?;
        encode_slice(&self.public_values, writer)?;
        self.public_values_commit.encode(writer)
    }
}

impl Encode for RootVmVerifierInput<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        encode_slice(&self.proofs, writer)?;
        encode_slice(&self.public_values, writer)
    }
}

impl Encode for Proof<SC> {
    // We need to know:
    // - Pcs is TwoAdicFriPcs
    // - Com<SC>: Into<[F; 8]>
    // For simplicity, we only implement for fixed `BabyBearPoseidon2Config`
    //
    /// Encode a proof using FRI as the PCS with `BabyBearPoseidon2Config`.
    /// The Merkle tree hashes have digest `[F; 8]`.
    /// ```
    /// pub struct Proof<SC: StarkGenericConfig> {
    ///     pub commitments: Commitments<Com<SC>>,
    ///     pub opening: OpeningProof<PcsProof<SC>, SC::Challenge>,
    ///     pub per_air: Vec<AirProofData<Val<SC>, SC::Challenge>>,
    ///     pub rap_phase_seq_proof: Option<RapPhaseSeqPartialProof<SC>>,
    /// }
    /// ```
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&CODEC_VERSION.to_le_bytes())?;
        // Encode commitments
        encode_commitments(&self.commitments.main_trace, writer)?;
        encode_commitments(&self.commitments.after_challenge, writer)?;
        let quotient_commit: [F; DIGEST_SIZE] = self.commitments.quotient.into();
        quotient_commit.encode(writer)?;

        // Encode OpeningProof
        encode_opening_proof(&self.opening, writer)?;

        // Encode per_air data
        encode_slice(&self.per_air, writer)?;

        writer.write_all(&[RapPhaseSeqKind::FriLogUp as u8])?;
        // Encode logup witness
        self.rap_phase_seq_proof.encode(writer)?;

        Ok(())
    }
}

// Helper function to encode OpeningProof
// ```
// pub struct OpeningProof<PcsProof, Challenge> {
//     pub proof: PcsProof,
//     pub values: OpenedValues<Challenge>,
// }
// ```
fn encode_opening_proof<W: Write>(
    opening: &OpeningProof<PcsProof<SC>, Challenge>,
    writer: &mut W,
) -> Result<()> {
    // Encode FRI proof
    opening.proof.encode(writer)?;
    encode_opened_values(&opening.values, writer)?;
    Ok(())
}

/// [OpenedValues] is a typedef for `Vec<Vec<Vec<Vec<F>>>>` for
/// - each round
///   - each matrix
///     - each point to open at
///       - evaluations for each column of matrix at that point
fn encode_opened_values<W: Write>(
    opened_values: &OpenedValues<Challenge>,
    writer: &mut W,
) -> Result<()> {
    encode_slice(&opened_values.preprocessed, writer)?;
    opened_values.main.len().encode(writer)?;
    for part in &opened_values.main {
        encode_slice(part, writer)?;
    }
    opened_values.after_challenge.len().encode(writer)?;
    for phase in &opened_values.after_challenge {
        encode_slice(phase, writer)?;
    }
    opened_values.quotient.len().encode(writer)?;
    for per_air in &opened_values.quotient {
        per_air.len().encode(writer)?;
        for chunk in per_air {
            encode_slice(chunk, writer)?;
        }
    }

    Ok(())
}

impl Encode for AdjacentOpenedValues<Challenge> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        encode_slice(&self.local, writer)?;
        encode_slice(&self.next, writer)?;
        Ok(())
    }
}

impl Encode for AirProofData<F, Challenge> {
    /// Encodes the struct
    /// ```
    /// pub struct OpenedValues<Challenge> {
    ///     pub preprocessed: Vec<AdjacentOpenedValues<Challenge>>,
    ///     pub main: Vec<Vec<AdjacentOpenedValues<Challenge>>>,
    ///     pub after_challenge: Vec<Vec<AdjacentOpenedValues<Challenge>>>,
    ///     pub quotient: Vec<Vec<Vec<Challenge>>>,
    /// }
    /// ```
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
    /// Encodes the struct
    /// ```
    /// pub struct FriProof<Challenge, M: Mmcs<Challenge>> {
    ///     pub commit_phase_commits: Vec<M::Commitment>,
    ///     pub query_proofs: Vec<QueryProof<Challenge, M, Vec<BatchOpening<F>>>>,
    ///     pub final_poly: Vec<Challenge>,
    ///     pub pow_witness: F,
    /// }
    /// ```
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        encode_commitments(&self.commit_phase_commits, writer)?;
        encode_slice(&self.query_proofs, writer)?;
        encode_slice(&self.final_poly, writer)?;
        self.pow_witness.encode(writer)?;
        Ok(())
    }
}

impl Encode for InnerQueryProof {
    /// Encodes the struct
    /// ```
    /// pub struct QueryProof<Challenge, M: Mmcs<Challenge>> {
    ///     pub input_proof: Vec<BatchOpening<F>>,
    ///     pub commit_phase_openings: Vec<CommitPhaseProofStep<Challenge, M>>,
    /// }
    ///
    /// pub struct BatchOpening<F> {
    ///     pub opened_values: Vec<Vec<F>>,
    ///     pub opening_proof: Vec<[F; DIGEST_SIZE]>,
    /// }
    ///
    /// pub struct CommitPhaseProofStep<Challenge, M: Mmcs<Challenge>> {
    ///     pub sibling_value: Challenge,
    ///     pub opening_proof: Vec<[F; DIGEST_SIZE]>,
    /// }
    /// ```
    // @dev [jpw]: We prefer to keep the implementation all in one function
    // without `impl Encode` on subtypes because it obfuscates what the overall
    // struct consists of.
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Input proof is Vec<BatchOpening<F>>
        self.input_proof.len().encode(writer)?;
        for batch_opening in &self.input_proof {
            batch_opening.opened_values.len().encode(writer)?;
            for vals in &batch_opening.opened_values {
                encode_slice(vals, writer)?;
            }
            // Opening proof is just a vector of siblings
            encode_slice(&batch_opening.opening_proof, writer)?;
        }
        self.commit_phase_openings.len().encode(writer)?;
        for step in &self.commit_phase_openings {
            step.sibling_value.encode(writer)?;
            encode_slice(&step.opening_proof, writer)?;
        }
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

// Can't implement Encode on Com<SC> because Rust complains about associated trait types when you
// don't own the trait (in this case SC)
impl Encode for [F; DIGEST_SIZE] {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        for val in self {
            val.encode(writer)?;
        }
        Ok(())
    }
}

/// Encodes length of slice and then each element
pub(crate) fn encode_slice<T: Encode, W: Write>(slice: &[T], writer: &mut W) -> Result<()> {
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

// ============ Decode implementation =============

impl Decode for ContinuationVmProof<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let per_segment = decode_vec(reader)?;
        let user_public_values = UserPublicValuesProof::decode(reader)?;
        Ok(Self {
            per_segment,
            user_public_values,
        })
    }
}

impl Decode for VmStarkProof<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let proof = Proof::decode(reader)?;
        let user_public_values = decode_vec(reader)?;
        Ok(Self {
            proof,
            user_public_values,
        })
    }
}

impl Decode for UserPublicValuesProof<DIGEST_SIZE, F> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let proof = decode_vec(reader)?;
        let public_values = decode_vec(reader)?;
        let public_values_commit = <[F; DIGEST_SIZE]>::decode(reader)?;
        Ok(Self {
            proof,
            public_values,
            public_values_commit,
        })
    }
}

impl Decode for RootVmVerifierInput<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let proofs = decode_vec(reader)?;
        let public_values = decode_vec(reader)?;
        Ok(Self {
            proofs,
            public_values,
        })
    }
}

impl Decode for Proof<SC> {
    /// Decode a proof using FRI as the PCS with `BabyBearPoseidon2Config`.
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        if version != CODEC_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid codec version. Expected {}, got {}",
                    CODEC_VERSION, version
                ),
            ));
        }

        // Decode commitments
        let main_trace = decode_commitments(reader)?;
        let after_challenge = decode_commitments(reader)?;
        let quotient = decode_commitment(reader)?;

        let commitments = Commitments {
            main_trace,
            after_challenge,
            quotient,
        };

        // Decode OpeningProof
        let opening = decode_opening_proof(reader)?;

        // Decode per_air data
        let per_air = decode_vec(reader)?;

        // Decode RAP phase sequence kind
        let mut kind_byte = [0u8; 1];
        reader.read_exact(&mut kind_byte)?;
        if kind_byte[0] != RapPhaseSeqKind::FriLogUp as u8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown RapPhaseSeqKind: {}", kind_byte[0]),
            ));
        }

        // Decode logup witness
        let rap_phase_seq_proof = Option::<FriLogUpPartialProof<F>>::decode(reader)?;

        Ok(Proof {
            commitments,
            opening,
            per_air,
            rap_phase_seq_proof,
        })
    }
}

fn decode_commitment<R: Read>(reader: &mut R) -> Result<Com<SC>> {
    let digest = <[F; DIGEST_SIZE]>::decode(reader)?;
    // Convert [F; DIGEST_SIZE] to Com<SC>
    Ok(digest.into())
}

fn decode_commitments<R: Read>(reader: &mut R) -> Result<Vec<Com<SC>>> {
    let coms_count = usize::decode(reader)?;
    let mut coms = Vec::with_capacity(coms_count);

    for _ in 0..coms_count {
        coms.push(decode_commitment(reader)?);
    }

    Ok(coms)
}

fn decode_opening_proof<R: Read>(reader: &mut R) -> Result<OpeningProof<PcsProof<SC>, Challenge>> {
    // Decode FRI proof
    let proof = InnerFriProof::decode(reader)?;
    let values = decode_opened_values(reader)?;

    Ok(OpeningProof { proof, values })
}

fn decode_opened_values<R: Read>(reader: &mut R) -> Result<OpenedValues<Challenge>> {
    let preprocessed = decode_vec(reader)?;

    let main_count = usize::decode(reader)?;
    let mut main = Vec::with_capacity(main_count);
    for _ in 0..main_count {
        main.push(decode_vec(reader)?);
    }

    let after_challenge_count = usize::decode(reader)?;
    let mut after_challenge = Vec::with_capacity(after_challenge_count);
    for _ in 0..after_challenge_count {
        after_challenge.push(decode_vec(reader)?);
    }

    let quotient_count = usize::decode(reader)?;
    let mut quotient = Vec::with_capacity(quotient_count);
    for _ in 0..quotient_count {
        let per_air_count = usize::decode(reader)?;
        let mut per_air = Vec::with_capacity(per_air_count);
        for _ in 0..per_air_count {
            per_air.push(decode_vec(reader)?);
        }
        quotient.push(per_air);
    }

    Ok(OpenedValues {
        preprocessed,
        main,
        after_challenge,
        quotient,
    })
}

impl Decode for AdjacentOpenedValues<Challenge> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let local = decode_vec(reader)?;
        let next = decode_vec(reader)?;

        Ok(AdjacentOpenedValues { local, next })
    }
}

impl Decode for AirProofData<F, Challenge> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let air_id = usize::decode(reader)?;
        let degree = usize::decode(reader)?;

        let exposed_values_count = usize::decode(reader)?;
        let mut exposed_values_after_challenge = Vec::with_capacity(exposed_values_count);
        for _ in 0..exposed_values_count {
            exposed_values_after_challenge.push(decode_vec(reader)?);
        }

        let public_values = decode_vec(reader)?;

        Ok(AirProofData {
            air_id,
            degree,
            exposed_values_after_challenge,
            public_values,
        })
    }
}

impl Decode for InnerFriProof {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let commit_phase_commits = decode_commitments(reader)?;
        let query_proofs = decode_vec(reader)?;
        let final_poly = decode_vec(reader)?;
        let pow_witness = F::decode(reader)?;

        Ok(InnerFriProof {
            commit_phase_commits,
            query_proofs,
            final_poly,
            pow_witness,
        })
    }
}

impl Decode for InnerQueryProof {
    /// See [InnerQueryProof::encode].
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let batch_opening_count = usize::decode(reader)?;
        let mut input_proof = Vec::with_capacity(batch_opening_count);
        for _ in 0..batch_opening_count {
            let opened_values_len = usize::decode(reader)?;
            let mut opened_values = Vec::with_capacity(opened_values_len);
            for _ in 0..opened_values_len {
                opened_values.push(decode_vec(reader)?);
            }
            let opening_proof = decode_vec(reader)?;

            let batch_opening = InnerBatchOpening {
                opened_values,
                opening_proof,
            };
            input_proof.push(batch_opening);
        }

        let commit_phase_openings_count = usize::decode(reader)?;
        let mut commit_phase_openings = Vec::with_capacity(commit_phase_openings_count);

        for _ in 0..commit_phase_openings_count {
            let sibling_value = Challenge::decode(reader)?;
            let opening_proof = decode_vec(reader)?;

            commit_phase_openings.push(CommitPhaseProofStep {
                sibling_value,
                opening_proof,
            });
        }

        Ok(InnerQueryProof {
            input_proof,
            commit_phase_openings,
        })
    }
}

impl Decode for Option<FriLogUpPartialProof<F>> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;

        let value = u32::from_le_bytes(bytes);
        // When `Option<FriLogUpPartialProof<F>>` is None, it's encoded as `u32::max`.
        if value == u32::MAX {
            return Ok(None);
        }

        // Reconstruct the field element from the u32 value
        let logup_pow_witness = F::from_canonical_u32(value);
        Ok(Some(FriLogUpPartialProof { logup_pow_witness }))
    }
}

impl Decode for Challenge {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        // For a BinomialExtensionField<F, 4>, we need to read 4 F elements
        let mut base_elements = [F::ZERO; 4];
        for base_element in &mut base_elements {
            *base_element = F::decode(reader)?;
        }

        // Construct the extension field from base elements
        Ok(Challenge::from_base_slice(&base_elements))
    }
}

impl Decode for [F; DIGEST_SIZE] {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let mut result = [F::ZERO; DIGEST_SIZE];
        for elt in &mut result {
            *elt = F::decode(reader)?;
        }
        Ok(result)
    }
}

/// Decodes a vector of elements
pub(crate) fn decode_vec<T: Decode, R: Read>(reader: &mut R) -> Result<Vec<T>> {
    let len = usize::decode(reader)?;
    let mut vec = Vec::with_capacity(len);

    for _ in 0..len {
        vec.push(T::decode(reader)?);
    }

    Ok(vec)
}

impl Decode for F {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;

        let value = u32::from_le_bytes(bytes);
        Ok(F::from_canonical_u32(value))
    }
}

impl Decode for usize {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;

        let value = u32::from_le_bytes(bytes);
        Ok(value as usize)
    }
}
