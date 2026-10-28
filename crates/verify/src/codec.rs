use std::io::{BufReader, Read};

use openvm_circuit::system::memory::merkle::public_values::UserPublicValuesProof;
use openvm_stark_backend::{
    codec::{Decode, Encode},
    keygen::types::MultiStarkVerifyingKey,
    proof::Proof,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as SC, DIGEST_SIZE, F,
};

use crate::{deferral::DeferralMerkleProofs, vk::VerificationBaseline, VmStarkProof};

/// Maximum zstd log window size accepted when decompressing a proof, i.e. 32 MiB. Bounds
/// the decoder's window allocation; zstd levels up to 20 use at most this size.
const MAX_ZSTD_WINDOW_LOG: u32 = 25;

impl Encode for VmStarkProof {
    fn encode<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.inner.encode(writer)?;
        self.user_pvs_proof.encode::<SC, _>(writer)?;
        (self.deferral_merkle_proofs.is_some() as u8).encode(writer)?;
        if let Some(ref proofs) = self.deferral_merkle_proofs {
            proofs.encode(writer)?;
        }
        Ok(())
    }
}

impl Decode for VmStarkProof {
    fn decode<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let inner = Proof::<SC>::decode(reader)?;
        Self::decode_suffix(inner, reader, usize::MAX, usize::MAX)
    }
}

impl VmStarkProof {
    /// Decode using size limits from a trusted vk and baseline.
    pub fn decode_using_vk<R: Read>(
        mvk: &MultiStarkVerifyingKey<SC>,
        baseline: &VerificationBaseline,
        reader: &mut R,
    ) -> std::io::Result<Self> {
        let inner = Proof::<SC>::decode_using_vk(&mvk.inner, reader)?;
        Self::decode_suffix(
            inner,
            reader,
            baseline.memory_dimensions.overall_height(),
            baseline.num_user_pvs,
        )
    }

    /// Decode using size limits from a trusted vk and baseline, rejecting trailing bytes.
    pub fn decode_from_bytes_using_vk(
        mvk: &MultiStarkVerifyingKey<SC>,
        baseline: &VerificationBaseline,
        bytes: &[u8],
    ) -> std::io::Result<Self> {
        let mut reader = bytes;
        let proof = Self::decode_using_vk(mvk, baseline, &mut reader)?;
        if !reader.is_empty() {
            return Err(std::io::Error::other("trailing bytes after decoded value"));
        }
        Ok(proof)
    }

    /// Decode a zstd-compressed encoding using size limits from a trusted vk and baseline,
    /// rejecting trailing bytes. The compressed stream is decoded directly, so only as much
    /// as the limits allow is ever decompressed. Callers reading encoded_proof from an
    /// untrusted source should bound its size before the read.
    pub fn decode_compressed_using_vk(
        mvk: &MultiStarkVerifyingKey<SC>,
        baseline: &VerificationBaseline,
        encoded_proof: &[u8],
    ) -> std::io::Result<Self> {
        let mut decoder = zstd::stream::read::Decoder::with_buffer(encoded_proof)?.single_frame();
        decoder.window_log_max(MAX_ZSTD_WINDOW_LOG)?;
        let mut reader = BufReader::new(decoder);
        let proof = Self::decode_using_vk(mvk, baseline, &mut reader)?;
        if reader.read(&mut [0u8; 1])? != 0 || !reader.into_inner().finish().is_empty() {
            return Err(std::io::Error::other("trailing bytes after decoded value"));
        }
        Ok(proof)
    }

    fn decode_suffix<R: Read>(
        inner: Proof<SC>,
        reader: &mut R,
        max_merkle_proof_len: usize,
        max_num_user_pvs: usize,
    ) -> std::io::Result<Self> {
        let user_pvs_proof = UserPublicValuesProof::decode_with_limits::<SC, _>(
            reader,
            max_merkle_proof_len,
            max_num_user_pvs,
        )?;
        let deferral_merkle_proofs = if u8::decode(reader)? != 0 {
            Some(DeferralMerkleProofs::decode_with_limit(
                reader,
                max_merkle_proof_len,
            )?)
        } else {
            None
        };
        Ok(Self {
            inner,
            user_pvs_proof,
            deferral_merkle_proofs,
        })
    }
}

/// Decode a [UserPublicValuesProof] using size limits from a trusted baseline.
pub fn decode_user_pvs_proof<R: Read>(
    reader: &mut R,
    baseline: &VerificationBaseline,
) -> std::io::Result<UserPublicValuesProof<DIGEST_SIZE, F>> {
    UserPublicValuesProof::decode_with_limits::<SC, _>(
        reader,
        baseline.memory_dimensions.overall_height(),
        baseline.num_user_pvs,
    )
}

/// Decode [DeferralMerkleProofs] using size limits from a trusted baseline.
pub fn decode_deferral_merkle_proofs<R: Read>(
    reader: &mut R,
    baseline: &VerificationBaseline,
) -> std::io::Result<DeferralMerkleProofs<F>> {
    DeferralMerkleProofs::decode_with_limit(reader, baseline.memory_dimensions.overall_height())
}
