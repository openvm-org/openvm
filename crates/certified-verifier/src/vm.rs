//! Wire encoders for the VM verifier's host-side inputs.

use std::io::{Error, ErrorKind, Result, Write};

use openvm_circuit::system::memory::merkle::public_values::UserPublicValuesProof;
use openvm_stark_backend::codec::EncodableConfig;
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as SC, DIGEST_SIZE, F,
};
use openvm_verify_stark_host::{pvs::VkCommit, vk::VerificationBaseline};

use crate::{
    magic::{write_header, MAGIC_USER_PUBLIC_VALUES, MAGIC_VM_BASELINE},
    primitives::write_usize_as_u32,
};

fn write_vk_commit<W: Write>(writer: &mut W, commit: &VkCommit<F>) -> Result<()> {
    SC::encode_digest(&commit.cached_commit, writer)?;
    SC::encode_digest(&commit.vk_pre_hash, writer)
}

/// Encode the non-deferral VM verification baseline consumed by Lean.
pub fn write_verification_baseline<W: Write>(
    writer: &mut W,
    baseline: &VerificationBaseline,
) -> Result<()> {
    if baseline.expected_def_hook_commit.is_some() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "the certified VM verifier does not support deferral proofs",
        ));
    }

    write_header(writer, MAGIC_VM_BASELINE)?;
    SC::encode_digest(&baseline.app_exe_commit, writer)?;
    write_usize_as_u32(writer, baseline.memory_dimensions.addr_space_height)?;
    write_usize_as_u32(writer, baseline.memory_dimensions.address_height)?;
    write_usize_as_u32(writer, baseline.num_user_pvs)?;
    write_vk_commit(writer, &baseline.app_vk_commit)?;
    write_vk_commit(writer, &baseline.leaf_vk_commit)?;
    write_vk_commit(writer, &baseline.internal_for_leaf_vk_commit)?;
    write_vk_commit(writer, &baseline.internal_recursive_vk_commit)
}

/// Encode the Merkle proof and values committed by the VM's public-values region.
pub fn write_user_public_values_proof<W: Write>(
    writer: &mut W,
    proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
) -> Result<()> {
    write_header(writer, MAGIC_USER_PUBLIC_VALUES)?;
    SC::encode_digest_slice(&proof.proof, writer)?;
    SC::encode_base_field_slice(&proof.public_values, writer)?;
    SC::encode_digest(&proof.public_values_commit, writer)
}
