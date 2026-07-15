use std::io::{self, Write};

use openvm_instructions::PUBLIC_VALUES_AS;
use openvm_stark_backend::{
    codec::{DecodableConfig, EncodableConfig},
    p3_util::log2_strict_usize,
};
use p3_field::Field;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::instrument;

use crate::{
    arch::{hasher::Hasher, MemoryCellType, SystemConfig, ADDR_SPACE_OFFSET, U16_CELL_SIZE},
    system::memory::{dimensions::MemoryDimensions, online::LinearMemory, MemoryImage},
};
pub const PUBLIC_VALUES_ADDRESS_SPACE_OFFSET: u32 = PUBLIC_VALUES_AS - ADDR_SPACE_OFFSET;

pub const fn public_values_cells_from_bytes(num_public_values_bytes: usize) -> usize {
    assert!(
        num_public_values_bytes.is_multiple_of(U16_CELL_SIZE),
        "num_public_values_bytes must be a multiple of U16_CELL_SIZE"
    );
    num_public_values_bytes / U16_CELL_SIZE
}

/// Validates that public values occupy a power-of-two number of complete merkle leaves.
#[inline(always)]
pub(crate) const fn assert_public_values_shape<const DIGEST_WIDTH: usize>(
    num_public_values: usize,
) {
    assert!(
        num_public_values.is_multiple_of(DIGEST_WIDTH),
        "num_public_values must be a multiple of DIGEST_WIDTH"
    );
    assert!(
        (num_public_values / DIGEST_WIDTH).is_power_of_two(),
        "public values merkle leaf count must be a power of two"
    );
}

/// Merkle proof for user public values in the memory state.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, [F; DIGEST_WIDTH]: Serialize",
    deserialize = "F: Deserialize<'de>, [F; DIGEST_WIDTH]: Deserialize<'de>"
))]
pub struct UserPublicValuesProof<const DIGEST_WIDTH: usize, F> {
    /// Proof of the path from the root of public values to the memory root in the format of
    /// sequence of sibling node hashes.
    pub proof: Vec<[F; DIGEST_WIDTH]>,
    /// Raw public values. Its length should be (a power of two) * DIGEST_WIDTH.
    pub public_values: Vec<F>,
    /// Merkle root of public values. The computation of this value follows the same logic of
    /// `MemoryNode`. The merkle tree doesn't pad because the length `public_values` implies the
    /// merkle tree is always a full binary tree.
    pub public_values_commit: [F; DIGEST_WIDTH],
}

#[derive(Error, Debug)]
pub enum UserPublicValuesProofError {
    #[error("unexpected length: {0}")]
    UnexpectedLength(usize),
    #[error("incorrect proof length: {0} (expected {1})")]
    IncorrectProofLength(usize, usize),
    #[error("user public values do not match commitment")]
    UserPublicValuesCommitMismatch,
    #[error("final memory root mismatch")]
    FinalMemoryRootMismatch,
}

impl<const DIGEST_WIDTH: usize, F: Field> UserPublicValuesProof<DIGEST_WIDTH, F> {
    /// Computes the proof of the public values from the final memory state and the Merkle top
    /// sub-tree of address space roots. This function will re-compute the empty merkle roots of
    /// each height `0..=address_height` internally.
    ///
    /// Memory dimensions and the public-values cell count are read off `system_config` (the
    /// public-values shape was validated when the config was constructed), so this signature
    /// makes it impossible to pass a wrong byte/cell length by accident.
    ///
    /// `top_tree` is 0-indexed and a segment tree of length `2 * 2^addr_space_height - 1`.
    #[instrument(name = "compute_user_public_values_proof", skip_all)]
    pub fn compute(
        system_config: &SystemConfig,
        hasher: &(impl Hasher<DIGEST_WIDTH, F> + Sync),
        final_memory: &MemoryImage,
        top_tree: &[[F; DIGEST_WIDTH]],
    ) -> Self {
        let memory_dimensions = system_config.memory_config.memory_dimensions();
        let num_public_values = system_config.num_public_values;
        let public_values = extract_public_value_cells(num_public_values, final_memory);
        let public_values_commit = hasher.merkle_root(&public_values);
        let proof = compute_merkle_proof_to_user_public_values_root(
            memory_dimensions,
            num_public_values,
            hasher,
            top_tree,
        );
        UserPublicValuesProof {
            proof,
            public_values,
            public_values_commit,
        }
    }

    pub fn verify(
        &self,
        hasher: &impl Hasher<DIGEST_WIDTH, F>,
        memory_dimensions: MemoryDimensions,
        final_memory_root: [F; DIGEST_WIDTH],
    ) -> Result<(), UserPublicValuesProofError> {
        // Verify user public values Merkle proof:
        // 0. Get correct indices for Merkle proof based on memory dimensions
        // 1. Verify user public values commitment with respect to the final memory root.
        // 2. Compare user public values commitment with Merkle root of user public values.
        let pv_commit = self.public_values_commit;
        // 0.
        let pv_as = PUBLIC_VALUES_AS;
        let pv_start_idx = memory_dimensions.label_to_index((pv_as, 0));
        let pvs = &self.public_values;
        if !pvs.len().is_multiple_of(DIGEST_WIDTH) || !(pvs.len() / DIGEST_WIDTH).is_power_of_two()
        {
            return Err(UserPublicValuesProofError::UnexpectedLength(pvs.len()));
        }
        let pv_height = log2_strict_usize(pvs.len() / DIGEST_WIDTH);
        let proof_len = memory_dimensions.overall_height() - pv_height;
        let idx_prefix = pv_start_idx >> pv_height;
        // 1.
        if self.proof.len() != proof_len {
            return Err(UserPublicValuesProofError::IncorrectProofLength(
                self.proof.len(),
                proof_len,
            ));
        }
        let mut curr_root = pv_commit;
        for (i, sibling_hash) in self.proof.iter().enumerate() {
            curr_root = if idx_prefix & (1 << i) != 0 {
                hasher.compress(sibling_hash, &curr_root)
            } else {
                hasher.compress(&curr_root, sibling_hash)
            }
        }
        if curr_root != final_memory_root {
            return Err(UserPublicValuesProofError::FinalMemoryRootMismatch);
        }
        // 2. Compute merkle root of public values
        if hasher.merkle_root(pvs) != pv_commit {
            return Err(UserPublicValuesProofError::UserPublicValuesCommitMismatch);
        }

        Ok(())
    }

    pub fn encode<SC: EncodableConfig<F = F, Digest = [F; DIGEST_WIDTH]>, W: Write>(
        &self,
        writer: &mut W,
    ) -> io::Result<()> {
        SC::encode_digest_slice(&self.proof, writer)?;
        SC::encode_base_field_slice(&self.public_values, writer)?;
        SC::encode_digest(&self.public_values_commit, writer)?;
        Ok(())
    }

    pub fn decode<SC: DecodableConfig<F = F, Digest = [F; DIGEST_WIDTH]>, R: io::Read>(
        reader: &mut R,
    ) -> io::Result<Self> {
        let proof = SC::decode_digest_vec(reader)?;
        let public_values = SC::decode_base_field_vec(reader)?;
        let public_values_commit = SC::decode_digest(reader)?;
        Ok(Self {
            proof,
            public_values,
            public_values_commit,
        })
    }
}

fn compute_merkle_proof_to_user_public_values_root<const DIGEST_WIDTH: usize, F: Field>(
    memory_dimensions: MemoryDimensions,
    num_public_values: usize,
    hasher: &(impl Hasher<DIGEST_WIDTH, F> + Sync),
    top_tree: &[[F; DIGEST_WIDTH]],
) -> Vec<[F; DIGEST_WIDTH]> {
    let address_height = memory_dimensions.address_height;
    let addr_space_height = memory_dimensions.addr_space_height;
    assert_eq!(top_tree.len(), (2 << addr_space_height) - 1);
    let num_pv_leaves: usize = num_public_values / DIGEST_WIDTH;
    let pv_height = log2_strict_usize(num_pv_leaves);
    let address_leading_zeros = address_height - pv_height;

    let mut cur_node_idx = 1; // root
    let mut proof = Vec::with_capacity(addr_space_height + address_leading_zeros);
    let zero_nodes: Vec<_> = (0..address_height)
        .scan(hasher.hash(&[F::ZERO; DIGEST_WIDTH]), |acc, _| {
            let result = Some(*acc);
            *acc = hasher.compress(acc, acc);
            result
        })
        .collect();
    for i in 0..addr_space_height {
        let bit = 1 << (memory_dimensions.addr_space_height - i - 1);
        // Recall: top_tree is 0-indexed, but cur_node_idx is 1-indexed
        if (PUBLIC_VALUES_AS - ADDR_SPACE_OFFSET) & bit != 0 {
            proof.push(top_tree[cur_node_idx * 2 - 1]);
            cur_node_idx = cur_node_idx * 2 + 1;
        } else {
            proof.push(top_tree[cur_node_idx * 2]);
            cur_node_idx *= 2;
        }
    }
    for i in 0..address_leading_zeros {
        // node is always on the left, the sibling is always zero node hash
        proof.push(zero_nodes[address_height - 1 - i]);
    }
    proof.reverse();
    proof
}

/// Extracts the first `num_public_values_bytes` bytes from `PUBLIC_VALUES_AS`.
pub fn extract_public_values(
    num_public_values_bytes: usize,
    final_memory: &MemoryImage,
) -> Vec<u8> {
    let mut public_values: Vec<u8> = {
        assert_eq!(
            final_memory.config[PUBLIC_VALUES_AS as usize].layout,
            MemoryCellType::U16
        );
        final_memory.mem[PUBLIC_VALUES_AS as usize]
            .as_slice()
            .to_vec()
    };

    assert!(
        public_values.len() >= num_public_values_bytes,
        "Public values address space has {} bytes of storage, but configuration has num_public_values_bytes={}",
        public_values.len(),
        num_public_values_bytes
    );
    public_values.truncate(num_public_values_bytes);
    public_values
}

fn extract_public_value_cells<F: Field>(
    num_public_values: usize,
    final_memory: &MemoryImage,
) -> Vec<F> {
    assert_eq!(
        final_memory.config[PUBLIC_VALUES_AS as usize].layout,
        MemoryCellType::U16
    );
    let storage = final_memory.mem[PUBLIC_VALUES_AS as usize].as_slice();
    assert!(
        storage.len() >= num_public_values * U16_CELL_SIZE,
        "Public values storage has {} bytes, but {} u16 cells ({} bytes) are required",
        storage.len(),
        num_public_values,
        num_public_values * U16_CELL_SIZE
    );
    storage
        .chunks_exact(U16_CELL_SIZE)
        .take(num_public_values)
        .map(|bytes| F::from_u16(u16::from_le_bytes([bytes[0], bytes[1]])))
        .collect()
}

#[cfg(test)]
mod tests {
    use openvm_instructions::{DIGEST_WIDTH, PUBLIC_VALUES_AS};
    use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::UserPublicValuesProof;
    use crate::{
        arch::{hasher::poseidon2::vm_poseidon2_hasher, MemoryConfig, SystemConfig},
        system::memory::{merkle::tree::MerkleTree, online::GuestMemory, AddressMap},
    };

    type F = BabyBear;
    #[test]
    fn test_public_value_happy_path() {
        let mut vm_config = SystemConfig::default();
        let addr_space_height = 4;
        vm_config.memory_config.addr_space_height = addr_space_height;
        vm_config.memory_config.pointer_max_bits = 5;
        let memory_dimensions = vm_config.memory_config.memory_dimensions();
        let num_public_values = DIGEST_WIDTH;
        let vm_config = vm_config.with_public_values(num_public_values);
        let mut addr_spaces_config = MemoryConfig::empty_address_space_configs(4);
        addr_spaces_config[PUBLIC_VALUES_AS as usize].num_cells = num_public_values;
        let mut memory = GuestMemory {
            memory: AddressMap::new(addr_spaces_config),
        };
        // Write the byte sequence [0, 0, 0, 1] at byte_ptr = 12. With u16 LE
        // storage this corresponds to u16 cells at ptr 6 = {0x0000} and
        // ptr 7 = {0x0100}, so when lifted to F the last cell is
        // F::from_u16(0x0100) = F::from_u16(256).
        unsafe {
            memory.write_bytes::<4>(PUBLIC_VALUES_AS, 12, [0, 0, 0, 1]);
        }
        let mut expected_pvs = F::zero_vec(num_public_values);
        expected_pvs[7] = F::from_u16(0x0100);

        let hasher = vm_poseidon2_hasher();
        let tree = MerkleTree::from_memory(&memory.memory, &memory_dimensions, &hasher);
        let top_tree = tree.top_tree(addr_space_height);
        let pv_proof = UserPublicValuesProof::<{ DIGEST_WIDTH }, F>::compute(
            &vm_config,
            &hasher,
            &memory.memory,
            &top_tree,
        );
        assert_eq!(pv_proof.public_values, expected_pvs);
        let final_memory_root =
            MerkleTree::from_memory(&memory.memory, &memory_dimensions, &hasher).root();
        pv_proof
            .verify(&hasher, memory_dimensions, final_memory_root)
            .unwrap();
    }

    #[test]
    #[should_panic]
    fn test_public_values_write_beyond_num_public_values_is_rejected() {
        let num_public_values = 16;
        let mut addr_spaces_config = MemoryConfig::empty_address_space_configs(4);
        addr_spaces_config[PUBLIC_VALUES_AS as usize].num_cells = num_public_values;
        let mut memory = GuestMemory {
            memory: AddressMap::new(addr_spaces_config),
        };
        unsafe {
            memory.write::<u8, 4>(PUBLIC_VALUES_AS, num_public_values as u32 + 4, [0, 0, 0, 1]);
        }
    }
}
