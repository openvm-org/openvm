use std::io::{self, Write};

use openvm_stark_backend::{
    codec::{DecodableConfig, EncodableConfig},
    p3_util::log2_strict_usize,
};
use p3_field::Field;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::instrument;

use crate::{
    arch::{hasher::Hasher, MemoryCellType, ADDR_SPACE_OFFSET, U16_CELL_SIZE},
    system::memory::{dimensions::MemoryDimensions, online::LinearMemory, MemoryImage},
};

pub const PUBLIC_VALUES_AS: u32 = 3;
pub const PUBLIC_VALUES_ADDRESS_SPACE_OFFSET: u32 = PUBLIC_VALUES_AS - ADDR_SPACE_OFFSET;

/// Number of u16-celled storage slots required to hold `num_public_values`
/// bytes. Rounds up so an odd byte count still fits (the trailing high byte
/// is zero-padded by the producer).
#[inline(always)]
pub const fn pv_cell_count(num_public_values: usize) -> usize {
    num_public_values.div_ceil(U16_CELL_SIZE)
}

/// Number of cell field elements in the merkle-commit shape: `pv_cell_count`
/// rounded up to a power-of-two number of `DIGEST_WIDTH`-sized leaves. Real
/// cells fill the first `pv_cell_count` slots; the rest are zero padding so
/// the merkle layout is always a full binary tree.
#[inline(always)]
pub const fn pv_commit_cell_count(num_public_values: usize, digest_width: usize) -> usize {
    let cells = pv_cell_count(num_public_values);
    let chunks = cells.div_ceil(digest_width);
    chunks.next_power_of_two() * digest_width
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
    /// Assumption:
    /// - `num_public_values` is a power of two * DIGEST_WIDTH. It cannot be 0.
    /// - `top_tree` is 0-indexed and a segment tree of length `2 * 2^addr_space_height - 1`.
    #[instrument(name = "compute_user_public_values_proof", skip_all)]
    pub fn compute(
        memory_dimensions: MemoryDimensions,
        num_public_values: usize,
        hasher: &(impl Hasher<DIGEST_WIDTH, F> + Sync),
        final_memory: &MemoryImage,
        top_tree: &[[F; DIGEST_WIDTH]],
    ) -> Self {
        // `public_values` is the merkle leaves: `pv_commit_cell_count` u16
        // cells lifted to F via little-endian decode. Real cells fill the
        // first `pv_cell_count` slots; trailing cells are zero padding so the
        // merkle layout is always a full binary tree.
        let public_values =
            extract_public_value_cells::<DIGEST_WIDTH, F>(num_public_values, final_memory);
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
    // PUBLIC_VALUES_AS is u16-celled; the merkle layout is over packed
    // u16 cells, padded out to a power-of-two number of `DIGEST_WIDTH` leaves.
    let pv_commit_cells = pv_commit_cell_count(num_public_values, DIGEST_WIDTH);
    let num_pv_chunks: usize = pv_commit_cells / DIGEST_WIDTH;
    // This enforces the number of public values cannot be 0.
    assert!(
        num_pv_chunks.is_power_of_two(),
        "pv_height must be a power of two"
    );
    let pv_height = log2_strict_usize(num_pv_chunks);
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

/// Extracts the `num_public_values` user-facing bytes from
/// `PUBLIC_VALUES_AS`. Storage is u16-celled in little-endian byte layout, so
/// `as_slice()` already yields the bytes in user order; we truncate to the
/// caller-requested length.
pub fn extract_public_values(num_public_values: usize, final_memory: &MemoryImage) -> Vec<u8> {
    let mut public_values: Vec<u8> = {
        assert_eq!(
            final_memory.config[PUBLIC_VALUES_AS as usize].layout,
            MemoryCellType::U16
        );
        final_memory.mem[PUBLIC_VALUES_AS as usize]
            .as_slice()
            .to_vec()
    };

    let byte_capacity = public_values.len();
    assert!(
        byte_capacity >= num_public_values,
        "Public values address space has {} bytes of storage, but configuration has num_public_values={}",
        byte_capacity,
        num_public_values
    );
    public_values.truncate(num_public_values);
    public_values
}

/// Reads `pv_commit_cell_count` u16 cells from `PUBLIC_VALUES_AS` and lifts
/// them to F via the configured cell layout. The first `pv_cell_count` slots
/// are real packed cells; the rest are zero padding required by the merkle
/// commit shape.
fn extract_public_value_cells<const DIGEST_WIDTH: usize, F: Field>(
    num_public_values: usize,
    final_memory: &MemoryImage,
) -> Vec<F> {
    assert_eq!(
        final_memory.config[PUBLIC_VALUES_AS as usize].layout,
        MemoryCellType::U16
    );
    let storage = final_memory.mem[PUBLIC_VALUES_AS as usize].as_slice();
    let cells = pv_cell_count(num_public_values);
    let commit_cells = pv_commit_cell_count(num_public_values, DIGEST_WIDTH);
    let byte_capacity = storage.len();
    assert!(
        byte_capacity >= cells * U16_CELL_SIZE,
        "Public values storage has {} bytes, but {} u16 cells ({} bytes) are required",
        byte_capacity,
        cells,
        cells * U16_CELL_SIZE
    );
    let mut public_values = Vec::with_capacity(commit_cells);
    for i in 0..cells {
        let lo = storage[i * U16_CELL_SIZE];
        let hi = storage[i * U16_CELL_SIZE + 1];
        public_values.push(F::from_u16(u16::from_le_bytes([lo, hi])));
    }
    public_values.resize(commit_cells, F::ZERO);
    public_values
}

#[cfg(test)]
mod tests {
    use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::UserPublicValuesProof;
    use crate::{
        arch::{hasher::poseidon2::vm_poseidon2_hasher, MemoryConfig, SystemConfig},
        system::memory::{
            merkle::{public_values::PUBLIC_VALUES_AS, tree::MerkleTree},
            online::GuestMemory,
            AddressMap, DIGEST_WIDTH,
        },
    };

    type F = BabyBear;
    #[test]
    fn test_public_value_happy_path() {
        let mut vm_config = SystemConfig::default();
        let addr_space_height = 4;
        vm_config.memory_config.addr_space_height = addr_space_height;
        vm_config.memory_config.pointer_max_bits = 5;
        let memory_dimensions = vm_config.memory_config.memory_dimensions();
        // 16 bytes of public values = 8 packed u16 cells = 1 merkle leaf
        // (DIGEST_WIDTH = 8 cells), already a power-of-two leaf count.
        let num_public_values = 16;
        let mut addr_spaces_config = MemoryConfig::empty_address_space_configs(4);
        addr_spaces_config[PUBLIC_VALUES_AS as usize].num_cells =
            super::pv_cell_count(num_public_values);
        let mut memory = GuestMemory {
            memory: AddressMap::new(addr_spaces_config),
        };
        // Write the byte sequence [0, 0, 0, 1] at byte_ptr = 12. With u16 LE
        // storage this corresponds to u16 cells at cell_idx 6 = {0x0000} and
        // cell_idx 7 = {0x0100}, so when lifted to F the last cell is
        // F::from_u16(0x0100) = F::from_u16(256).
        unsafe {
            memory.write::<u8, 4>(PUBLIC_VALUES_AS, 12, [0, 0, 0, 1]);
        }
        let mut expected_pvs =
            F::zero_vec(super::pv_commit_cell_count(num_public_values, DIGEST_WIDTH));
        expected_pvs[7] = F::from_u16(0x0100);

        let hasher = vm_poseidon2_hasher();
        let tree = MerkleTree::from_memory(&memory.memory, &memory_dimensions, &hasher);
        let top_tree = tree.top_tree(addr_space_height);
        let pv_proof = UserPublicValuesProof::<{ DIGEST_WIDTH }, F>::compute(
            memory_dimensions,
            num_public_values,
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
}
