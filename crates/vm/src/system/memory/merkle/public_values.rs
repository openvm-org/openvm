use itertools::Itertools;
use openvm_stark_backend::{p3_field::PrimeField32, p3_util::log2_strict_usize};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::instrument;

use crate::{
    arch::{hasher::Hasher, MemoryCellType, ADDR_SPACE_OFFSET},
    system::memory::{dimensions::MemoryDimensions, online::LinearMemory, MemoryImage},
};

pub const PUBLIC_VALUES_AS: u32 = 3;
pub const PUBLIC_VALUES_ADDRESS_SPACE_OFFSET: u32 = PUBLIC_VALUES_AS - ADDR_SPACE_OFFSET;

/// Merkle proof for user public values in the memory state.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, [F; CHUNK]: Serialize",
    deserialize = "F: Deserialize<'de>, [F; CHUNK]: Deserialize<'de>"
))]
pub struct UserPublicValuesProof<const CHUNK: usize, F> {
    /// Proof of the path from the root of public values to the memory root in the format of
    /// sequence of sibling node hashes.
    pub proof: Vec<[F; CHUNK]>,
    /// Raw public values. Its length should be (a power of two) * CHUNK.
    pub public_values: Vec<F>,
    /// Merkle root of public values. The computation of this value follows the same logic of
    /// `MemoryNode`. The merkle tree doesn't pad because the length `public_values` implies the
    /// merkle tree is always a full binary tree.
    pub public_values_commit: [F; CHUNK],
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

impl<const CHUNK: usize, F: PrimeField32> UserPublicValuesProof<CHUNK, F> {
    /// Computes the proof of the public values from the final memory state and the Merkle top
    /// sub-tree of address space roots. This function will re-compute the empty merkle roots of
    /// each height `0..=address_height` internally.
    ///
    /// Assumption:
    /// - `num_public_values` is a power of two * CHUNK. It cannot be 0.
    /// - `top_tree` is 0-indexed and a segment tree of length `2 * 2^addr_space_height - 1`.
    #[instrument(name = "compute_user_public_values_proof", skip_all)]
    pub fn compute(
        memory_dimensions: MemoryDimensions,
        num_public_values: usize,
        hasher: &(impl Hasher<CHUNK, F> + Sync),
        final_memory: &MemoryImage,
        top_tree: &[[F; CHUNK]],
    ) -> Self {
        let public_values = extract_public_values(num_public_values, final_memory)
            .iter()
            .map(|&x| F::from_canonical_u8(x))
            .collect_vec();
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
        hasher: &impl Hasher<CHUNK, F>,
        memory_dimensions: MemoryDimensions,
        final_memory_root: [F; CHUNK],
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
        if pvs.len() % CHUNK != 0 || !(pvs.len() / CHUNK).is_power_of_two() {
            return Err(UserPublicValuesProofError::UnexpectedLength(pvs.len()));
        }
        let pv_height = log2_strict_usize(pvs.len() / CHUNK);
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
}

fn compute_merkle_proof_to_user_public_values_root<const CHUNK: usize, F: PrimeField32>(
    memory_dimensions: MemoryDimensions,
    num_public_values: usize,
    hasher: &(impl Hasher<CHUNK, F> + Sync),
    top_tree: &[[F; CHUNK]],
) -> Vec<[F; CHUNK]> {
    assert_eq!(
        num_public_values % CHUNK,
        0,
        "num_public_values must be a multiple of memory chunk {CHUNK}"
    );
    let address_height = memory_dimensions.address_height;
    let addr_space_height = memory_dimensions.addr_space_height;
    assert_eq!(top_tree.len(), (2 << addr_space_height) - 1);
    let num_pv_chunks: usize = num_public_values / CHUNK;
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
        .scan(hasher.hash(&[F::ZERO; CHUNK]), |acc, _| {
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

pub fn extract_public_values(num_public_values: usize, final_memory: &MemoryImage) -> Vec<u8> {
    let mut public_values: Vec<u8> = {
        assert_eq!(
            final_memory.config[PUBLIC_VALUES_AS as usize].layout,
            MemoryCellType::U8
        );
        final_memory.mem[PUBLIC_VALUES_AS as usize]
            .as_slice()
            .to_vec()
    };

    assert!(
        public_values.len() >= num_public_values,
        "Public values address space has {} elements, but configuration has num_public_values={}",
        public_values.len(),
        num_public_values
    );
    public_values.truncate(num_public_values);
    public_values
}

#[cfg(test)]
mod tests {
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::UserPublicValuesProof;
    use crate::{
        arch::{hasher::poseidon2::vm_poseidon2_hasher, MemoryConfig, SystemConfig},
        system::memory::{
            merkle::{public_values::PUBLIC_VALUES_AS, tree::MerkleTree},
            online::GuestMemory,
            AddressMap, CHUNK,
        },
    };

    type F = BabyBear;
    #[test]
    fn test_public_value_happy_path() {
        let mut vm_config = SystemConfig::default().without_continuations();
        let addr_space_height = 4;
        vm_config.memory_config.addr_space_height = addr_space_height;
        vm_config.memory_config.pointer_max_bits = 5;
        let memory_dimensions = vm_config.memory_config.memory_dimensions();
        let num_public_values = 16;
        let mut addr_spaces_config = MemoryConfig::empty_address_space_configs(4);
        addr_spaces_config[PUBLIC_VALUES_AS as usize].num_cells = num_public_values;
        let mut memory = GuestMemory {
            memory: AddressMap::new(addr_spaces_config),
        };
        unsafe {
            memory.write::<u8, 4>(PUBLIC_VALUES_AS, 12, [0, 0, 0, 1]);
        }
        let mut expected_pvs = F::zero_vec(num_public_values);
        expected_pvs[15] = F::ONE;

        let hasher = vm_poseidon2_hasher();
        let tree = MerkleTree::from_memory(&memory.memory, &memory_dimensions, &hasher);
        let top_tree = tree.top_tree(addr_space_height);
        let pv_proof = UserPublicValuesProof::<{ CHUNK }, F>::compute(
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
