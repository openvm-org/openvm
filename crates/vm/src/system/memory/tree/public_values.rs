use std::{collections::BTreeMap, sync::Arc};

use openvm_stark_backend::{p3_field::PrimeField32, p3_util::log2_strict_usize};
use serde::{Deserialize, Serialize};

use crate::{
    arch::hasher::Hasher,
    system::memory::{
        dimensions::MemoryDimensions, paged_vec::Address, tree::MemoryNode, MemoryImage,
    },
};

pub const PUBLIC_VALUES_ADDRESS_SPACE_OFFSET: u32 = 2;

/// Merkle proof for user public values in the memory state.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, [F; CHUNK]: Serialize",
    deserialize = "F: Deserialize<'de>, [F; CHUNK]: Deserialize<'de>"
))]
pub struct UserPublicValuesProof<const CHUNK: usize, F> {
    /// Proof of the path from the root of public values to the memory root in the format of (`bit`, `hash`)
    /// `bit`: If `bit` is true, public values are in the left child, otherwise in the right child.
    /// `hash`: Hash of the sibling node.
    pub proof: Vec<(bool, [F; CHUNK])>,
    /// Raw public values. Its length should be a power of two * CHUNK.
    pub public_values: Vec<F>,
    /// Merkle root of public values. The computation of this value follows the same logic of
    /// `MemoryNode`. The merkle tree doesn't pad because the length `public_values` implies the
    /// merkle tree is always a full binary tree.
    pub public_values_commit: [F; CHUNK],
}

impl<const CHUNK: usize, F: PrimeField32> UserPublicValuesProof<CHUNK, F> {
    /// Computes the proof of the public values from the final memory state.
    /// Assumption:
    /// - `num_public_values` is a power of two * CHUNK. It cannot be 0.
    pub fn compute(
        memory_dimensions: MemoryDimensions,
        num_public_values: usize,
        hasher: &(impl Hasher<CHUNK, F> + Sync),
        final_memory: &MemoryImage<F>,
    ) -> Self {
        let proof = compute_merkle_proof_to_user_public_values_root(
            memory_dimensions,
            num_public_values,
            hasher,
            final_memory,
        );
        let public_values =
            extract_public_values(&memory_dimensions, num_public_values, final_memory);
        let public_values_commit = hasher.merkle_root(&public_values);
        UserPublicValuesProof {
            proof,
            public_values,
            public_values_commit,
        }
    }
}

fn compute_merkle_proof_to_user_public_values_root<const CHUNK: usize, F: PrimeField32>(
    memory_dimensions: MemoryDimensions,
    num_public_values: usize,
    hasher: &(impl Hasher<CHUNK, F> + Sync),
    final_memory: &MemoryImage<F>,
) -> Vec<(bool, [F; CHUNK])> {
    assert_eq!(
        num_public_values % CHUNK,
        0,
        "num_public_values must be a multiple of memory chunk {CHUNK}"
    );
    let root = MemoryNode::tree_from_memory(memory_dimensions, final_memory, hasher);
    let num_pv_chunks: usize = num_public_values / CHUNK;
    // This enforces the number of public values cannot be 0.
    assert!(
        num_pv_chunks.is_power_of_two(),
        "pv_height must be a power of two"
    );
    let pv_height = log2_strict_usize(num_pv_chunks);
    let address_leading_zeros = memory_dimensions.address_height - pv_height;

    let mut curr_node = Arc::new(root);
    let mut proof = Vec::with_capacity(memory_dimensions.as_height + address_leading_zeros);
    for i in 0..memory_dimensions.as_height {
        let bit = 1 << (memory_dimensions.as_height - i - 1);
        if let MemoryNode::NonLeaf { left, right, .. } = curr_node.as_ref().clone() {
            if PUBLIC_VALUES_ADDRESS_SPACE_OFFSET & bit != 0 {
                curr_node = right;
                proof.push((true, left.hash()));
            } else {
                curr_node = left;
                proof.push((false, right.hash()));
            }
        } else {
            unreachable!()
        }
    }
    for _ in 0..address_leading_zeros {
        if let MemoryNode::NonLeaf { left, right, .. } = curr_node.as_ref().clone() {
            curr_node = left;
            proof.push((false, right.hash()));
        } else {
            unreachable!()
        }
    }
    proof.reverse();
    proof
}

pub fn extract_public_values<F: PrimeField32>(
    memory_dimensions: &MemoryDimensions,
    num_public_values: usize,
    final_memory: &MemoryImage<F>,
) -> Vec<F> {
    // All (addr, value) pairs in the public value address space.
    let f_as_start = PUBLIC_VALUES_ADDRESS_SPACE_OFFSET + memory_dimensions.as_offset;
    let f_as_end = PUBLIC_VALUES_ADDRESS_SPACE_OFFSET + memory_dimensions.as_offset + 1;

    // This clones the entire memory. Ideally this should run in time proportional to
    // the size of the PV address space, not entire memory.
    let final_memory: BTreeMap<Address, F> = final_memory.items().collect();

    let used_pvs: Vec<_> = final_memory
        .range((f_as_start, 0)..(f_as_end, 0))
        .map(|(&(_, pointer), &value)| (pointer as usize, value))
        .collect();
    if let Some(&last_pv) = used_pvs.last() {
        assert!(
            last_pv.0 < num_public_values || last_pv.1 == F::ZERO,
            "Last public value is out of bounds"
        );
    }
    let mut public_values = F::zero_vec(num_public_values);
    for (i, pv) in used_pvs {
        if i < num_public_values {
            public_values[i] = pv;
        }
    }
    public_values
}

#[cfg(test)]
mod tests {
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::{UserPublicValuesProof, PUBLIC_VALUES_ADDRESS_SPACE_OFFSET};
    use crate::{
        arch::{
            hasher::{poseidon2::vm_poseidon2_hasher, Hasher},
            SystemConfig,
        },
        system::memory::{paged_vec::AddressMap, tree::MemoryNode, CHUNK},
    };

    type F = BabyBear;
    #[test]
    fn test_public_value_happy_path() {
        let mut vm_config = SystemConfig::default();
        vm_config.memory_config.as_height = 4;
        vm_config.memory_config.pointer_max_bits = 5;
        let memory_dimensions = vm_config.memory_config.memory_dimensions();
        let pv_as = PUBLIC_VALUES_ADDRESS_SPACE_OFFSET + memory_dimensions.as_offset;
        let num_public_values = 16;
        let memory = AddressMap::from_iter(
            memory_dimensions.as_offset,
            1 << memory_dimensions.as_height,
            1 << memory_dimensions.address_height,
            [((pv_as, 15), F::ONE)],
        );
        let mut expected_pvs = F::zero_vec(num_public_values);
        expected_pvs[15] = F::ONE;

        let hasher = vm_poseidon2_hasher();
        let pv_proof = UserPublicValuesProof::<{ CHUNK }, F>::compute(
            memory_dimensions,
            num_public_values,
            &hasher,
            &memory,
        );
        assert_eq!(pv_proof.public_values, expected_pvs);
        let final_memory_root = MemoryNode::tree_from_memory(memory_dimensions, &memory, &hasher);
        let mut curr_root = pv_proof.public_values_commit;
        for (is_right, sibling_hash) in &pv_proof.proof {
            curr_root = if *is_right {
                hasher.compress(sibling_hash, &curr_root)
            } else {
                hasher.compress(&curr_root, sibling_hash)
            }
        }
        assert_eq!(curr_root, final_memory_root.hash());
    }
}
