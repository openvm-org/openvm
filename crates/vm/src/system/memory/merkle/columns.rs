use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;

#[derive(Debug, AlignedBorrow, StructReflection)]
#[repr(C)]
pub struct MemoryMerkleCols<T, const DIGEST_WIDTH: usize> {
    // `expand_direction` =  1 corresponds to initial memory state
    // `expand_direction` = -1 corresponds to final memory state
    // `expand_direction` =  0 corresponds to irrelevant row (all interactions multiplicity 0)
    pub expand_direction: T,

    // height_section = 1 indicates that as_label is being expanded
    // height_section = 0 indicates that address_label is being expanded
    pub height_section: T,
    pub parent_height: T,
    pub parent_height_inv: T,
    pub is_root: T,

    pub parent_as_label: T,
    pub parent_address_label: T,

    pub parent_hash: [T; DIGEST_WIDTH],
    pub left_child_hash: [T; DIGEST_WIDTH],
    pub right_child_hash: [T; DIGEST_WIDTH],

    // Each child has a mode in {0, 1, 2}. Its meaning depends on `expand_direction`.
    //
    // +---------+------------------+------------+------------------------------------+
    // | Row     | expand_direction | mode       | Child interaction                  |
    // +---------+------------------+------------+------------------------------------+
    // | Initial |                1 | 0, 1, or 2 | Initial multiplicity is `-mode`    |
    // | Final   |               -1 |          0 | Final multiplicity is `+1`         |
    // | Final   |               -1 |          1 | Initial multiplicity is `+1`       |
    // | Padding |                0 |          0 | None                               |
    // +---------+------------------+------------+------------------------------------+
    //
    // Initial row (`expand_direction` = 1): the negative multiplicity of the child's
    // initial interaction — one for a touched child, plus one when this node's final row
    // uses the child's initial hash. An untouched child of a node with no final row has
    // mode 0.
    //
    // Final row (`expand_direction` = -1): mode 1 borrows the child from the initial
    // tree (untouched or touched-clean), while mode 0 uses the final child. The
    // multiplicity is `+1` in either case.
    //
    // Padding row (`expand_direction` = 0): must be 0.
    pub left_child_mode: T,
    pub right_child_mode: T,
}

#[derive(Debug, Clone, Copy, AlignedBorrow, StructReflection)]
#[repr(C)]
pub struct MemoryMerklePvs<T, const DIGEST_WIDTH: usize> {
    /// The memory state root before the execution of this segment.
    pub initial_root: [T; DIGEST_WIDTH],
    /// The memory state root after the execution of this segment.
    pub final_root: [T; DIGEST_WIDTH],
}
