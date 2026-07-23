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

    // indicate whether `expand_direction` is different from origin
    // when `expand_direction` != -1, must be 0
    pub left_direction_different: T,
    pub right_direction_different: T,

    // Reference-count adjustments for the child interactions of *initial* rows
    // (`expand_direction` = 1): the child's initial claim is consumed `1 + adj` times.
    //
    // In {-1, 0, 1}; both must be 0 when `expand_direction` != 1.
    //  `+1` = this row's final counterpart dd-borrows the child's initial hash, so this
    //         row consumes the child's initial claim twice (multiplicity -2).
    //  `-1` = the child is untouched and this node has no final row to prop the
    //         reference, so this row consumes nothing (multiplicity 0).
    pub left_adj_ref: T,
    pub right_adj_ref: T,
}

#[derive(Debug, Clone, Copy, AlignedBorrow, StructReflection)]
#[repr(C)]
pub struct MemoryMerklePvs<T, const DIGEST_WIDTH: usize> {
    /// The memory state root before the execution of this segment.
    pub initial_root: [T; DIGEST_WIDTH],
    /// The memory state root after the execution of this segment.
    pub final_root: [T; DIGEST_WIDTH],
}
