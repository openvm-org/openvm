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
    // (`expand_direction` = 1); all four must be 0 when `expand_direction` != 1.
    //
    // `*_extra_ref` = 1 means this row's final counterpart dd-borrows the child's initial
    // hash, so this row consumes the child's initial claim twice (multiplicity -2).
    // `*_absent_ref` = 1 means the child is untouched and this node has no final row to
    // prop the reference, so this row consumes nothing (multiplicity 0).
    pub left_extra_ref: T,
    pub right_extra_ref: T,
    pub left_absent_ref: T,
    pub right_absent_ref: T,
}

#[derive(Debug, Clone, Copy, AlignedBorrow, StructReflection)]
#[repr(C)]
pub struct MemoryMerklePvs<T, const DIGEST_WIDTH: usize> {
    /// The memory state root before the execution of this segment.
    pub initial_root: [T; DIGEST_WIDTH],
    /// The memory state root after the execution of this segment.
    pub final_root: [T; DIGEST_WIDTH],
}
