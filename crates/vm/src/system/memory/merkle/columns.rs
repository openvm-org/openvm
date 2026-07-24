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

    // One child-reference descriptor per side, in {0, 1, 2}. Its meaning depends on
    // `expand_direction`
    //
    // Initial row (`expand_direction` = 1): number of times this row consumes the child's
    // *initial* claim (count `-mode`) — one copy for a touched child, plus one more when
    // this node's final row dd-borrows the child's initial hash. An untouched child of a
    // node that emits no final row consumes nothing (`mode` = 0).
    //
    // Final row (`expand_direction` = -1): the "direction different" bit in {0, 1} —
    // 1 iff the child is borrowed from the initial tree (untouched or touched-clean)
    // rather than expanded as the final child. The count is `+1` regardless.
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
