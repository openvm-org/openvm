//! Constants shared by metered execution and native RVR tracing.

/// Number of Merkle-tree levels represented by one `u64` metering page mask.
pub const PAGE_MASK_LEAF_BITS_U32: u32 = u64::BITS.ilog2();
/// `usize` form for shifts, ranges, and indexing.
pub const PAGE_MASK_LEAF_BITS: usize = PAGE_MASK_LEAF_BITS_U32 as usize;

/// Maximum number of instructions in a generated metered RVR block.
pub const MAX_METERED_BLOCK_INSNS: u32 = 1000;

/// Interval between metered-execution segment checks.
pub const SEGMENT_CHECK_INSNS: u32 = 1000;

const _: () = assert!(SEGMENT_CHECK_INSNS >= MAX_METERED_BLOCK_INSNS);
