//! Constants shared by metered execution and native RVR tracing.

/// Number of Merkle-tree levels represented by one `u64` metering page mask.
pub const PAGE_MASK_LEAF_BITS: usize = u64::BITS.ilog2() as usize;

/// Maximum number of instructions in one generated metered RVR block.
///
/// Segment checks happen at block boundaries. This must not exceed
/// [`SEGMENT_CHECK_INSNS`], so a freshly reset check interval can accommodate
/// every generated block.
pub const MAX_METERED_BLOCK_INSNS: u32 = 1000;

/// Interval between metered-execution segment checks.
///
/// The tracer checks before executing a block that would cross the remaining
/// interval. It therefore never starts a block with an insufficient countdown.
pub const SEGMENT_CHECK_INSNS: u32 = 1000;

const _: () = assert!(SEGMENT_CHECK_INSNS >= MAX_METERED_BLOCK_INSNS);
