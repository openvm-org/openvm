//! Constants shared by metered execution and native RVR tracing.

/// Number of Merkle-tree levels represented by one `u64` metering page mask.
pub const PAGE_MASK_LEAF_BITS: usize = u64::BITS.ilog2() as usize;

/// Maximum number of instructions in one generated metered RVR block.
///
/// Segment checks happen only at block boundaries, so tracer buffer-capacity
/// bounds must accommodate this many instructions beyond a check interval.
pub const MAX_METERED_BLOCK_INSNS: u32 = 1000;

/// Default interval between metered-execution segment checks.
///
/// This currently matches [`MAX_METERED_BLOCK_INSNS`] so the default check
/// interval and generated-block granularity share one source of truth. Buffer
/// capacities must account for both values because a check can overshoot by
/// one complete block.
pub const DEFAULT_SEGMENT_CHECK_INSNS: u32 = MAX_METERED_BLOCK_INSNS;
