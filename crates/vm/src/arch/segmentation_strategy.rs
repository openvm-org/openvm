use std::sync::Arc;

pub const DEFAULT_MAX_SEGMENT_LEN: usize = (1 << 22) - 100;
// a heuristic number for the maximum number of cells per chip in a segment
// a few reasons for this number:
//  1. `VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8>` is
//    the chip with the most cells in a segment from the reth-benchmark.
//  2. `VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8>`:
//    its trace width is 36 and its after challenge trace width is 80.
pub const DEFAULT_MAX_CELLS_PER_CHIP_IN_SEGMENT: usize = DEFAULT_MAX_SEGMENT_LEN * 120;

pub trait SegmentationStrategy:
    std::fmt::Debug + Send + Sync + std::panic::UnwindSafe + std::panic::RefUnwindSafe
{
    /// Whether the execution should segment based on the trace heights and cells.
    ///
    /// Air names are provided for debugging purposes.
    fn should_segment(
        &self,
        air_names: &[String],
        trace_heights: &[usize],
        trace_cells: &[usize],
    ) -> bool;

    /// A strategy that segments more aggressively than the current one.
    ///
    /// Called when `should_segment` results in a segment that is infeasible. Execution will be
    /// re-run with the stricter segmentation strategy.
    fn stricter_strategy(&self) -> Arc<dyn SegmentationStrategy>;
}

/// Default segmentation strategy: segment if any chip's height or cells exceed the limits.
#[derive(Debug, Clone)]
pub struct DefaultSegmentationStrategy {
    max_segment_len: usize,
    max_cells_per_chip_in_segment: usize,
}

impl Default for DefaultSegmentationStrategy {
    fn default() -> Self {
        Self {
            max_segment_len: DEFAULT_MAX_SEGMENT_LEN,
            max_cells_per_chip_in_segment: DEFAULT_MAX_CELLS_PER_CHIP_IN_SEGMENT,
        }
    }
}

impl DefaultSegmentationStrategy {
    pub fn new_with_max_segment_len(max_segment_len: usize) -> Self {
        Self {
            max_segment_len,
            max_cells_per_chip_in_segment: max_segment_len * 120,
        }
    }

    pub fn new(max_segment_len: usize, max_cells_per_chip_in_segment: usize) -> Self {
        Self {
            max_segment_len,
            max_cells_per_chip_in_segment,
        }
    }

    pub fn max_segment_len(&self) -> usize {
        self.max_segment_len
    }
}

const SEGMENTATION_BACKOFF_FACTOR: usize = 4;

impl SegmentationStrategy for DefaultSegmentationStrategy {
    fn should_segment(
        &self,
        air_names: &[String],
        trace_heights: &[usize],
        trace_cells: &[usize],
    ) -> bool {
        for (i, &height) in trace_heights.iter().enumerate() {
            if height > self.max_segment_len {
                tracing::info!(
                    "Should segment because chip {} (name: {}) has height {}",
                    i,
                    air_names[i],
                    height
                );
                return true;
            }
        }
        for (i, &num_cells) in trace_cells.iter().enumerate() {
            if num_cells > self.max_cells_per_chip_in_segment {
                tracing::info!(
                    "Should segment because chip {} (name: {}) has {} cells",
                    i,
                    air_names[i],
                    num_cells
                );
                return true;
            }
        }
        false
    }

    fn stricter_strategy(&self) -> Arc<dyn SegmentationStrategy> {
        Arc::new(Self {
            max_segment_len: self.max_segment_len / SEGMENTATION_BACKOFF_FACTOR,
            max_cells_per_chip_in_segment: self.max_cells_per_chip_in_segment
                / SEGMENTATION_BACKOFF_FACTOR,
        })
    }
}
