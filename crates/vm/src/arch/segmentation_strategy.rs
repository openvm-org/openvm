use std::sync::Arc;

pub const DEFAULT_MAX_SEGMENT_LEN: usize = (1 << 22) - 100;
pub const DEFAULT_MAX_CELLS_IN_SEGMENT: usize = 2_000_000_000; // 2B

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

    /// Maximum height of any chip in a segment.
    fn max_trace_height(&self) -> usize;

    /// Maximum number of cells in a segment.
    fn max_cells(&self) -> usize;
}

/// Default segmentation strategy: segment if any chip's height or cells exceed the limits.
#[derive(Debug, Clone)]
pub struct DefaultSegmentationStrategy {
    max_segment_len: usize,
    max_cells_in_segment: usize,
}

impl Default for DefaultSegmentationStrategy {
    fn default() -> Self {
        Self {
            max_segment_len: DEFAULT_MAX_SEGMENT_LEN,
            max_cells_in_segment: DEFAULT_MAX_CELLS_IN_SEGMENT,
        }
    }
}

impl DefaultSegmentationStrategy {
    pub fn new_with_max_segment_len(max_segment_len: usize) -> Self {
        Self {
            max_segment_len,
            max_cells_in_segment: DEFAULT_MAX_CELLS_IN_SEGMENT,
        }
    }

    pub fn new(max_segment_len: usize, max_cells_in_segment: usize) -> Self {
        Self {
            max_segment_len,
            max_cells_in_segment,
        }
    }

    pub fn max_segment_len(&self) -> usize {
        self.max_segment_len
    }
}

const SEGMENTATION_BACKOFF_FACTOR: usize = 4;

impl SegmentationStrategy for DefaultSegmentationStrategy {
    fn max_trace_height(&self) -> usize {
        self.max_segment_len
    }

    fn max_cells(&self) -> usize {
        self.max_cells_in_segment
    }

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
        let total_cells: usize = trace_cells.iter().sum();
        if total_cells > self.max_cells_in_segment {
            tracing::info!(
                "Should segment because total cells across all chips is {}",
                total_cells
            );
            return true;
        }
        false
    }

    fn stricter_strategy(&self) -> Arc<dyn SegmentationStrategy> {
        Arc::new(Self {
            max_segment_len: self.max_segment_len / SEGMENTATION_BACKOFF_FACTOR,
            max_cells_in_segment: self.max_cells_in_segment / SEGMENTATION_BACKOFF_FACTOR,
        })
    }
}
