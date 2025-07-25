use getset::WithSetters;
use openvm_stark_backend::p3_field::PrimeField32;
use p3_baby_bear::BabyBear;
use serde::{Deserialize, Serialize};

const DEFAULT_MAX_TRACE_HEIGHT: u32 = (1 << 23) - 10000;
const DEFAULT_MAX_CELLS: usize = 2_000_000_000; // 2B
const DEFAULT_MAX_INTERACTIONS: usize = BabyBear::ORDER_U32 as usize;

#[derive(derive_new::new, Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub instret_start: u64,
    pub num_insns: u64,
    pub trace_heights: Vec<u32>,
}

#[derive(Clone, Copy, Debug, WithSetters)]
pub struct SegmentationLimits {
    #[getset(set_with = "pub")]
    pub max_trace_height: u32,
    #[getset(set_with = "pub")]
    pub max_cells: usize,
    #[getset(set_with = "pub")]
    pub max_interactions: usize,
}

impl Default for SegmentationLimits {
    fn default() -> Self {
        Self {
            max_trace_height: DEFAULT_MAX_TRACE_HEIGHT,
            max_cells: DEFAULT_MAX_CELLS,
            max_interactions: DEFAULT_MAX_INTERACTIONS,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SegmentationCtx {
    pub segments: Vec<Segment>,
    pub(crate) air_names: Vec<String>,
    widths: Vec<usize>,
    interactions: Vec<usize>,
    pub(crate) segmentation_limits: SegmentationLimits,
}

impl SegmentationCtx {
    pub fn new(
        air_names: Vec<String>,
        widths: Vec<usize>,
        interactions: Vec<usize>,
        segmentation_limits: SegmentationLimits,
    ) -> Self {
        debug_assert_eq!(air_names.len(), widths.len());
        debug_assert_eq!(air_names.len(), interactions.len());

        Self {
            segments: Vec::new(),
            air_names,
            widths,
            interactions,
            segmentation_limits,
        }
    }

    pub fn new_with_default_segmentation_limits(
        air_names: Vec<String>,
        widths: Vec<usize>,
        interactions: Vec<usize>,
    ) -> Self {
        debug_assert_eq!(air_names.len(), widths.len());
        debug_assert_eq!(air_names.len(), interactions.len());

        Self {
            segments: Vec::new(),
            air_names,
            widths,
            interactions,
            segmentation_limits: SegmentationLimits::default(),
        }
    }

    pub fn set_max_trace_height(&mut self, max_trace_height: u32) {
        self.segmentation_limits.max_trace_height = max_trace_height;
    }

    pub fn set_max_cells(&mut self, max_cells: usize) {
        self.segmentation_limits.max_cells = max_cells;
    }

    pub fn set_max_interactions(&mut self, max_interactions: usize) {
        self.segmentation_limits.max_interactions = max_interactions;
    }

    /// Calculate the total cells used based on trace heights and widths
    #[inline(always)]
    fn calculate_total_cells(&self, trace_heights: &[u32]) -> usize {
        debug_assert_eq!(trace_heights.len(), self.widths.len());

        unsafe {
            trace_heights
                .iter()
                .zip(self.widths.get_unchecked(..trace_heights.len()))
                .map(|(&height, &width)| height as usize * width)
                .sum()
        }
    }

    /// Calculate the total interactions based on trace heights and interaction counts
    #[inline(always)]
    fn calculate_total_interactions(&self, trace_heights: &[u32]) -> usize {
        debug_assert_eq!(trace_heights.len(), self.interactions.len());

        unsafe {
            trace_heights
                .iter()
                .zip(self.interactions.get_unchecked(..trace_heights.len()))
                // We add 1 for the zero messages from the padding rows
                .map(|(&height, &interactions)| (height + 1) as usize * interactions)
                .sum()
        }
    }

    #[inline(always)]
    fn should_segment(
        &self,
        instret: u64,
        trace_heights: &[u32],
        is_trace_height_constant: &[bool],
    ) -> bool {
        debug_assert_eq!(trace_heights.len(), is_trace_height_constant.len());
        debug_assert_eq!(trace_heights.len(), self.air_names.len());

        let instret_start = self
            .segments
            .last()
            .map_or(0, |s| s.instret_start + s.num_insns);
        let num_insns = instret - instret_start;

        // Segment should contain at least one cycle
        if num_insns == 0 {
            return false;
        }

        unsafe {
            for i in 0..trace_heights.len() {
                let height = *trace_heights.get_unchecked(i);
                let is_constant = *is_trace_height_constant.get_unchecked(i);

                // Only segment if the height is not constant and exceeds the maximum height
                if !is_constant && height > self.segmentation_limits.max_trace_height {
                    tracing::info!(
                        "Segment {:2} | instret {:9} | chip {} ({}) height ({:8}) > max ({:8})",
                        self.segments.len(),
                        instret,
                        i,
                        self.air_names.get_unchecked(i),
                        height,
                        self.segmentation_limits.max_trace_height
                    );
                    return true;
                }
            }
        }

        let total_cells = self.calculate_total_cells(trace_heights);
        if total_cells > self.segmentation_limits.max_cells {
            tracing::info!(
                "Segment {:2} | instret {:9} | total cells ({:10}) > max ({:10})",
                self.segments.len(),
                instret,
                total_cells,
                self.segmentation_limits.max_cells
            );
            return true;
        }

        let total_interactions = self.calculate_total_interactions(trace_heights);
        if total_interactions > self.segmentation_limits.max_interactions {
            tracing::info!(
                "Segment {:2} | instret {:9} | total interactions ({:11}) > max ({:11})",
                self.segments.len(),
                instret,
                total_interactions,
                self.segmentation_limits.max_interactions
            );
            return true;
        }

        false
    }

    #[inline(always)]
    pub fn check_and_segment(
        &mut self,
        instret: u64,
        trace_heights: &[u32],
        is_trace_height_constant: &[bool],
    ) -> bool {
        let ret = self.should_segment(instret, trace_heights, is_trace_height_constant);
        if ret {
            self.segment(instret, trace_heights);
        }
        ret
    }

    /// Try segment if there is at least one cycle
    #[inline(always)]
    pub fn segment(&mut self, instret: u64, trace_heights: &[u32]) {
        let instret_start = self
            .segments
            .last()
            .map_or(0, |s| s.instret_start + s.num_insns);
        let num_insns = instret - instret_start;

        debug_assert!(num_insns > 0, "Segment should contain at least one cycle");

        self.segments.push(Segment {
            instret_start,
            num_insns,
            trace_heights: trace_heights.to_vec(),
        });
    }
}
