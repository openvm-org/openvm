use openvm_stark_backend::p3_field::PrimeField32;
use p3_baby_bear::BabyBear;
use serde::{Deserialize, Serialize};

/// Check segment every 100 instructions.
const DEFAULT_SEGMENT_CHECK_INSNS: u64 = 100;

const DEFAULT_MAX_TRACE_HEIGHT: u32 = (1 << 23) - 100;
const DEFAULT_MAX_CELLS: usize = 2_000_000_000; // 2B
const DEFAULT_MAX_INTERACTIONS: usize = BabyBear::ORDER_U32 as usize;

#[derive(derive_new::new, Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub instret_start: u64,
    pub num_insns: u64,
    pub trace_heights: Vec<u32>,
}

#[derive(Debug)]
pub struct SegmentationLimits {
    pub max_trace_height: u32,
    pub max_cells: usize,
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

#[derive(Debug)]
pub struct SegmentationCtx {
    pub segments: Vec<Segment>,
    instret_last_segment_check: u64,
    pub(crate) air_names: Vec<String>,
    widths: Vec<usize>,
    interactions: Vec<usize>,
    segment_check_insns: u64,
    segmentation_limits: SegmentationLimits,
}

impl SegmentationCtx {
    pub fn new(air_names: Vec<String>, widths: Vec<usize>, interactions: Vec<usize>) -> Self {
        Self {
            segments: Vec::new(),
            air_names,
            widths,
            interactions,
            segment_check_insns: DEFAULT_SEGMENT_CHECK_INSNS,
            segmentation_limits: SegmentationLimits::default(),
            instret_last_segment_check: 0,
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

    pub fn set_segment_check_insns(&mut self, segment_check_insns: u64) {
        self.segment_check_insns = segment_check_insns;
    }

    /// Calculate the total cells used based on trace heights and widths
    fn calculate_total_cells(&self, trace_heights: &[u32]) -> usize {
        trace_heights
            .iter()
            .zip(&self.widths)
            .map(|(&height, &width)| height as usize * width)
            .sum()
    }

    /// Calculate the total interactions based on trace heights and interaction counts
    fn calculate_total_interactions(&self, trace_heights: &[u32]) -> usize {
        trace_heights
            .iter()
            .zip(&self.interactions)
            // We add 1 for the zero messages from the padding rows
            .map(|(&height, &interactions)| (height + 1) as usize * interactions)
            .sum()
    }

    fn should_segment(
        &self,
        instret: u64,
        trace_heights: &[u32],
        is_trace_height_constant: &[bool],
    ) -> bool {
        let instret_start = self
            .segments
            .last()
            .map_or(0, |s| s.instret_start + s.num_insns);
        let num_insns = instret - instret_start;
        // Segment should contain at least one cycle
        if num_insns == 0 {
            return false;
        }
        for (i, &height) in trace_heights.iter().enumerate() {
            // Only segment if the height is not constant and exceeds the maximum height
            if !is_trace_height_constant[i] && height > self.segmentation_limits.max_trace_height {
                tracing::info!(
                    "Segment {:2} | instret {:9} | chip {} ({}) height ({:8}) > max ({:8})",
                    self.segments.len(),
                    instret,
                    i,
                    self.air_names[i],
                    height,
                    self.segmentation_limits.max_trace_height
                );
                return true;
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

    pub fn check_and_segment(
        &mut self,
        instret: u64,
        trace_heights: &[u32],
        is_trace_height_constant: &[bool],
    ) -> bool {
        // Avoid checking segment too often.
        if instret < self.instret_last_segment_check + self.segment_check_insns {
            return false;
        }

        let ret = self.should_segment(instret, trace_heights, is_trace_height_constant);
        if ret {
            self.segment(instret, trace_heights);
        }
        self.instret_last_segment_check = instret;
        ret
    }

    /// Try segment if there is at least one cycle
    pub fn segment(&mut self, instret: u64, trace_heights: &[u32]) {
        let instret_start = self
            .segments
            .last()
            .map_or(0, |s| s.instret_start + s.num_insns);
        let num_insns = instret - instret_start;
        self.segments.push(Segment {
            instret_start,
            num_insns,
            trace_heights: trace_heights.to_vec(),
        });
    }

    pub fn add_final_segment(&mut self, instret: u64, trace_heights: &[u32]) {
        tracing::info!(
            "Segment {:2} | instret {:9} | terminated",
            self.segments.len(),
            instret,
        );
        // Add the last segment
        let instret_start = self
            .segments
            .last()
            .map_or(0, |s| s.instret_start + s.num_insns);
        let segment = Segment {
            instret_start,
            num_insns: instret - instret_start,
            trace_heights: trace_heights.to_vec(),
        };
        self.segments.push(segment);
    }
}
