use getset::WithSetters;
use openvm_stark_backend::p3_field::PrimeField32;
use p3_baby_bear::BabyBear;
use serde::{Deserialize, Serialize};

pub const DEFAULT_SEGMENT_CHECK_INSNS: u64 = 1000;

pub const DEFAULT_MAX_TRACE_HEIGHT: u32 = 1 << 23;
pub const DEFAULT_MAX_CELLS: usize = 2_000_000_000; // 2B
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

#[derive(Clone, Debug, WithSetters)]
pub struct SegmentationCtx {
    pub segments: Vec<Segment>,
    pub(crate) air_names: Vec<String>,
    pub(crate) widths: Vec<usize>,
    interactions: Vec<usize>,
    pub(crate) segmentation_limits: SegmentationLimits,
    pub instret_last_segment_check: u64,
    #[getset(set_with = "pub")]
    pub segment_check_insns: u64,
    /// Checkpoint of trace heights at last known state where all thresholds satisfied
    pub(crate) checkpoint_trace_heights: Vec<u32>,
    /// Instruction count at the checkpoint
    checkpoint_instret: u64,
}

impl SegmentationCtx {
    pub fn new(
        air_names: Vec<String>,
        widths: Vec<usize>,
        interactions: Vec<usize>,
        segmentation_limits: SegmentationLimits,
    ) -> Self {
        assert_eq!(air_names.len(), widths.len());
        assert_eq!(air_names.len(), interactions.len());

        let num_airs = air_names.len();
        Self {
            segments: Vec::new(),
            air_names,
            widths,
            interactions,
            segmentation_limits,
            segment_check_insns: DEFAULT_SEGMENT_CHECK_INSNS,
            instret_last_segment_check: 0,
            checkpoint_trace_heights: vec![0; num_airs],
            checkpoint_instret: 0,
        }
    }

    pub fn new_with_default_segmentation_limits(
        air_names: Vec<String>,
        widths: Vec<usize>,
        interactions: Vec<usize>,
    ) -> Self {
        assert_eq!(air_names.len(), widths.len());
        assert_eq!(air_names.len(), interactions.len());

        let num_airs = air_names.len();
        Self {
            segments: Vec::new(),
            air_names,
            widths,
            interactions,
            segmentation_limits: SegmentationLimits::default(),
            segment_check_insns: DEFAULT_SEGMENT_CHECK_INSNS,
            instret_last_segment_check: 0,
            checkpoint_trace_heights: vec![0; num_airs],
            checkpoint_instret: 0,
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

    #[inline(always)]
    fn should_segment(
        &self,
        instret: u64,
        trace_heights: &[u32],
        is_trace_height_constant: &[bool],
    ) -> bool {
        debug_assert_eq!(trace_heights.len(), is_trace_height_constant.len());
        debug_assert_eq!(trace_heights.len(), self.air_names.len());
        debug_assert_eq!(trace_heights.len(), self.widths.len());
        debug_assert_eq!(trace_heights.len(), self.interactions.len());

        let instret_start = self
            .segments
            .last()
            .map_or(0, |s| s.instret_start + s.num_insns);
        let num_insns = instret - instret_start;

        // Segment should contain at least one cycle
        if num_insns == 0 {
            return false;
        }

        let mut total_cells = 0;
        for (i, ((padded_height, width), is_constant)) in trace_heights
            .iter()
            .map(|&height| height.next_power_of_two())
            .zip(self.widths.iter())
            .zip(is_trace_height_constant.iter())
            .enumerate()
        {
            // Only segment if the height is not constant and exceeds the maximum height after
            // padding
            if !is_constant && padded_height > self.segmentation_limits.max_trace_height {
                let air_name = unsafe { self.air_names.get_unchecked(i) };
                tracing::info!(
                    "instret {:9} | chip {} ({}) height ({:8}) > max ({:8})",
                    instret,
                    i,
                    air_name,
                    padded_height,
                    self.segmentation_limits.max_trace_height
                );
                return true;
            }
            total_cells += padded_height as usize * width;
        }

        if total_cells > self.segmentation_limits.max_cells {
            tracing::info!(
                "instret {:9} | total cells ({:10}) > max ({:10})",
                instret,
                total_cells,
                self.segmentation_limits.max_cells
            );
            return true;
        }

        // All padding rows contribute a single message to the interactions (+1) since
        // we assume chips don't send/receive with nonzero multiplicity on padding rows.
        let total_interactions: usize = trace_heights
            .iter()
            .zip(self.interactions.iter())
            .map(|(&height, &interactions)| (height + 1) as usize * interactions)
            .sum();
        if total_interactions > self.segmentation_limits.max_interactions {
            tracing::info!(
                "instret {:9} | total interactions ({:11}) > max ({:11})",
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
        trace_heights: &mut [u32],
        is_trace_height_constant: &[bool],
    ) -> bool {
        let should_seg = self.should_segment(instret, trace_heights, is_trace_height_constant);

        if should_seg {
            self.create_segment_from_checkpoint(instret, trace_heights, is_trace_height_constant);
        } else {
            self.update_checkpoint(instret, trace_heights);
        }

        self.instret_last_segment_check = instret;
        should_seg
    }

    #[inline(always)]
    fn create_segment_from_checkpoint(
        &mut self,
        instret: u64,
        trace_heights: &mut [u32],
        is_trace_height_constant: &[bool],
    ) {
        let instret_start = self
            .segments
            .last()
            .map_or(0, |s| s.instret_start + s.num_insns);

        let (segment_instret, segment_heights) = if self.checkpoint_instret > instret_start {
            (
                self.checkpoint_instret,
                self.checkpoint_trace_heights.clone(),
            )
        } else {
            // No valid checkpoint, use current values
            (instret, trace_heights.to_vec())
        };

        // Reset current trace heights and checkpoint
        self.reset_trace_heights(trace_heights, &segment_heights, is_trace_height_constant);
        self.checkpoint_instret = 0;

        tracing::info!(
            "Segment {:2} | instret {:9} | {} instructions",
            self.segments.len(),
            instret_start,
            segment_instret - instret_start
        );
        self.segments.push(Segment {
            instret_start,
            num_insns: segment_instret - instret_start,
            trace_heights: segment_heights,
        });
    }

    /// Resets trace heights by subtracting segment heights
    #[inline(always)]
    fn reset_trace_heights(
        &self,
        trace_heights: &mut [u32],
        segment_heights: &[u32],
        is_trace_height_constant: &[bool],
    ) {
        for ((trace_height, &segment_height), &is_trace_height_constant) in trace_heights
            .iter_mut()
            .zip(segment_heights.iter())
            .zip(is_trace_height_constant.iter())
        {
            if !is_trace_height_constant {
                *trace_height = trace_height.checked_sub(segment_height).unwrap();
            }
        }
    }

    /// Updates the checkpoint with current safe state
    #[inline(always)]
    fn update_checkpoint(&mut self, instret: u64, trace_heights: &[u32]) {
        self.checkpoint_trace_heights.copy_from_slice(trace_heights);
        self.checkpoint_instret = instret;
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

        tracing::info!(
            "Segment {:2} | instret {:9} | {} instructions [FINAL]",
            self.segments.len(),
            instret_start,
            num_insns
        );
        self.segments.push(Segment {
            instret_start,
            num_insns,
            trace_heights: trace_heights.to_vec(),
        });
    }
}
