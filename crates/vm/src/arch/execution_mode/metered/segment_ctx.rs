use std::{cmp::max, mem::size_of};

use bytesize::ByteSize;
use getset::{Setters, WithSetters};
use openvm_stark_backend::p3_field::PrimeField32;
use p3_baby_bear::BabyBear;
use serde::{Deserialize, Serialize};

/// Extension field size.
const D_EF: usize = 4;
pub const DEFAULT_SEGMENT_CHECK_INSNS: u64 = 1000;

pub const DEFAULT_MAX_TRACE_HEIGHT_BITS: u8 = 22;
pub const DEFAULT_MAX_TRACE_HEIGHT: u32 = 1 << DEFAULT_MAX_TRACE_HEIGHT_BITS;
pub const DEFAULT_MAX_MEMORY: usize = 15 << 30; // 15GiB
const DEFAULT_MAX_INTERACTIONS: usize = BabyBear::ORDER_U32 as usize;
const DEFAULT_MAIN_CELL_WEIGHT: usize = 3; // 1 + 2^{log_blowup=1}
const DEFAULT_MAIN_CELL_SECONDARY_WEIGHT: f64 = 0.5;
/// Each interaction contributes 2 * D_EF base field elements to the GKR segment
/// tree leaves. We then use additional buffer whose size is a quarter of that.
const DEFAULT_INTERACTION_CELL_WEIGHT: usize = 2 * D_EF + 2 * D_EF / 4;

#[derive(derive_new::new, Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub instret_start: u64,
    pub num_insns: u64,
    pub trace_heights: Vec<u32>,
}

#[derive(Clone, Debug, WithSetters)]
pub struct SegmentationConfig {
    pub limits: SegmentationLimits,
    /// Weight multiplier for main trace cells in memory calculation.
    #[getset(set_with = "pub")]
    pub main_cell_weight: usize,
    /// Second order memory contribution from main cells. This term is maxed with the weighted
    /// interaction contribution.
    #[getset(set_with = "pub")]
    pub main_cell_secondary_weight: f64,
    /// Weight multiplier for interaction cells in memory calculation.
    #[getset(set_with = "pub")]
    pub interaction_cell_weight: usize,
    /// Size of the base field in bytes. Used to convert cell count to memory bytes.
    #[getset(set_with = "pub")]
    pub base_field_size: usize,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            limits: SegmentationLimits::default(),
            main_cell_weight: DEFAULT_MAIN_CELL_WEIGHT,
            main_cell_secondary_weight: DEFAULT_MAIN_CELL_SECONDARY_WEIGHT,
            interaction_cell_weight: DEFAULT_INTERACTION_CELL_WEIGHT,
            base_field_size: size_of::<u32>(),
        }
    }
}

#[derive(Clone, Debug, WithSetters, Setters)]
pub struct SegmentationLimits {
    pub max_trace_height: u32,
    #[getset(set = "pub", set_with = "pub")]
    pub max_memory: usize,
    #[getset(set_with = "pub")]
    pub max_interactions: usize,
}

impl Default for SegmentationLimits {
    fn default() -> Self {
        Self {
            max_trace_height: DEFAULT_MAX_TRACE_HEIGHT,
            max_memory: DEFAULT_MAX_MEMORY,
            max_interactions: DEFAULT_MAX_INTERACTIONS,
        }
    }
}

impl SegmentationLimits {
    pub fn new(max_trace_height: u32, max_memory: usize, max_interactions: usize) -> Self {
        debug_assert!(
            max_trace_height.is_power_of_two(),
            "max_trace_height should be a power of two"
        );
        Self {
            max_trace_height,
            max_memory,
            max_interactions,
        }
    }

    pub fn with_max_trace_height(mut self, max_trace_height: u32) -> Self {
        debug_assert!(
            max_trace_height.is_power_of_two(),
            "max_trace_height should be a power of two"
        );
        self.max_trace_height = max_trace_height;
        self
    }

    pub fn set_max_trace_height(&mut self, max_trace_height: u32) {
        debug_assert!(
            max_trace_height.is_power_of_two(),
            "max_trace_height should be a power of two"
        );
        self.max_trace_height = max_trace_height;
    }
}

#[derive(Clone, Debug, WithSetters)]
pub struct SegmentationCtx {
    pub segments: Vec<Segment>,
    pub(crate) air_names: Vec<String>,
    pub(crate) widths: Vec<usize>,
    interactions: Vec<usize>,
    pub(crate) config: SegmentationConfig,
    pub instret: u64,
    pub instrets_until_check: u64,
    pub(super) segment_check_insns: u64,
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
        config: SegmentationConfig,
    ) -> Self {
        assert_eq!(air_names.len(), widths.len());
        assert_eq!(air_names.len(), interactions.len());

        let num_airs = air_names.len();
        Self {
            segments: Vec::new(),
            air_names,
            widths,
            interactions,
            config,
            instret: 0,
            instrets_until_check: DEFAULT_SEGMENT_CHECK_INSNS,
            segment_check_insns: DEFAULT_SEGMENT_CHECK_INSNS,
            checkpoint_trace_heights: vec![0; num_airs],
            checkpoint_instret: 0,
        }
    }

    pub fn set_max_trace_height(&mut self, max_trace_height: u32) {
        self.config.limits.set_max_trace_height(max_trace_height);
    }

    pub fn set_max_memory(&mut self, max_memory: usize) {
        self.config.limits.max_memory = max_memory;
    }

    pub fn set_max_interactions(&mut self, max_interactions: usize) {
        self.config.limits.max_interactions = max_interactions;
    }

    pub fn set_main_cell_weight(&mut self, weight: usize) {
        self.config.main_cell_weight = weight;
    }

    pub fn set_main_cell_secondary_weight(&mut self, weight: f64) {
        self.config.main_cell_secondary_weight = weight;
    }

    pub fn set_interaction_cell_weight(&mut self, weight: usize) {
        self.config.interaction_cell_weight = weight;
    }

    pub fn set_base_field_size(&mut self, base_field_size: usize) {
        self.config.base_field_size = base_field_size;
    }

    /// Calculate the maximum trace height and corresponding air name
    #[inline(always)]
    fn calculate_max_trace_height_with_name(&self, trace_heights: &[u32]) -> (u32, &str) {
        trace_heights
            .iter()
            .enumerate()
            .map(|(i, &height)| (height.next_power_of_two(), i))
            .max_by_key(|(height, _)| *height)
            .map(|(height, idx)| (height, self.air_names[idx].as_str()))
            .unwrap_or((0, "unknown"))
    }

    /// Calculate total memory in bytes based on trace heights and widths.
    /// Formula: base_field_size * (main_cell_weight * main_cells + interaction_cell_weight *
    /// interaction_cells)
    #[inline(always)]
    fn calculate_total_memory(
        &self,
        trace_heights: &[u32],
    ) -> (
        usize, /* memory */
        usize, /* main */
        usize, /* interaction */
    ) {
        debug_assert_eq!(trace_heights.len(), self.widths.len());

        let main_weight = self.config.main_cell_weight;
        let main_secondary_weight = self.config.main_cell_secondary_weight;
        let interaction_weight = self.config.interaction_cell_weight;
        let base_field_size = self.config.base_field_size;

        let mut main_cnt = 0;
        let mut interaction_cnt = 0;
        for ((&height, &width), &interactions) in trace_heights
            .iter()
            .zip(self.widths.iter())
            .zip(self.interactions.iter())
        {
            let padded_height = height.next_power_of_two() as usize;
            main_cnt += padded_height * width;
            interaction_cnt += padded_height * interactions;
        }

        let main_memory = main_cnt * main_weight * base_field_size;
        let main_secondary_memory =
            (((main_cnt * base_field_size) as f64) * main_secondary_weight).round() as usize;
        let interaction_memory =
            (interaction_cnt + 1).next_power_of_two() * interaction_weight * base_field_size;
        (
            main_memory + max(main_secondary_memory, interaction_memory),
            main_memory,
            interaction_memory,
        )
    }

    /// Calculate the total interactions based on trace heights
    /// All padding rows contribute a single message to the interactions (+1) since
    /// we assume chips don't send/receive with nonzero multiplicity on padding rows.
    #[inline(always)]
    fn calculate_total_interactions(&self, trace_heights: &[u32]) -> usize {
        debug_assert_eq!(trace_heights.len(), self.interactions.len());

        trace_heights
            .iter()
            .zip(self.interactions.iter())
            .map(|(&height, &interactions)| (height + 1) as usize * interactions)
            .sum()
    }

    #[inline(always)]
    pub(crate) fn should_segment(
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

        let main_weight = self.config.main_cell_weight;
        let main_secondary_weight = self.config.main_cell_secondary_weight;
        let interaction_weight = self.config.interaction_cell_weight;
        let base_field_size = self.config.base_field_size;
        let mut main_cnt = 0usize;
        let mut interaction_cnt = 0usize;
        for (i, (((padded_height, width), interactions), is_constant)) in trace_heights
            .iter()
            .map(|&height| height.next_power_of_two())
            .zip(self.widths.iter())
            .zip(self.interactions.iter())
            .zip(is_trace_height_constant.iter())
            .enumerate()
        {
            // Only segment if the height is not constant and exceeds the maximum height after
            // padding
            if !is_constant && padded_height > self.config.limits.max_trace_height {
                let air_name = unsafe { self.air_names.get_unchecked(i) };
                tracing::info!(
                    "overshoot: instret {:10} | height ({:8}) > max ({:8}) | chip {:3} ({}) ",
                    instret,
                    padded_height,
                    self.config.limits.max_trace_height,
                    i,
                    air_name,
                );
                return true;
            }
            main_cnt += padded_height as usize * width;
            interaction_cnt += padded_height as usize * interactions;
        }
        let main_memory = main_cnt * main_weight * base_field_size;
        let main_secondary_memory =
            (((main_cnt * base_field_size) as f64) * main_secondary_weight).round() as usize;
        // interaction rounding to match n_logup calculation
        let interaction_memory =
            (interaction_cnt + 1).next_power_of_two() * interaction_weight * base_field_size;
        let total_memory = main_memory + max(main_secondary_memory, interaction_memory);

        if total_memory > self.config.limits.max_memory {
            tracing::info!(
                "overshoot: instret {:10} | total memory ({:10}) > max ({:10}) | main ({:10}) | interaction ({:10})",
                instret,
                total_memory,
                self.config.limits.max_memory,
                main_cnt,
                interaction_cnt
            );
            return true;
        }

        let total_interactions = self.calculate_total_interactions(trace_heights);
        if total_interactions > self.config.limits.max_interactions {
            tracing::info!(
                "overshoot: instret {:10} | total interactions ({:10}) > max ({:10})",
                instret,
                total_interactions,
                self.config.limits.max_interactions
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
            self.create_segment_from_checkpoint(instret, trace_heights);
        }
        should_seg
    }

    #[inline(always)]
    fn create_segment_from_checkpoint(&mut self, instret: u64, trace_heights: &mut [u32]) {
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
            let trace_heights_str = trace_heights
                .iter()
                .zip(self.air_names.iter())
                .filter(|(&height, _)| height > 0)
                .map(|(&height, name)| format!("  {name} = {height}"))
                .collect::<Vec<_>>()
                .join("\n");
            tracing::warn!(
                "No valid checkpoint, creating segment using instret={instret}\ntrace_heights=[\n{trace_heights_str}\n]"
            );
            // No valid checkpoint, use current values
            (instret, trace_heights.to_vec())
        };

        let num_insns = segment_instret - instret_start;
        self.create_segment::<false>(instret_start, num_insns, segment_heights);
    }

    /// Initialize state for a new segment
    #[inline(always)]
    pub(crate) fn initialize_segment(
        &mut self,
        trace_heights: &mut [u32],
        is_trace_height_constant: &[bool],
    ) {
        // Reset trace heights by subtracting the last segment's heights
        let last_segment = self.segments.last().unwrap();
        self.reset_trace_heights(
            trace_heights,
            &last_segment.trace_heights,
            is_trace_height_constant,
        );
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
    pub(crate) fn update_checkpoint(&mut self, instret: u64, trace_heights: &[u32]) {
        self.checkpoint_trace_heights.copy_from_slice(trace_heights);
        self.checkpoint_instret = instret;
    }

    /// Try segment if there is at least one instruction
    #[inline(always)]
    pub fn create_final_segment(&mut self, trace_heights: &[u32]) {
        self.instret += self.segment_check_insns - self.instrets_until_check;
        self.instrets_until_check = self.segment_check_insns;
        let instret_start = self
            .segments
            .last()
            .map_or(0, |s| s.instret_start + s.num_insns);

        let num_insns = self.instret - instret_start;
        self.create_segment::<true>(instret_start, num_insns, trace_heights.to_vec());
    }

    /// Push a new segment with logging
    #[inline(always)]
    fn create_segment<const IS_FINAL: bool>(
        &mut self,
        instret_start: u64,
        num_insns: u64,
        trace_heights: Vec<u32>,
    ) {
        debug_assert!(
            num_insns > 0,
            "Segment should contain at least one instruction"
        );

        self.log_segment_info::<IS_FINAL>(instret_start, num_insns, &trace_heights);
        self.segments.push(Segment {
            instret_start,
            num_insns,
            trace_heights,
        });
    }

    /// Calculate trace utilization: ratio of used cells to padded cells (as percentage).
    /// This measures how efficiently the trace is packed before padding to power of two.
    /// Note: this is an overestimate because memory-related trace heights are overestimated.
    #[inline(always)]
    fn calculate_trace_utilization(&self, trace_heights: &[u32]) -> f64 {
        let (used, padded) = trace_heights
            .iter()
            .zip(self.widths.iter())
            .map(|(&height, &width)| {
                let used = height as usize * width;
                let padded = height.next_power_of_two() as usize * width;
                (used, padded)
            })
            .fold((0, 0), |(u, p), (used, padded)| (u + used, p + padded));
        if padded == 0 {
            0.0
        } else {
            100.0 * used as f64 / padded as f64
        }
    }

    /// Log segment information
    #[inline(always)]
    fn log_segment_info<const IS_FINAL: bool>(
        &self,
        instret_start: u64,
        num_insns: u64,
        trace_heights: &[u32],
    ) {
        let (max_trace_height, air_name) = self.calculate_max_trace_height_with_name(trace_heights);
        let (total_memory, main_memory, interaction_memory) =
            self.calculate_total_memory(trace_heights);
        let total_interactions = self.calculate_total_interactions(trace_heights);
        let utilization = self.calculate_trace_utilization(trace_heights);

        let final_marker = if IS_FINAL { " [TERMINATED]" } else { "" };

        tracing::info!(
            "Segment {:3} | instret {:10} | {:8} instructions | {:5} memory ({:5}, {:5}) | {:10} interactions | {:8} max height ({}) | {:.2}% utilization{}",
            self.segments.len(),
            instret_start,
            num_insns,
            ByteSize::b(total_memory as u64),
            ByteSize::b(main_memory as u64),
            ByteSize::b(interaction_memory as u64),
            total_interactions,
            max_trace_height,
            air_name,
            utilization,
            final_marker
        );
    }
}
