use bytesize::ByteSize;
use getset::{Setters, WithSetters};
#[cfg(feature = "metrics")]
use openvm_stark_backend::memory_metering::INTERACTION_MEMORY_OVERHEAD;
use openvm_stark_backend::{
    memory_metering::{ProvingMemoryConfig, ProvingMemoryCounts},
    p3_field::PrimeField32,
};
use p3_baby_bear::BabyBear;
use serde::{Deserialize, Serialize};

use crate::utils::{add_one_or_zero, next_power_of_two_or_zero};

pub const DEFAULT_SEGMENT_CHECK_INSNS: u64 = 1000;

pub const DEFAULT_MAX_TRACE_HEIGHT_BITS: u8 = 22;
pub const DEFAULT_MAX_TRACE_HEIGHT: u32 = 1 << DEFAULT_MAX_TRACE_HEIGHT_BITS;
pub const DEFAULT_MAX_MEMORY: usize = 15 << 30; // 15GiB
const DEFAULT_MAX_INTERACTIONS: usize = BabyBear::ORDER_U32 as usize;

#[derive(derive_new::new, Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub instret_start: u64,
    pub num_insns: u64,
    pub trace_heights: Vec<u32>,
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
    pub(crate) interactions: Vec<usize>,
    pub(crate) need_rot: Vec<bool>,
    pub(crate) limits: SegmentationLimits,
    pub(crate) memory_config: ProvingMemoryConfig,
    pub instret: u64,
    pub instrets_until_check: u64,
    pub segment_check_insns: u64,
    /// Checkpoint of trace heights at last known state where all thresholds satisfied
    pub(crate) checkpoint_trace_heights: Vec<u32>,
    /// Instruction count at the checkpoint
    checkpoint_instret: u64,
}

#[cfg_attr(not(feature = "metrics"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
enum SegmentationTrigger {
    Height { air_id: usize },
    Memory,
    Interactions,
}

#[cfg(feature = "metrics")]
impl SegmentationTrigger {
    fn reason(self) -> &'static str {
        match self {
            SegmentationTrigger::Height { .. } => "height",
            SegmentationTrigger::Memory => "memory",
            SegmentationTrigger::Interactions => "interactions",
        }
    }
}

#[cfg(feature = "metrics")]
#[derive(Default)]
struct MeteredCounts {
    /// Rows before power-of-two padding.
    unpadded_rows: usize,
    /// Rows added by power-of-two padding.
    padding_rows: usize,
    /// Main trace cells for AIRs that open next-row rotations, before padding.
    main_unpadded_with_rot: usize,
    /// Main trace cells for AIRs that open next-row rotations, from padding rows.
    main_padding_with_rot: usize,
    /// Main trace cells for AIRs without next-row rotations, before padding.
    main_unpadded_no_rot: usize,
    /// Main trace cells for AIRs without next-row rotations, from padding rows.
    main_padding_no_rot: usize,
    /// Metered row-interaction slots before padding.
    interaction_cells_unpadded: usize,
    /// Metered row-interaction slots from padding rows.
    interaction_cells_padding: usize,
}

#[cfg(feature = "metrics")]
struct MeteredMemoryBreakdown {
    /// Total selected segment memory estimate.
    total: usize,
    /// Unpadded-row contribution to the selected memory estimate.
    unpadded: usize,
    /// Padding-row contribution to the selected memory estimate.
    padding: usize,
}

impl SegmentationCtx {
    pub fn new(
        air_names: Vec<String>,
        widths: Vec<usize>,
        interactions: Vec<usize>,
        need_rot: Vec<bool>,
        limits: SegmentationLimits,
        memory_config: ProvingMemoryConfig,
    ) -> Self {
        assert_eq!(air_names.len(), widths.len());
        assert_eq!(air_names.len(), interactions.len());
        assert_eq!(air_names.len(), need_rot.len());

        let num_airs = air_names.len();
        Self {
            segments: Vec::new(),
            air_names,
            widths,
            interactions,
            need_rot,
            limits,
            memory_config,
            instret: 0,
            instrets_until_check: DEFAULT_SEGMENT_CHECK_INSNS,
            segment_check_insns: DEFAULT_SEGMENT_CHECK_INSNS,
            checkpoint_trace_heights: vec![0; num_airs],
            checkpoint_instret: 0,
        }
    }

    pub fn set_max_trace_height(&mut self, max_trace_height: u32) {
        self.limits.set_max_trace_height(max_trace_height);
    }

    pub fn set_max_memory(&mut self, max_memory: usize) {
        self.limits.max_memory = max_memory;
    }

    pub fn set_max_interactions(&mut self, max_interactions: usize) {
        self.limits.max_interactions = max_interactions;
    }

    /// Calculate the maximum trace height and corresponding air name
    #[inline(always)]
    fn calculate_max_trace_height_with_name(&self, trace_heights: &[u32]) -> (u32, &str) {
        trace_heights
            .iter()
            .enumerate()
            .map(|(i, &height)| (next_power_of_two_or_zero(height as usize) as u32, i))
            .max_by_key(|(height, _)| *height)
            .map(|(height, idx)| (height, self.air_names[idx].as_str()))
            .unwrap_or((0, "unknown"))
    }

    /// Convert main trace cells and interaction cells to memory bytes.
    #[inline(always)]
    fn counts_to_memory(
        &self,
        main_cnt_with_rot: usize,
        main_cnt_no_rot: usize,
        interaction_cells: usize,
    ) -> (
        usize, /* memory */
        usize, /* main */
        usize, /* interaction */
    ) {
        let estimate = self.memory_config.estimate(ProvingMemoryCounts::new(
            main_cnt_with_rot,
            main_cnt_no_rot,
            interaction_cells,
        ));
        (estimate.total, estimate.main, estimate.interaction)
    }

    /// Sum padded main trace cells and interaction cells across all chips, splitting main
    /// cells by per-AIR `need_rot`.
    #[cfg(feature = "metrics")]
    #[inline(always)]
    fn calculate_count_breakdown(&self, trace_heights: &[u32]) -> MeteredCounts {
        debug_assert_eq!(trace_heights.len(), self.widths.len());
        debug_assert_eq!(trace_heights.len(), self.interactions.len());
        debug_assert_eq!(trace_heights.len(), self.need_rot.len());

        let mut counts = MeteredCounts::default();
        for (((&height, &width), &interactions), &need_rot) in trace_heights
            .iter()
            .zip(self.widths.iter())
            .zip(self.interactions.iter())
            .zip(self.need_rot.iter())
        {
            let padded_height = next_power_of_two_or_zero(height as usize);
            let unpadded_height = height as usize;
            let padding_height = padded_height - unpadded_height;
            counts.unpadded_rows += unpadded_height;
            counts.padding_rows += padding_height;
            let main_unpadded_cells = unpadded_height * width;
            let main_padding_cells = padding_height * width;
            if need_rot {
                counts.main_unpadded_with_rot += main_unpadded_cells;
                counts.main_padding_with_rot += main_padding_cells;
            } else {
                counts.main_unpadded_no_rot += main_unpadded_cells;
                counts.main_padding_no_rot += main_padding_cells;
            }
            counts.interaction_cells_unpadded += unpadded_height * interactions;
            counts.interaction_cells_padding += padding_height * interactions;
        }
        counts
    }

    /// Sum padded main trace cells and interaction cells across all chips, splitting main
    /// cells by per-AIR `need_rot`.
    #[inline(always)]
    fn calculate_cell_counts(&self, trace_heights: &[u32]) -> (usize, usize, usize) {
        debug_assert_eq!(trace_heights.len(), self.widths.len());
        debug_assert_eq!(trace_heights.len(), self.interactions.len());
        debug_assert_eq!(trace_heights.len(), self.need_rot.len());

        let mut main_cnt_with_rot = 0;
        let mut main_cnt_no_rot = 0;
        let mut interaction_cells = 0;
        for (((&height, &width), &interactions), &need_rot) in trace_heights
            .iter()
            .zip(self.widths.iter())
            .zip(self.interactions.iter())
            .zip(self.need_rot.iter())
        {
            let padded_height = next_power_of_two_or_zero(height as usize);
            let main_cells = padded_height * width;
            if need_rot {
                main_cnt_with_rot += main_cells;
            } else {
                main_cnt_no_rot += main_cells;
            }
            interaction_cells += padded_height * interactions;
        }
        (main_cnt_with_rot, main_cnt_no_rot, interaction_cells)
    }

    /// Calculate total memory in bytes based on trace heights and widths.
    #[inline(always)]
    fn calculate_total_memory(
        &self,
        trace_heights: &[u32],
    ) -> (
        usize, /* memory */
        usize, /* main */
        usize, /* interaction */
    ) {
        let (main_cnt_with_rot, main_cnt_no_rot, interaction_cells) =
            self.calculate_cell_counts(trace_heights);
        self.counts_to_memory(main_cnt_with_rot, main_cnt_no_rot, interaction_cells)
    }

    #[cfg(feature = "metrics")]
    #[inline(always)]
    fn calculate_memory_breakdown(&self, counts: &MeteredCounts) -> MeteredMemoryBreakdown {
        let unpadded = self.memory_config.estimate(ProvingMemoryCounts::new(
            counts.main_unpadded_with_rot,
            counts.main_unpadded_no_rot,
            counts.interaction_cells_unpadded,
        ));
        let total = self.memory_config.estimate(ProvingMemoryCounts::new(
            counts.main_unpadded_with_rot + counts.main_padding_with_rot,
            counts.main_unpadded_no_rot + counts.main_padding_no_rot,
            counts.interaction_cells_unpadded + counts.interaction_cells_padding,
        ));

        MeteredMemoryBreakdown {
            total: total.total,
            unpadded: unpadded.total,
            padding: total.total - unpadded.total,
        }
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
            .map(|(&height, &interactions)| add_one_or_zero(height) * interactions)
            .sum()
    }

    #[inline(always)]
    pub(crate) fn should_segment(
        &self,
        instret: u64,
        trace_heights: &[u32],
        is_trace_height_constant: &[bool],
    ) -> bool {
        self.segmentation_trigger(instret, trace_heights, is_trace_height_constant)
            .is_some()
    }

    #[inline(always)]
    fn segmentation_trigger(
        &self,
        instret: u64,
        trace_heights: &[u32],
        is_trace_height_constant: &[bool],
    ) -> Option<SegmentationTrigger> {
        debug_assert_eq!(trace_heights.len(), is_trace_height_constant.len());
        debug_assert_eq!(trace_heights.len(), self.air_names.len());
        debug_assert_eq!(trace_heights.len(), self.widths.len());
        debug_assert_eq!(trace_heights.len(), self.interactions.len());
        debug_assert_eq!(trace_heights.len(), self.need_rot.len());

        let instret_start = self
            .segments
            .last()
            .map_or(0, |s| s.instret_start + s.num_insns);
        let num_insns = instret - instret_start;

        // Segment should contain at least one cycle
        if num_insns == 0 {
            return None;
        }

        let mut main_cnt_with_rot = 0usize;
        let mut main_cnt_no_rot = 0usize;
        let mut interaction_cells = 0usize;
        for (i, ((((padded_height, width), interactions), is_constant), &need_rot)) in trace_heights
            .iter()
            .map(|&height| next_power_of_two_or_zero(height as usize) as u32)
            .zip(self.widths.iter())
            .zip(self.interactions.iter())
            .zip(is_trace_height_constant.iter())
            .zip(self.need_rot.iter())
            .enumerate()
        {
            // Only segment if the height is not constant and exceeds the maximum height after
            // padding
            if !is_constant && padded_height > self.limits.max_trace_height {
                let air_name = unsafe { self.air_names.get_unchecked(i) };
                tracing::info!(
                    "overshoot: instret {:10} | height ({:8}) > max ({:8}) | chip {:3} ({}) ",
                    instret,
                    padded_height,
                    self.limits.max_trace_height,
                    i,
                    air_name,
                );
                return Some(SegmentationTrigger::Height { air_id: i });
            }
            let main_cells = padded_height as usize * width;
            if need_rot {
                main_cnt_with_rot += main_cells;
            } else {
                main_cnt_no_rot += main_cells;
            }
            interaction_cells += padded_height as usize * interactions;
        }

        let (total_memory, main_memory, interaction_memory) =
            self.counts_to_memory(main_cnt_with_rot, main_cnt_no_rot, interaction_cells);
        if total_memory > self.limits.max_memory {
            tracing::info!(
                "overshoot: instret {:10} | total memory ({:5}) > max ({:5}) | main ({:5}) | interaction ({:5})",
                instret,
                ByteSize::b(total_memory as u64),
                ByteSize::b(self.limits.max_memory as u64),
                ByteSize::b(main_memory as u64),
                ByteSize::b(interaction_memory as u64),
            );
            return Some(SegmentationTrigger::Memory);
        }

        let total_interactions = self.calculate_total_interactions(trace_heights);
        if total_interactions > self.limits.max_interactions {
            tracing::info!(
                "overshoot: instret {:10} | total interactions ({:10}) > max ({:10})",
                instret,
                total_interactions,
                self.limits.max_interactions
            );
            return Some(SegmentationTrigger::Interactions);
        }

        None
    }

    #[inline(always)]
    pub fn check_and_segment(
        &mut self,
        instret: u64,
        trace_heights: &mut [u32],
        is_trace_height_constant: &[bool],
    ) -> bool {
        let trigger = self.segmentation_trigger(instret, trace_heights, is_trace_height_constant);
        let should_segment = trigger.is_some();

        #[cfg(feature = "metrics")]
        if let Some(trigger) = trigger {
            self.emit_segmentation_trigger_metric(trigger);
        }

        if should_segment {
            self.create_segment_from_checkpoint(instret, trace_heights);
            true
        } else {
            false
        }
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
        #[cfg(feature = "metrics")]
        {
            let segment = self.segments.len().to_string();
            self.emit_metered_segment_metrics(&segment, &trace_heights);
            self.emit_metered_air_metrics(&segment, &trace_heights);
        }
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
                let padded = next_power_of_two_or_zero(height as usize) * width;
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

#[cfg(feature = "metrics")]
impl SegmentationCtx {
    fn emit_segmentation_trigger_metric(&self, trigger: SegmentationTrigger) {
        let segment = self.segments.len().to_string();
        let reason = trigger.reason();
        match trigger {
            SegmentationTrigger::Height { air_id } => {
                let labels = [
                    ("segment", segment),
                    ("reason", reason.to_string()),
                    ("air_id", air_id.to_string()),
                    ("air_name", self.air_names[air_id].clone()),
                ];
                metrics::counter!("segmentation_trigger", &labels).absolute(1);
            }
            SegmentationTrigger::Memory | SegmentationTrigger::Interactions => {
                let labels = [("segment", segment), ("reason", reason.to_string())];
                metrics::counter!("segmentation_trigger", &labels).absolute(1);
            }
        }
    }

    fn emit_metered_segment_metrics(&self, segment: &str, trace_heights: &[u32]) {
        let counts = self.calculate_count_breakdown(trace_heights);
        let memory = self.calculate_memory_breakdown(&counts);
        let labels = [("segment", segment.to_string())];
        metrics::counter!("metered_memory_bytes", &labels).absolute(memory.total as u64);
        metrics::counter!("metered_memory_unpadded_bytes", &labels)
            .absolute(memory.unpadded as u64);
        metrics::counter!("metered_memory_padding_bytes", &labels).absolute(memory.padding as u64);
        metrics::counter!("metered_interaction_memory_overhead_bytes", &labels)
            .absolute(INTERACTION_MEMORY_OVERHEAD as u64);
    }

    fn emit_metered_air_metrics(&self, segment: &str, trace_heights: &[u32]) {
        let memory_config = self.memory_config;

        for (air_id, ((((&height, &width), &interactions), &need_rot), air_name)) in trace_heights
            .iter()
            .zip(self.widths.iter())
            .zip(self.interactions.iter())
            .zip(self.need_rot.iter())
            .zip(self.air_names.iter())
            .enumerate()
        {
            let padded_height = next_power_of_two_or_zero(height as usize);
            let unpadded_height = height as usize;
            let padding_height = padded_height - unpadded_height;
            if padded_height == 0 {
                continue;
            }
            let labels = [
                ("air_name", air_name.clone()),
                ("air_id", air_id.to_string()),
                ("segment", segment.to_string()),
            ];
            let unpadded_cells = unpadded_height * width;
            let padding_cells = padding_height * width;
            // One interaction cell is one metered row-interaction slot.
            let interaction_cells_unpadded = unpadded_height * interactions;
            let interaction_cells_padding = padding_height * interactions;
            let main_secondary_unpadded =
                memory_config.main_secondary_memory_bytes_for_rot(unpadded_cells, need_rot);
            let main_secondary = memory_config
                .main_secondary_memory_bytes_for_rot(unpadded_cells + padding_cells, need_rot);
            let interaction_unpadded =
                memory_config.interaction_memory_bytes_without_overhead(interaction_cells_unpadded);
            let interaction_total = memory_config.interaction_memory_bytes_without_overhead(
                interaction_cells_unpadded + interaction_cells_padding,
            );

            metrics::counter!("metered_rows_unpadded", &labels).absolute(height as u64);
            metrics::counter!("metered_rows_padding", &labels).absolute(padding_height as u64);
            metrics::counter!("metered_main_cells_unpadded", &labels)
                .absolute(unpadded_cells as u64);
            metrics::counter!("metered_main_cells_padding", &labels).absolute(padding_cells as u64);
            metrics::counter!("metered_interaction_cells_unpadded", &labels)
                .absolute(interaction_cells_unpadded as u64);
            metrics::counter!("metered_interaction_cells_padding", &labels)
                .absolute(interaction_cells_padding as u64);
            metrics::counter!("metered_main_memory_unpadded_bytes", &labels)
                .absolute(memory_config.main_memory_bytes(unpadded_cells) as u64);
            metrics::counter!("metered_main_memory_padding_bytes", &labels)
                .absolute(memory_config.main_memory_bytes(padding_cells) as u64);
            metrics::counter!("metered_main_secondary_memory_unpadded_bytes", &labels)
                .absolute(main_secondary_unpadded as u64);
            metrics::counter!("metered_main_secondary_memory_padding_bytes", &labels)
                .absolute((main_secondary - main_secondary_unpadded) as u64);
            metrics::counter!("metered_interaction_memory_unpadded_bytes", &labels)
                .absolute(interaction_unpadded as u64);
            metrics::counter!("metered_interaction_memory_padding_bytes", &labels)
                .absolute((interaction_total - interaction_unpadded) as u64);
        }
    }
}
