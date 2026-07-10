use bytesize::ByteSize;
#[cfg(feature = "metrics")]
use openvm_stark_backend::memory_metering::INTERACTION_MEMORY_OVERHEAD;
use openvm_stark_backend::memory_metering::{ProvingMemoryConfig, ProvingMemoryCounts};
use serde::{Deserialize, Serialize};

use crate::utils::{add_one_or_zero, next_power_of_two_or_zero};

pub const DEFAULT_SEGMENT_CHECK_INSNS: u64 = 1000;

pub const DEFAULT_MAX_MEMORY: usize = 15 << 30; // 15GiB

#[derive(derive_new::new, Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub instret_start: u64,
    pub num_insns: u64,
    pub trace_heights: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct SegmentationLimits {
    pub max_trace_height_bits: u8,
    pub max_memory: usize,
    pub max_interactions: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "ProvingMemoryConfig")]
struct ProvingMemoryConfigSerde {
    base_field_size: usize,
    extension_degree: usize,
    log_blowup: usize,
    l_skip: usize,
    max_constraint_degree: usize,
    cache_rs_code_matrix: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentationConfig {
    air_names: Vec<String>,
    widths: Vec<usize>,
    interactions: Vec<usize>,
    need_rot: Vec<bool>,
    max_trace_height: u32,
    max_memory: usize,
    max_interactions: u32,
    #[serde(with = "ProvingMemoryConfigSerde")]
    memory_config: ProvingMemoryConfig,
    segment_check_insns: u64,
}

impl SegmentationConfig {
    fn new(
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
        assert!(
            limits.max_trace_height_bits < u32::BITS as u8,
            "max_trace_height_bits must be less than {}",
            u32::BITS
        );

        let max_trace_height = 1u32
            .checked_shl(u32::from(limits.max_trace_height_bits))
            .expect("max_trace_height_bits must fit in u32 trace height");
        assert!(
            u64::from(max_trace_height) >= 2 * DEFAULT_SEGMENT_CHECK_INSNS,
            "max_trace_height must be at least twice DEFAULT_SEGMENT_CHECK_INSNS"
        );

        Self {
            air_names,
            widths,
            interactions,
            need_rot,
            max_trace_height,
            max_memory: limits.max_memory,
            max_interactions: limits.max_interactions,
            memory_config,
            segment_check_insns: DEFAULT_SEGMENT_CHECK_INSNS,
        }
    }

    #[inline(always)]
    pub fn air_names(&self) -> &[String] {
        &self.air_names
    }

    #[inline(always)]
    pub fn widths(&self) -> &[usize] {
        &self.widths
    }

    #[inline(always)]
    pub fn segment_check_insns(&self) -> u64 {
        self.segment_check_insns
    }

    pub fn set_max_memory(&mut self, max_memory: usize) {
        self.max_memory = max_memory;
    }

    pub fn set_cache_rs_code_matrix(&mut self, cache_rs_code_matrix: bool) {
        self.memory_config.cache_rs_code_matrix = cache_rs_code_matrix;
    }

    #[inline(always)]
    pub fn cache_rs_code_matrix(&self) -> bool {
        self.memory_config.cache_rs_code_matrix
    }

    #[inline(always)]
    pub fn max_memory(&self) -> usize {
        self.max_memory
    }
}

/// Per-AIR data for the periodic segmentation scan, packed contiguously so the
/// hot loop touches one cache line per few AIRs instead of five parallel slices.
#[derive(Clone, Debug)]
struct ScanAir {
    air_idx: u32,
    width: u32,
    interactions: u32,
    need_rot: bool,
}

/// Precomputed data for the periodic segmentation scan. Constant-height AIRs
/// contribute fixed cell/interaction counts, so they are aggregated once here
/// and the per-check loop walks only the non-constant AIRs.
#[derive(Clone, Debug)]
struct ScanPlan {
    /// Non-constant AIRs in index order.
    airs: Vec<ScanAir>,
    /// Padded main-trace cells of constant-height AIRs that open rotations.
    const_main_with_rot: usize,
    /// Padded main-trace cells of constant-height AIRs without rotations.
    const_main_no_rot: usize,
    /// Padded interaction cells of constant-height AIRs.
    const_interaction_cells: usize,
    /// Total interactions of constant-height AIRs.
    const_total_interactions: u64,
}

/// Accumulated totals from one periodic scan.
struct ScanResult {
    main_cnt_with_rot: usize,
    main_cnt_no_rot: usize,
    interaction_cells: usize,
    total_interactions: u64,
    /// First non-constant AIR whose padded height exceeds the max, if any.
    /// When set, the other accumulators are meaningless.
    height_overshoot_air: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct SegmentationCtx {
    pub segments: Vec<Segment>,
    config: SegmentationConfig,
    pub instret: u64,
    pub instrets_until_check: u64,
    /// Checkpoint of trace heights at last known state where all thresholds satisfied
    pub(crate) checkpoint_trace_heights: Vec<u32>,
    /// Instruction count at the checkpoint
    checkpoint_instret: u64,
    /// Built via [`Self::prepare_scan_plan`]; `segmentation_trigger` falls back
    /// to scanning every AIR when absent.
    scan_plan: Option<ScanPlan>,
}

#[derive(Clone, Copy, Debug)]
enum SegmentationTrigger {
    Height {
        #[cfg(feature = "metrics")]
        air_id: usize,
    },
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

struct MeteredMemoryBreakdown {
    /// Total selected segment memory estimate.
    total: usize,
    /// Unpadded-row contribution to the selected memory estimate.
    unpadded: usize,
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
        let num_airs = air_names.len();
        let config = SegmentationConfig::new(
            air_names,
            widths,
            interactions,
            need_rot,
            limits,
            memory_config,
        );
        Self {
            segments: Vec::new(),
            instrets_until_check: config.segment_check_insns,
            config,
            instret: 0,
            checkpoint_trace_heights: vec![0; num_airs],
            checkpoint_instret: 0,
            scan_plan: None,
        }
    }

    /// Precompute the periodic-scan plan: aggregate the contributions of
    /// constant-height AIRs (their `trace_heights` entries never change) and
    /// pack the remaining AIRs' scan parameters contiguously.
    ///
    /// `trace_heights` must hold the constant AIRs' heights at their fixed
    /// values (true from construction onward).
    pub(crate) fn prepare_scan_plan(
        &mut self,
        trace_heights: &[u32],
        is_trace_height_constant: &[bool],
    ) {
        debug_assert_eq!(trace_heights.len(), is_trace_height_constant.len());
        debug_assert_eq!(trace_heights.len(), self.config.widths.len());

        let mut plan = ScanPlan {
            airs: Vec::with_capacity(trace_heights.len()),
            const_main_with_rot: 0,
            const_main_no_rot: 0,
            const_interaction_cells: 0,
            const_total_interactions: 0,
        };
        for (i, (&height, &is_constant)) in trace_heights
            .iter()
            .zip(is_trace_height_constant.iter())
            .enumerate()
        {
            let width = self.config.widths[i];
            let interactions = self.config.interactions[i];
            if is_constant {
                let padded_height = next_power_of_two_or_zero(height as usize);
                let main_cells = padded_height * width;
                if self.config.need_rot[i] {
                    plan.const_main_with_rot += main_cells;
                } else {
                    plan.const_main_no_rot += main_cells;
                }
                plan.const_interaction_cells += padded_height * interactions;
                plan.const_total_interactions +=
                    add_one_or_zero(height) as u64 * interactions as u64;
            } else {
                plan.airs.push(ScanAir {
                    air_idx: i as u32,
                    width: width.try_into().expect("AIR width must fit in u32"),
                    interactions: interactions
                        .try_into()
                        .expect("AIR interaction count must fit in u32"),
                    need_rot: self.config.need_rot[i],
                });
            }
        }
        self.scan_plan = Some(plan);
    }

    #[inline(always)]
    pub(crate) fn air_names(&self) -> &[String] {
        &self.config.air_names
    }

    #[inline(always)]
    pub(crate) fn widths(&self) -> &[usize] {
        &self.config.widths
    }

    #[inline(always)]
    pub(crate) fn segment_check_insns(&self) -> u64 {
        self.config.segment_check_insns
    }

    pub fn set_max_memory(&mut self, max_memory: usize) {
        self.config.set_max_memory(max_memory);
    }

    pub fn set_cache_rs_code_matrix(&mut self, cache_rs_code_matrix: bool) {
        self.config.set_cache_rs_code_matrix(cache_rs_code_matrix);
    }

    #[inline(always)]
    pub fn cache_rs_code_matrix(&self) -> bool {
        self.config.cache_rs_code_matrix()
    }

    pub fn config(&self) -> &SegmentationConfig {
        &self.config
    }

    pub fn from_config(config: SegmentationConfig) -> Self {
        let num_airs = config.air_names.len();
        Self {
            segments: Vec::new(),
            instrets_until_check: config.segment_check_insns,
            config,
            instret: 0,
            checkpoint_trace_heights: vec![0; num_airs],
            checkpoint_instret: 0,
            scan_plan: None,
        }
    }

    /// Calculate the maximum trace height and corresponding air name
    #[inline(always)]
    fn calculate_max_trace_height_with_name(&self, trace_heights: &[u32]) -> (u32, &str) {
        trace_heights
            .iter()
            .enumerate()
            .map(|(i, &height)| (next_power_of_two_or_zero(height as usize) as u32, i))
            .max_by_key(|(height, _)| *height)
            .map(|(height, idx)| (height, self.config.air_names[idx].as_str()))
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
        let estimate = self.config.memory_config.estimate(ProvingMemoryCounts::new(
            main_cnt_with_rot,
            main_cnt_no_rot,
            interaction_cells,
        ));
        (estimate.total, estimate.main, estimate.interaction)
    }

    /// Sum padded main trace cells and interaction cells across all chips, splitting main
    /// cells by per-AIR `need_rot`.
    #[inline(always)]
    fn calculate_count_breakdown(&self, trace_heights: &[u32]) -> MeteredCounts {
        debug_assert_eq!(trace_heights.len(), self.config.widths.len());
        debug_assert_eq!(trace_heights.len(), self.config.interactions.len());
        debug_assert_eq!(trace_heights.len(), self.config.need_rot.len());

        let mut counts = MeteredCounts::default();
        for (((&height, &width), &interactions), &need_rot) in trace_heights
            .iter()
            .zip(self.config.widths.iter())
            .zip(self.config.interactions.iter())
            .zip(self.config.need_rot.iter())
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
        debug_assert_eq!(trace_heights.len(), self.config.widths.len());
        debug_assert_eq!(trace_heights.len(), self.config.interactions.len());
        debug_assert_eq!(trace_heights.len(), self.config.need_rot.len());

        let mut main_cnt_with_rot = 0;
        let mut main_cnt_no_rot = 0;
        let mut interaction_cells = 0;
        for (((&height, &width), &interactions), &need_rot) in trace_heights
            .iter()
            .zip(self.config.widths.iter())
            .zip(self.config.interactions.iter())
            .zip(self.config.need_rot.iter())
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

    #[inline(always)]
    fn calculate_memory_breakdown(&self, counts: &MeteredCounts) -> MeteredMemoryBreakdown {
        let unpadded = self.config.memory_config.estimate(ProvingMemoryCounts::new(
            counts.main_unpadded_with_rot,
            counts.main_unpadded_no_rot,
            counts.interaction_cells_unpadded,
        ));
        let total = self.config.memory_config.estimate(ProvingMemoryCounts::new(
            counts.main_unpadded_with_rot + counts.main_padding_with_rot,
            counts.main_unpadded_no_rot + counts.main_padding_no_rot,
            counts.interaction_cells_unpadded + counts.interaction_cells_padding,
        ));

        MeteredMemoryBreakdown {
            total: total.total,
            unpadded: unpadded.total,
        }
    }

    /// Calculate the total interactions based on trace heights
    /// All padding rows contribute a single message to the interactions (+1) since
    /// we assume chips don't send/receive with nonzero multiplicity on padding rows.
    #[inline(always)]
    fn calculate_total_interactions(&self, trace_heights: &[u32]) -> u64 {
        debug_assert_eq!(trace_heights.len(), self.config.interactions.len());

        trace_heights
            .iter()
            .zip(self.config.interactions.iter())
            .map(|(&height, &interactions)| add_one_or_zero(height) as u64 * interactions as u64)
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
        debug_assert_eq!(trace_heights.len(), self.config.air_names.len());
        debug_assert_eq!(trace_heights.len(), self.config.widths.len());
        debug_assert_eq!(trace_heights.len(), self.config.interactions.len());
        debug_assert_eq!(trace_heights.len(), self.config.need_rot.len());

        let instret_start = self
            .segments
            .last()
            .map_or(0, |s| s.instret_start + s.num_insns);
        let num_insns = instret - instret_start;

        // Segment should contain at least one cycle
        if num_insns == 0 {
            return None;
        }

        let scan = if let Some(plan) = &self.scan_plan {
            self.scan_with_plan(plan, trace_heights)
        } else {
            self.scan_all_airs(trace_heights, is_trace_height_constant)
        };

        if let Some(i) = scan.height_overshoot_air {
            let padded_height = next_power_of_two_or_zero(trace_heights[i] as usize);
            tracing::info!(
                "overshoot: instret {:10} | height ({:8}) > max ({:8}) | chip {:3} ({}) ",
                instret,
                padded_height,
                self.config.max_trace_height,
                i,
                self.config.air_names[i],
            );
            return Some(SegmentationTrigger::Height {
                #[cfg(feature = "metrics")]
                air_id: i,
            });
        }

        let (total_memory, main_memory, interaction_memory) = self.counts_to_memory(
            scan.main_cnt_with_rot,
            scan.main_cnt_no_rot,
            scan.interaction_cells,
        );
        if total_memory > self.config.max_memory {
            tracing::info!(
                "overshoot: instret {:10} | total memory ({:5}) > max ({:5}) | main ({:5}) | interaction ({:5})",
                instret,
                ByteSize::b(total_memory as u64),
                ByteSize::b(self.config.max_memory as u64),
                ByteSize::b(main_memory as u64),
                ByteSize::b(interaction_memory as u64),
            );
            return Some(SegmentationTrigger::Memory);
        }

        if scan.total_interactions > u64::from(self.config.max_interactions) {
            tracing::info!(
                "overshoot: instret {:10} | total interactions ({:10}) > max ({:10})",
                instret,
                scan.total_interactions,
                self.config.max_interactions
            );
            return Some(SegmentationTrigger::Interactions);
        }

        None
    }

    /// Scan only the non-constant AIRs, seeding accumulators with the
    /// precomputed constant-AIR contributions. Single fused pass over packed
    /// per-AIR data (heights, memory, and interactions together).
    #[inline(always)]
    fn scan_with_plan(&self, plan: &ScanPlan, trace_heights: &[u32]) -> ScanResult {
        let mut result = ScanResult {
            main_cnt_with_rot: plan.const_main_with_rot,
            main_cnt_no_rot: plan.const_main_no_rot,
            interaction_cells: plan.const_interaction_cells,
            total_interactions: plan.const_total_interactions,
            height_overshoot_air: None,
        };
        for air in &plan.airs {
            let i = air.air_idx as usize;
            debug_assert!(i < trace_heights.len());
            // SAFETY: `prepare_scan_plan` builds `air_idx` from enumerating a
            // slice of the same length as `trace_heights` (asserted above).
            let height = unsafe { *trace_heights.get_unchecked(i) };
            let padded_height = next_power_of_two_or_zero(height as usize);
            if padded_height as u32 > self.config.max_trace_height {
                result.height_overshoot_air = Some(i);
                return result;
            }
            let main_cells = padded_height * air.width as usize;
            if air.need_rot {
                result.main_cnt_with_rot += main_cells;
            } else {
                result.main_cnt_no_rot += main_cells;
            }
            result.interaction_cells += padded_height * air.interactions as usize;
            result.total_interactions += add_one_or_zero(height) as u64 * air.interactions as u64;
        }
        result
    }

    /// Fallback scan over every AIR, for contexts constructed without a scan
    /// plan. Decision-equivalent to [`Self::scan_with_plan`].
    fn scan_all_airs(
        &self,
        trace_heights: &[u32],
        is_trace_height_constant: &[bool],
    ) -> ScanResult {
        let mut result = ScanResult {
            main_cnt_with_rot: 0,
            main_cnt_no_rot: 0,
            interaction_cells: 0,
            total_interactions: 0,
            height_overshoot_air: None,
        };
        for (i, ((((&height, &width), &interactions), &is_constant), &need_rot)) in trace_heights
            .iter()
            .zip(self.config.widths.iter())
            .zip(self.config.interactions.iter())
            .zip(is_trace_height_constant.iter())
            .zip(self.config.need_rot.iter())
            .enumerate()
        {
            let padded_height = next_power_of_two_or_zero(height as usize);
            // Only segment if the height is not constant and exceeds the maximum height after
            // padding
            if !is_constant && padded_height as u32 > self.config.max_trace_height {
                result.height_overshoot_air = Some(i);
                return result;
            }
            let main_cells = padded_height * width;
            if need_rot {
                result.main_cnt_with_rot += main_cells;
            } else {
                result.main_cnt_no_rot += main_cells;
            }
            result.interaction_cells += padded_height * interactions;
            result.total_interactions += add_one_or_zero(height) as u64 * interactions as u64;
        }
        result
    }

    #[inline(always)]
    fn format_nonzero_trace_heights(&self, trace_heights: &[u32]) -> String {
        trace_heights
            .iter()
            .zip(self.config.air_names.iter())
            .filter(|(&height, _)| height > 0)
            .map(|(&height, name)| format!("  {name} = {height}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[inline(always)]
    pub(crate) fn warn_if_exceeds_limits(
        &self,
        instret: u64,
        trace_heights: &[u32],
        is_trace_height_constant: &[bool],
    ) {
        if self.should_segment(instret, trace_heights, is_trace_height_constant) {
            let trace_heights_str = self.format_nonzero_trace_heights(trace_heights);
            tracing::warn!(
                "Segment initialized with heights that exceed limits\n\
                 instret={instret}\n\
                 trace_heights=[\n{trace_heights_str}\n]"
            );
        }
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
            let trace_heights_str = self.format_nonzero_trace_heights(trace_heights);
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
        self.instret += self.config.segment_check_insns - self.instrets_until_check;
        self.instrets_until_check = self.config.segment_check_insns;
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

    /// Calculate memory utilization: ratio of unpadded memory estimate to padded memory estimate.
    /// This measures how much of the selected proving memory estimate is useful work vs
    /// power-of-two trace padding. Note: this inherits memory-related trace-height overestimates.
    #[inline(always)]
    fn calculate_memory_utilization(&self, trace_heights: &[u32]) -> f64 {
        let counts = self.calculate_count_breakdown(trace_heights);
        let memory = self.calculate_memory_breakdown(&counts);
        if memory.total == 0 {
            0.0
        } else {
            100.0 * memory.unpadded as f64 / memory.total as f64
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
        let utilization = self.calculate_memory_utilization(trace_heights);

        let final_marker = if IS_FINAL { " [TERMINATED]" } else { "" };

        tracing::info!(
            "Segment {:3} | instret {:10} | {:8} instructions | {:5} memory ({:5}, {:5}) | {:10} interactions | {:8} max height ({}) | {:.2}% memory util{}",
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
                    ("air_name", self.config.air_names[air_id].clone()),
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
        let padding = memory.total - memory.unpadded;
        let labels = [("segment", segment.to_string())];
        metrics::counter!("metered_memory_bytes", &labels).absolute(memory.total as u64);
        metrics::counter!("metered_memory_unpadded_bytes", &labels)
            .absolute(memory.unpadded as u64);
        metrics::counter!("metered_memory_padding_bytes", &labels).absolute(padding as u64);
        metrics::counter!("metered_interaction_memory_overhead_bytes", &labels)
            .absolute(INTERACTION_MEMORY_OVERHEAD as u64);
    }

    fn emit_metered_air_metrics(&self, segment: &str, trace_heights: &[u32]) {
        let memory_config = self.config.memory_config;

        for (air_id, ((((&height, &width), &interactions), &need_rot), air_name)) in trace_heights
            .iter()
            .zip(self.config.widths.iter())
            .zip(self.config.interactions.iter())
            .zip(self.config.need_rot.iter())
            .zip(self.config.air_names.iter())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn small_segmentation_ctx() -> SegmentationCtx {
        let limits = SegmentationLimits {
            max_trace_height_bits: 11,
            max_memory: 1,
            max_interactions: u32::MAX,
        };
        let memory_config = ProvingMemoryConfig {
            base_field_size: 4,
            extension_degree: 4,
            log_blowup: 1,
            l_skip: 4,
            max_constraint_degree: 4,
            cache_rs_code_matrix: false,
        };
        SegmentationCtx::new(
            vec!["air".to_string()],
            vec![1],
            vec![0],
            vec![false],
            limits,
            memory_config,
        )
    }

    #[test]
    fn test_check_and_segment_uses_last_safe_checkpoint() {
        let mut ctx = small_segmentation_ctx();
        ctx.update_checkpoint(10, &[2]);

        let mut trace_heights = vec![8];
        assert!(ctx.check_and_segment(15, &mut trace_heights, &[false]));

        assert_eq!(ctx.segments.len(), 1);
        assert_eq!(ctx.segments[0].instret_start, 0);
        assert_eq!(ctx.segments[0].num_insns, 10);
        assert_eq!(ctx.segments[0].trace_heights, vec![2]);
    }

    /// The precomputed scan plan must reach the same segmentation decisions as
    /// the fallback scan over every AIR, across all three trigger types.
    #[test]
    fn test_scan_plan_matches_fallback_scan() {
        let num_airs = 6;
        let is_constant = [false, true, false, true, false, false];
        // Constant AIRs keep these heights in every vector below.
        let base_heights: [u32; 6] = [0, 100, 0, 7, 0, 0];

        let make_ctx = |max_memory: usize, max_interactions: u32| {
            SegmentationCtx::new(
                (0..num_airs).map(|i| format!("air{i}")).collect(),
                vec![4, 2, 8, 1, 16, 3],
                vec![2, 1, 0, 3, 5, 2],
                vec![false, true, false, false, true, false],
                SegmentationLimits {
                    max_trace_height_bits: 11,
                    max_memory,
                    max_interactions,
                },
                ProvingMemoryConfig {
                    base_field_size: 4,
                    extension_degree: 4,
                    log_blowup: 1,
                    l_skip: 4,
                    max_constraint_degree: 4,
                    cache_rs_code_matrix: false,
                },
            )
        };

        let limit_cases = [
            (usize::MAX, u32::MAX), // only height can trigger
            (200_000, u32::MAX),    // memory can trigger
            (usize::MAX, 10_000),   // interactions can trigger
        ];
        let height_cases: [[u32; 6]; 5] = [
            base_heights,
            [1000, 100, 500, 7, 100, 900],
            [3000, 100, 0, 7, 0, 0], // height overshoot (padded 4096 > 2048)
            [2048, 100, 2048, 7, 2048, 2048],
            [1, 100, 1, 7, 1, 1],
        ];

        for (max_memory, max_interactions) in limit_cases {
            let fallback_ctx = make_ctx(max_memory, max_interactions);
            let mut plan_ctx = make_ctx(max_memory, max_interactions);
            plan_ctx.prepare_scan_plan(&base_heights, &is_constant);
            assert!(plan_ctx.scan_plan.is_some());
            for heights in &height_cases {
                assert_eq!(
                    plan_ctx.should_segment(50, heights, &is_constant),
                    fallback_ctx.should_segment(50, heights, &is_constant),
                    "plan and fallback disagree for heights={heights:?}, \
                     max_memory={max_memory}, max_interactions={max_interactions}"
                );
            }
        }
    }
}
