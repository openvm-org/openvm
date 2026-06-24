use itertools::Itertools;
use openvm_instructions::riscv::{RV64_IMM_AS, RV64_REGISTER_AS};
use openvm_stark_backend::memory_metering::ProvingMemoryConfig;
use serde::{Deserialize, Serialize};

use super::{
    memory_ctx::MemoryCtx,
    segment_ctx::{Segment, SegmentationConfig, SegmentationCtx, SegmentationLimits},
};
use crate::{
    arch::{
        execution_mode::{ExecutionCtxTrait, MeteredExecutionCtxTrait},
        SystemConfig, VmExecState, BOUNDARY_AIR_ID, MERKLE_AIR_ID,
    },
    system::memory::online::GuestMemory,
};

pub const DEFAULT_PAGE_BITS: usize = 6;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeteredCtxConfig {
    pub initial_trace_heights: Vec<u32>,
    pub is_trace_height_constant: Vec<bool>,
    // TODO: Remove this once segmented execution is selected by typed executor entry points across
    // all backends. RVR already treats the compiled suspender policy as the source of truth.
    pub suspend_on_segment: bool,
}

#[derive(Clone, Debug)]
pub struct MeteredCtx<const PAGE_BITS: usize = DEFAULT_PAGE_BITS> {
    pub config: MeteredCtxConfig,
    pub trace_heights: Vec<u32>,
    pub memory_ctx: MemoryCtx<PAGE_BITS>,
    pub segmentation_ctx: SegmentationCtx,
}

pub struct MeteredCtxInputs<'a> {
    pub constant_trace_heights: &'a [Option<usize>],
    pub air_names: &'a [String],
    pub widths: &'a [usize],
    pub interactions: &'a [usize],
    pub need_rot: &'a [bool],
    pub segmentation_limits: SegmentationLimits,
}

#[cfg(feature = "rvr")]
pub(crate) struct MeteredCtxParts<const PAGE_BITS: usize = DEFAULT_PAGE_BITS> {
    pub config: MeteredCtxConfig,
    pub trace_heights: Vec<u32>,
    pub memory_ctx: MemoryCtx<PAGE_BITS>,
    pub segmentation_ctx: SegmentationCtx,
}

impl<const PAGE_BITS: usize> MeteredCtx<PAGE_BITS> {
    // Note: prefer to use `build_metered_ctx` in `VmExecutor` or `VirtualMachine`.
    pub fn new(
        inputs: MeteredCtxInputs<'_>,
        config: &SystemConfig,
        memory_config: ProvingMemoryConfig,
    ) -> Self {
        let (trace_heights, is_trace_height_constant): (Vec<u32>, Vec<bool>) = inputs
            .constant_trace_heights
            .iter()
            .map(|&constant_height| {
                if let Some(height) = constant_height {
                    (height as u32, true)
                } else {
                    (0, false)
                }
            })
            .unzip();

        let segmentation_ctx = SegmentationCtx::new(
            inputs.air_names.to_vec(),
            inputs.widths.to_vec(),
            inputs.interactions.to_vec(),
            inputs.need_rot.to_vec(),
            inputs.segmentation_limits,
            memory_config,
        );
        let memory_ctx = MemoryCtx::new(config, segmentation_ctx.segment_check_insns());

        // Assert that the indices are correct
        let air_names = segmentation_ctx.air_names();
        debug_assert!(
            air_names[BOUNDARY_AIR_ID].contains("Boundary"),
            "air_name={}",
            air_names[BOUNDARY_AIR_ID]
        );
        debug_assert!(
            air_names[MERKLE_AIR_ID].contains("Merkle"),
            "air_name={}",
            air_names[MERKLE_AIR_ID]
        );
        let mut ctx = Self {
            config: MeteredCtxConfig {
                initial_trace_heights: trace_heights.clone(),
                is_trace_height_constant,
                suspend_on_segment: false,
            },
            trace_heights,
            memory_ctx,
            segmentation_ctx,
        };

        // Add merkle height contributions for all registers
        ctx.memory_ctx.add_register_merkle_heights();
        ctx.memory_ctx
            .lazy_update_boundary_heights(&mut ctx.trace_heights);

        ctx
    }

    pub fn with_max_memory(mut self, max_memory: usize) -> Self {
        self.segmentation_ctx.set_max_memory(max_memory);
        self
    }

    pub fn set_cache_rs_code_matrix(&mut self, cache_rs_code_matrix: bool) {
        self.segmentation_ctx
            .set_cache_rs_code_matrix(cache_rs_code_matrix);
    }

    pub fn with_cache_rs_code_matrix(mut self, cache_rs_code_matrix: bool) -> Self {
        self.set_cache_rs_code_matrix(cache_rs_code_matrix);
        self
    }

    #[inline(always)]
    pub fn cache_rs_code_matrix(&self) -> bool {
        self.segmentation_ctx.cache_rs_code_matrix()
    }

    pub fn from_config(
        config: MeteredCtxConfig,
        segmentation_config: SegmentationConfig,
        system_config: &SystemConfig,
    ) -> Self {
        let segmentation_ctx = SegmentationCtx::from_config(segmentation_config);
        let mut memory_ctx = MemoryCtx::new(system_config, segmentation_ctx.segment_check_insns());
        let mut trace_heights = config.initial_trace_heights.clone();
        memory_ctx.add_register_merkle_heights();
        memory_ctx.lazy_update_boundary_heights(&mut trace_heights);
        Self {
            trace_heights,
            config,
            memory_ctx,
            segmentation_ctx,
        }
    }

    pub fn set_suspend_on_segment(&mut self, suspend_on_segment: bool) {
        self.config.suspend_on_segment = suspend_on_segment;
    }

    pub fn with_suspend_on_segment(mut self, suspend_on_segment: bool) -> Self {
        self.set_suspend_on_segment(suspend_on_segment);
        self
    }

    pub fn suspend_on_segment(&self) -> &bool {
        &self.config.suspend_on_segment
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segmentation_ctx.segments
    }

    pub fn into_segments(self) -> Vec<Segment> {
        self.segmentation_ctx.segments
    }

    #[cfg(feature = "rvr")]
    pub(crate) fn into_parts(self) -> MeteredCtxParts<PAGE_BITS> {
        MeteredCtxParts {
            config: self.config,
            trace_heights: self.trace_heights,
            memory_ctx: self.memory_ctx,
            segmentation_ctx: self.segmentation_ctx,
        }
    }

    #[cfg(feature = "rvr")]
    pub(crate) fn from_parts(parts: MeteredCtxParts<PAGE_BITS>) -> Self {
        Self {
            config: parts.config,
            trace_heights: parts.trace_heights,
            memory_ctx: parts.memory_ctx,
            segmentation_ctx: parts.segmentation_ctx,
        }
    }

    #[inline(always)]
    pub fn check_and_segment(&mut self) -> bool {
        // We track the segmentation check by instrets_until_check instead of instret in order to
        // save a register in AOT mode.
        if self.segmentation_ctx.instrets_until_check > 0 {
            return false;
        }
        let segment_check_insns = self.segmentation_ctx.segment_check_insns();
        self.segmentation_ctx.instrets_until_check = segment_check_insns;
        self.segmentation_ctx.instret += segment_check_insns;

        self.memory_ctx
            .lazy_update_boundary_heights(&mut self.trace_heights);
        let did_segment = self.segmentation_ctx.check_and_segment(
            self.segmentation_ctx.instret,
            &mut self.trace_heights,
            &self.config.is_trace_height_constant,
        );

        if did_segment {
            // Initialize contexts for new segment
            self.segmentation_ctx.initialize_segment(
                &mut self.trace_heights,
                &self.config.is_trace_height_constant,
            );
            self.memory_ctx.initialize_segment(&mut self.trace_heights);

            // Check if the new segment is within limits
            self.segmentation_ctx.warn_if_exceeds_limits(
                self.segmentation_ctx.instret,
                &self.trace_heights,
                &self.config.is_trace_height_constant,
            );
        }

        // Update checkpoints
        self.segmentation_ctx
            .update_checkpoint(self.segmentation_ctx.instret, &self.trace_heights);
        self.memory_ctx.update_checkpoint();

        did_segment
    }

    #[allow(dead_code)]
    pub fn print_segment(&self) {
        println!("{}", "-".repeat(80));
        println!("Segment {}", self.segmentation_ctx.segments.len() - 1);
        println!("{}", "-".repeat(80));
        println!("{:>10} {:>10} {:<30}", "Width", "Height", "Air Name");
        println!("{}", "-".repeat(80));
        for ((&width, &height), air_name) in self
            .segmentation_ctx
            .widths()
            .iter()
            .zip_eq(self.trace_heights.iter())
            .zip_eq(self.segmentation_ctx.air_names().iter())
        {
            println!("{:>10} {:>10} {:<30}", width, height, air_name.as_str());
        }
    }
}

impl<const PAGE_BITS: usize> ExecutionCtxTrait for MeteredCtx<PAGE_BITS> {
    #[inline(always)]
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: u32) {
        debug_assert!(
            address_space != RV64_IMM_AS,
            "address space must not be immediate"
        );
        debug_assert!(size > 0, "size must be greater than 0, got {size}");
        debug_assert!(
            size.is_power_of_two(),
            "size must be a power of 2, got {size}"
        );

        // Handle merkle tree updates
        if address_space != RV64_REGISTER_AS {
            self.memory_ctx
                .update_boundary_merkle_heights(address_space, ptr, size);
        }
    }

    #[inline(always)]
    fn should_suspend<F>(exec_state: &mut VmExecState<F, GuestMemory, Self>) -> bool {
        // ATTENTION: Please make sure to update the corresponding logic in the
        // `asm_bridge` crate and `aot.rs`` when you change this function.
        // If `segment_suspend` is set, suspend when a segment is determined (but the VM state might
        // be after the segment boundary because the segment happens in the previous checkpoint).
        // Otherwise, execute until termination.
        if exec_state.ctx.check_and_segment() && exec_state.ctx.config.suspend_on_segment {
            true
        } else {
            exec_state.ctx.segmentation_ctx.instrets_until_check -= 1;
            false
        }
    }

    #[inline(always)]
    fn on_terminate<F>(exec_state: &mut VmExecState<F, GuestMemory, Self>) {
        exec_state
            .ctx
            .memory_ctx
            .lazy_update_boundary_heights(&mut exec_state.ctx.trace_heights);
        exec_state
            .ctx
            .segmentation_ctx
            .create_final_segment(&exec_state.ctx.trace_heights);
    }
}

impl<const PAGE_BITS: usize> MeteredExecutionCtxTrait for MeteredCtx<PAGE_BITS> {
    #[inline(always)]
    fn on_height_change(&mut self, chip_idx: usize, height_delta: u32) {
        debug_assert!(
            chip_idx < self.trace_heights.len(),
            "chip_idx out of bounds"
        );
        // SAFETY: chip_idx is created in executor_idx_to_air_idx and is always within bounds
        unsafe {
            *self.trace_heights.get_unchecked_mut(chip_idx) = self
                .trace_heights
                .get_unchecked(chip_idx)
                .wrapping_add(height_delta);
        }
    }
}

#[cfg(all(test, feature = "rvr"))]
mod tests {
    use openvm_stark_backend::StarkEngine;

    use super::*;
    use crate::{
        arch::{BOUNDARY_AIR_ID, MERKLE_AIR_ID},
        utils::{test_cpu_engine, test_system_config},
    };

    #[test]
    fn rvr_metered_ctx_parts_roundtrip_preserves_execution_state() {
        let system_config = test_system_config();
        let num_airs = 6;
        let mut air_names = (0..num_airs)
            .map(|idx| format!("Air {idx}"))
            .collect::<Vec<_>>();
        air_names[BOUNDARY_AIR_ID] = "Memory Boundary".to_string();
        air_names[MERKLE_AIR_ID] = "Memory Merkle".to_string();

        let mut segmentation_ctx = SegmentationCtx::new(
            air_names,
            vec![1; num_airs],
            vec![0; num_airs],
            vec![false; num_airs],
            SegmentationLimits {
                max_trace_height_bits: 12,
                max_memory: usize::MAX,
                max_interactions: u32::MAX,
            },
            test_cpu_engine().proving_memory_config(),
        );
        segmentation_ctx.instret = 123;
        segmentation_ctx.instrets_until_check = 1;

        let mut memory_ctx = MemoryCtx::<DEFAULT_PAGE_BITS>::new(&system_config, 1);
        memory_ctx.record_page_access(7, 1);
        memory_ctx.record_page_access(8, 1);
        memory_ctx.record_page_access(9, 1);

        let mut ctx = MeteredCtx::<DEFAULT_PAGE_BITS> {
            config: MeteredCtxConfig {
                initial_trace_heights: vec![1, 2, 3, 4, 5, 6],
                is_trace_height_constant: vec![false, true, false, true, false, true],
                suspend_on_segment: true,
            },
            trace_heights: vec![1, 2, 3, 4, 5, 6],
            memory_ctx,
            segmentation_ctx,
        };
        ctx.set_cache_rs_code_matrix(false);

        assert!(!ctx.cache_rs_code_matrix());
        let restored = MeteredCtx::from_parts(ctx.into_parts());

        assert!(!restored.cache_rs_code_matrix());
        assert_eq!(restored.trace_heights, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(
            restored.config.is_trace_height_constant,
            vec![false, true, false, true, false, true]
        );
        assert_eq!(restored.segmentation_ctx.instret, 123);
        assert_eq!(restored.memory_ctx.page_indices_since_checkpoint_len, 3);
        assert!(*restored.suspend_on_segment());
    }
}
