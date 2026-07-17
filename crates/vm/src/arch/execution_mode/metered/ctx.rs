use itertools::Itertools;
use openvm_instructions::{
    exe::SparseMemoryImage,
    metering::SEGMENT_CHECK_INSNS,
    riscv::{RV64_IMM_AS, RV64_REGISTER_AS},
};
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeteredCtxConfig {
    pub initial_trace_heights: Vec<u32>,
    pub is_trace_height_constant: Vec<bool>,
    // TODO: Remove this once segmented execution is selected by typed executor entry points across
    // all backends. RVR already selects it through the compiled execution kind.
    pub suspend_on_segment: bool,
}

#[derive(Clone, Debug)]
pub struct MeteredCtx {
    pub config: MeteredCtxConfig,
    pub trace_heights: Vec<u32>,
    pub memory_ctx: MemoryCtx,
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

impl MeteredCtx {
    // Note: prefer to use `build_metered_ctx` in `VmExecutor` or `VirtualMachine`.
    pub fn new(
        inputs: MeteredCtxInputs<'_>,
        config: &SystemConfig,
        memory_config: ProvingMemoryConfig,
    ) -> Self {
        let (mut trace_heights, is_trace_height_constant): (Vec<u32>, Vec<bool>) = inputs
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

        let segmentation_config = SegmentationConfig::new(
            inputs.air_names.to_vec(),
            inputs.widths.to_vec(),
            inputs.interactions.to_vec(),
            inputs.need_rot.to_vec(),
            inputs.segmentation_limits,
            memory_config,
        );
        let initial_trace_heights = trace_heights.clone();
        let mut memory_ctx = MemoryCtx::new(config);
        memory_ctx.add_register_merkle_heights();
        memory_ctx.apply_height_updates(&mut trace_heights);
        memory_ctx.update_checkpoint();
        let segmentation_ctx = SegmentationCtx::new(
            segmentation_config,
            &trace_heights,
            &is_trace_height_constant,
        );

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
        debug_assert!(air_names.len() >= 2);
        let poseidon2_idx = air_names.len() - 2;
        debug_assert!(
            air_names[poseidon2_idx].contains("Poseidon"),
            "air_name={}",
            air_names[poseidon2_idx]
        );
        Self {
            config: MeteredCtxConfig {
                initial_trace_heights,
                is_trace_height_constant,
                suspend_on_segment: false,
            },
            trace_heights,
            memory_ctx,
            segmentation_ctx,
        }
    }

    pub fn with_max_memory(mut self, max_memory: usize) -> Self {
        self.segmentation_ctx.set_max_memory(max_memory);
        self
    }

    pub fn seed_initial_memory(&mut self, initial_memory: &SparseMemoryImage) {
        self.memory_ctx.seed_initial_memory(initial_memory);
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
        let mut memory_ctx = MemoryCtx::new(system_config);
        let mut trace_heights = config.initial_trace_heights.clone();
        memory_ctx.add_register_merkle_heights();
        memory_ctx.apply_height_updates(&mut trace_heights);
        memory_ctx.update_checkpoint();
        let segmentation_ctx = SegmentationCtx::new(
            segmentation_config,
            &trace_heights,
            &config.is_trace_height_constant,
        );
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

    #[inline(always)]
    pub fn check_and_segment(&mut self) -> bool {
        // We track the segmentation check by instrets_until_check instead of instret in order to
        // save a register in native execution modes.
        if self.segmentation_ctx.instrets_until_check > 0 {
            return false;
        }
        self.segmentation_ctx.instrets_until_check = u64::from(SEGMENT_CHECK_INSNS);
        self.segmentation_ctx.instret += u64::from(SEGMENT_CHECK_INSNS);

        self.memory_ctx
            .apply_height_updates(&mut self.trace_heights);
        let did_segment = self
            .segmentation_ctx
            .check_and_segment(self.segmentation_ctx.instret, &mut self.trace_heights);

        if did_segment {
            // Initialize contexts for new segment
            self.segmentation_ctx
                .initialize_segment(&mut self.trace_heights);
            self.memory_ctx.initialize_segment(&mut self.trace_heights);

            // Check if the new segment is within limits
            self.segmentation_ctx
                .warn_if_exceeds_limits(self.segmentation_ctx.instret, &self.trace_heights);
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

impl ExecutionCtxTrait for MeteredCtx {
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
    fn should_suspend(exec_state: &mut VmExecState<GuestMemory, Self>) -> bool {
        // If `segment_suspend` is set, suspend when a segment is determined (but the VM state might
        // be after the segment boundary because the segment happens in the previous checkpoint).
        // Otherwise, execute until termination.
        if exec_state.ctx.segmentation_ctx.instrets_until_check > 0 {
            exec_state.ctx.segmentation_ctx.instrets_until_check -= 1;
            return false;
        }
        if exec_state.ctx.check_and_segment() && exec_state.ctx.config.suspend_on_segment {
            true
        } else {
            exec_state.ctx.segmentation_ctx.instrets_until_check -= 1;
            false
        }
    }

    #[inline(always)]
    fn on_terminate(exec_state: &mut VmExecState<GuestMemory, Self>) {
        exec_state
            .ctx
            .memory_ctx
            .apply_height_updates(&mut exec_state.ctx.trace_heights);
        exec_state
            .ctx
            .segmentation_ctx
            .create_final_segment(&exec_state.ctx.trace_heights);
    }
}

impl MeteredExecutionCtxTrait for MeteredCtx {
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
