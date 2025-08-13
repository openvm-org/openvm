use std::num::NonZero;

use openvm_instructions::riscv::{RV32_IMM_AS, RV32_REGISTER_AS};

use super::{
    memory_ctx::MemoryCtx,
    segment_ctx::{Segment, SegmentationCtx},
};
use crate::{
    arch::{
        execution_mode::{ExecutionCtxTrait, MeteredExecutionCtxTrait},
        SystemConfig, VmExecState,
    },
    system::memory::online::GuestMemory,
};

pub const DEFAULT_PAGE_BITS: usize = 6;

#[derive(Clone, Debug)]
pub struct MeteredCtx<const PAGE_BITS: usize = DEFAULT_PAGE_BITS> {
    pub trace_heights: Vec<u32>,
    pub is_trace_height_constant: Vec<bool>,
    pub memory_ctx: MemoryCtx<PAGE_BITS>,
    pub segmentation_ctx: SegmentationCtx,
}

impl<const PAGE_BITS: usize> MeteredCtx<PAGE_BITS> {
    // Note[jpw]: prefer to use `build_metered_ctx` in `VmExecutor` or `VirtualMachine`.
    pub fn new(
        constant_trace_heights: Vec<Option<usize>>,
        air_names: Vec<String>,
        widths: Vec<usize>,
        interactions: Vec<usize>,
        config: &SystemConfig,
    ) -> Self {
        let (trace_heights, is_trace_height_constant): (Vec<u32>, Vec<bool>) =
            constant_trace_heights
                .iter()
                .map(|&constant_height| {
                    if let Some(height) = constant_height {
                        (height as u32, true)
                    } else {
                        (0, false)
                    }
                })
                .unzip();

        let memory_ctx = MemoryCtx::new(config);

        // Assert that the indices are correct
        debug_assert!(
            air_names[memory_ctx.boundary_idx].contains("Boundary"),
            "air_name={}",
            air_names[memory_ctx.boundary_idx]
        );
        if let Some(merkle_tree_index) = memory_ctx.merkle_tree_index {
            debug_assert!(
                air_names[merkle_tree_index].contains("Merkle"),
                "air_name={}",
                air_names[merkle_tree_index]
            );
        }
        debug_assert!(
            air_names[memory_ctx.adapter_offset].contains("AccessAdapterAir<2>"),
            "air_name={}",
            air_names[memory_ctx.adapter_offset]
        );

        let segmentation_ctx =
            SegmentationCtx::new(air_names, widths, interactions, config.segmentation_limits);

        let mut ctx = Self {
            trace_heights,
            is_trace_height_constant,
            memory_ctx,
            segmentation_ctx,
        };
        if !config.continuation_enabled {
            // force single segment
            ctx.segmentation_ctx.segment_check_insns = u64::MAX;
        }

        // Add merkle height contributions for all registers
        ctx.memory_ctx.add_register_merkle_heights();

        ctx
    }

    pub fn with_max_trace_height(mut self, max_trace_height: u32) -> Self {
        self.segmentation_ctx.set_max_trace_height(max_trace_height);
        let max_check_freq = (max_trace_height / 2) as u64;
        if max_check_freq < self.segmentation_ctx.segment_check_insns {
            self.segmentation_ctx.segment_check_insns = max_check_freq;
        }
        self
    }

    pub fn with_max_cells(mut self, max_cells: usize) -> Self {
        self.segmentation_ctx.set_max_cells(max_cells);
        self
    }

    pub fn with_max_interactions(mut self, max_interactions: usize) -> Self {
        self.segmentation_ctx.set_max_interactions(max_interactions);
        self
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segmentation_ctx.segments
    }

    pub fn into_segments(self) -> Vec<Segment> {
        self.segmentation_ctx.segments
    }

    fn reset_segment(&mut self) {
        self.memory_ctx.clear();
        for (i, &is_constant) in self.is_trace_height_constant.iter().enumerate() {
            if !is_constant {
                self.trace_heights[i] = 0;
            }
        }
        // Add merkle height contributions for all registers
        self.memory_ctx.add_register_merkle_heights();
    }

    #[inline(always)]
    pub fn check_and_segment(&mut self, instret: u64) {
        let threshold = self
            .segmentation_ctx
            .instret_last_segment_check
            .wrapping_add(self.segmentation_ctx.segment_check_insns);
        debug_assert!(
            threshold >= self.segmentation_ctx.instret_last_segment_check,
            "overflow in segment check threshold calculation"
        );
        if instret < threshold {
            return;
        }

        self.memory_ctx
            .lazy_update_boundary_heights(&mut self.trace_heights);
        let did_segment = self.segmentation_ctx.check_and_segment(
            instret,
            &self.trace_heights,
            &self.is_trace_height_constant,
        );

        if did_segment {
            self.reset_segment();
        }
    }

    #[allow(dead_code)]
    pub fn print_heights(&self) {
        println!("{:>10} {:<30}", "Height", "Air Name");
        println!("{}", "-".repeat(42));
        for (i, height) in self.trace_heights.iter().enumerate() {
            let air_name = self
                .segmentation_ctx
                .air_names
                .get(i)
                .map(|s| s.as_str())
                .unwrap_or("Unknown");
            println!("{:>10} {:<30}", height, air_name);
        }
    }
}

impl<const PAGE_BITS: usize> ExecutionCtxTrait for MeteredCtx<PAGE_BITS> {
    #[inline(always)]
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: u32) {
        debug_assert!(
            address_space != RV32_IMM_AS,
            "address space must not be immediate"
        );
        debug_assert!(size > 0, "size must be greater than 0, got {}", size);
        debug_assert!(
            size.is_power_of_two(),
            "size must be a power of 2, got {}",
            size
        );

        // Handle access adapter updates
        // SAFETY: size passed is always a non-zero power of 2
        let size_bits = unsafe { NonZero::new_unchecked(size).ilog2() };
        self.memory_ctx
            .update_adapter_heights(&mut self.trace_heights, address_space, size_bits);

        // Handle merkle tree updates
        if address_space != RV32_REGISTER_AS {
            self.memory_ctx
                .update_boundary_merkle_heights(address_space, ptr, size);
        }
    }

    #[inline(always)]
    fn should_suspend<F>(vm_state: &mut VmExecState<F, GuestMemory, Self>) -> bool {
        // E2 always runs until termination. Here we use the function as a hook called every
        // instruction.
        vm_state.ctx.check_and_segment(vm_state.instret);
        false
    }

    #[inline(always)]
    fn on_terminate<F>(vm_state: &mut VmExecState<F, GuestMemory, Self>) {
        vm_state
            .ctx
            .memory_ctx
            .lazy_update_boundary_heights(&mut vm_state.ctx.trace_heights);
        vm_state
            .ctx
            .segmentation_ctx
            .segment(vm_state.instret, &vm_state.ctx.trace_heights);
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
