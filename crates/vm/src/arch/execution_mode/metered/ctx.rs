use openvm_instructions::riscv::{
    RV32_IMM_AS, RV32_NUM_REGISTERS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS,
};

use super::{
    memory_ctx::MemoryCtx,
    segment_ctx::{Segment, SegmentationCtx},
};
use crate::{arch::execution_mode::E1E2ExecutionCtx, system::memory::dimensions::MemoryDimensions};

#[derive(Debug)]
pub struct MeteredCtx<const PAGE_BITS: usize = 6> {
    pub trace_heights: Vec<u32>,
    pub is_trace_height_constant: Vec<bool>,

    pub memory_ctx: MemoryCtx<PAGE_BITS>,
    pub segmentation_ctx: SegmentationCtx,
    pub continuations_enabled: bool,
}

impl<const PAGE_BITS: usize> MeteredCtx<PAGE_BITS> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        constant_trace_heights: Vec<Option<usize>>,
        has_public_values_chip: bool,
        continuations_enabled: bool,
        as_byte_alignment_bits: Vec<u8>,
        memory_dimensions: MemoryDimensions,
        air_names: Vec<String>,
        widths: Vec<usize>,
        interactions: Vec<usize>,
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

        let memory_ctx = MemoryCtx::new(
            has_public_values_chip,
            continuations_enabled,
            as_byte_alignment_bits,
            memory_dimensions,
        );

        // Assert that the indices are correct
        debug_assert_eq!(&air_names[memory_ctx.boundary_idx], "Boundary");
        if let Some(merkle_tree_index) = memory_ctx.merkle_tree_index {
            debug_assert_eq!(&air_names[merkle_tree_index], "Merkle");
        }
        debug_assert_eq!(&air_names[memory_ctx.adapter_offset], "AccessAdapter<2>");

        let segmentation_ctx = SegmentationCtx::new(air_names, widths, interactions);

        let mut ctx = Self {
            trace_heights,
            is_trace_height_constant,
            memory_ctx,
            segmentation_ctx,
            continuations_enabled,
        };

        // Add merkle height contributions for all registers
        ctx.add_register_merkle_heights();

        ctx
    }

    fn add_register_merkle_heights(&mut self) {
        if self.continuations_enabled {
            self.memory_ctx.update_boundary_merkle_heights(
                &mut self.trace_heights,
                RV32_REGISTER_AS,
                0,
                (RV32_NUM_REGISTERS * RV32_REGISTER_NUM_LIMBS) as u32,
            );
        }
    }

    pub fn with_max_trace_height(mut self, max_trace_height: u32) -> Self {
        self.segmentation_ctx.set_max_trace_height(max_trace_height);
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

    pub fn with_segment_check_insns(mut self, segment_check_insns: u64) -> Self {
        self.segmentation_ctx
            .set_segment_check_insns(segment_check_insns);
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
        self.add_register_merkle_heights();
    }

    pub fn check_and_segment(&mut self, instret: u64) {
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

impl<const PAGE_BITS: usize> E1E2ExecutionCtx for MeteredCtx<PAGE_BITS> {
    #[inline(always)]
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: u32) {
        debug_assert!(
            address_space != RV32_IMM_AS,
            "address space must not be immediate"
        );
        debug_assert!(
            size.is_power_of_two(),
            "size must be a power of 2, got {}",
            size
        );

        // Handle access adapter updates
        let size_bits = size.ilog2();
        self.memory_ctx
            .update_adapter_heights(&mut self.trace_heights, address_space, size_bits);

        // Handle merkle tree updates
        if address_space != RV32_REGISTER_AS {
            self.memory_ctx.update_boundary_merkle_heights(
                &mut self.trace_heights,
                address_space,
                ptr,
                size,
            );
        }
    }
}
