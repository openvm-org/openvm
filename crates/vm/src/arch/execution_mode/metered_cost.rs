use std::num::NonZero;

use getset::WithSetters;
use openvm_instructions::riscv::RV32_IMM_AS;

use crate::{
    arch::{
        execution_mode::metered::segment_ctx::DEFAULT_MAX_CELLS as DEFAULT_SEGMENT_MAX_CELLS,
        ExecutionCtxTrait, MeteredExecutionCtxTrait, SystemConfig, VmExecState,
    },
    system::memory::online::GuestMemory,
};

const DEFAULT_MAX_SEGMENTS: u64 = 100;
pub const DEFAULT_MAX_COST: u64 = DEFAULT_MAX_SEGMENTS * DEFAULT_SEGMENT_MAX_CELLS as u64;

#[derive(Clone, Debug)]
pub struct AccessAdapterCtx {
    min_block_size_bits: Vec<u8>,
    idx_offset: usize,
}

impl AccessAdapterCtx {
    pub fn new(config: &SystemConfig) -> Self {
        Self {
            min_block_size_bits: config.memory_config.min_block_size_bits(),
            idx_offset: config.access_adapter_air_id_offset(),
        }
    }

    #[inline(always)]
    pub fn update_cells(
        &self,
        cost: &mut u64,
        address_space: u32,
        size_bits: u32,
        widths: &[usize],
    ) {
        debug_assert!((address_space as usize) < self.min_block_size_bits.len());

        // SAFETY: address_space passed is usually a hardcoded constant or derived from an
        // Instruction where it is bounds checked before passing
        let align_bits = unsafe {
            *self
                .min_block_size_bits
                .get_unchecked(address_space as usize)
        };
        debug_assert!(
            align_bits as u32 <= size_bits,
            "align_bits ({}) must be <= size_bits ({})",
            align_bits,
            size_bits
        );

        for adapter_bits in (align_bits as u32 + 1..=size_bits).rev() {
            let adapter_idx = self.idx_offset + adapter_bits as usize - 1;
            debug_assert!(adapter_idx < widths.len());
            // SAFETY: widths is initialized taking access adapters into account
            let width = unsafe { *widths.get_unchecked(adapter_idx) };
            let height_delta = 1 << (size_bits - adapter_bits + 1);
            *cost += (height_delta as u64) * (width as u64);
        }
    }
}

#[derive(Clone, Debug, WithSetters)]
pub struct MeteredCostCtx {
    pub widths: Vec<usize>,
    pub access_adapter_ctx: AccessAdapterCtx,
    #[getset(set_with = "pub")]
    pub max_execution_cost: u64,
    // Cost is number of trace cells (height * width)
    pub cost: u64,
}

impl MeteredCostCtx {
    pub fn new(widths: Vec<usize>, config: &SystemConfig) -> Self {
        let access_adapter_ctx = AccessAdapterCtx::new(config);
        Self {
            widths,
            access_adapter_ctx,
            max_execution_cost: DEFAULT_MAX_COST,
            cost: 0,
        }
    }

    #[cold]
    fn panic_cost_exceeded(&self) -> ! {
        panic!(
            "Execution cost {} exceeded maximum allowed cost of {}",
            self.cost,
            2 * DEFAULT_MAX_COST
        );
    }
}

impl ExecutionCtxTrait for MeteredCostCtx {
    #[inline(always)]
    fn on_memory_operation(&mut self, address_space: u32, _ptr: u32, size: u32) {
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
        // Prevent unbounded memory accesses per instruction
        if self.cost > 2 * std::cmp::max(self.max_execution_cost, DEFAULT_MAX_COST) {
            self.panic_cost_exceeded();
        }

        // Handle access adapter updates
        // SAFETY: size passed is always a non-zero power of 2
        let size_bits = unsafe { NonZero::new_unchecked(size).ilog2() };
        self.access_adapter_ctx.update_cells(
            &mut self.cost,
            address_space,
            size_bits,
            &self.widths,
        );
    }

    #[inline(always)]
    fn should_suspend<F>(
        _instret: u64,
        _pc: u32,
        max_execution_cost: u64,
        exec_state: &mut VmExecState<F, GuestMemory, Self>,
    ) -> bool {
        exec_state.ctx.cost > max_execution_cost
    }
}

impl MeteredExecutionCtxTrait for MeteredCostCtx {
    #[inline(always)]
    fn on_height_change(&mut self, chip_idx: usize, height_delta: u32) {
        debug_assert!(chip_idx < self.widths.len(), "chip_idx out of bounds");
        // SAFETY: chip_idx is created in executor_idx_to_air_idx and is always within bounds
        let width = unsafe { *self.widths.get_unchecked(chip_idx) };
        self.cost += (height_delta as u64) * (width as u64);
    }
}
