use getset::WithSetters;
use openvm_instructions::riscv::RV32_IMM_AS;

use crate::{
    arch::{
        execution_mode::metered::segment_ctx::DEFAULT_MAX_MEMORY as DEFAULT_SEGMENT_MAX_MEMORY,
        ExecutionCtxTrait, MeteredExecutionCtxTrait, VmExecState,
    },
    system::memory::online::GuestMemory,
};

const DEFAULT_MAX_SEGMENTS: u64 = 100;
pub const DEFAULT_MAX_COST: u64 = DEFAULT_MAX_SEGMENTS * DEFAULT_SEGMENT_MAX_MEMORY as u64;

#[derive(Clone, Debug, WithSetters)]
pub struct MeteredCostCtx {
    pub widths: Vec<usize>,
    #[getset(set_with = "pub")]
    pub max_execution_cost: u64,
    // Cost is number of trace cells (height * width)
    pub cost: u64,
    /// To measure instructions/s
    pub instret: u64,
}

impl MeteredCostCtx {
    pub fn new(widths: Vec<usize>) -> Self {
        Self {
            widths,
            max_execution_cost: DEFAULT_MAX_COST,
            cost: 0,
            instret: 0,
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
        debug_assert!(size > 0, "size must be greater than 0, got {size}");
        debug_assert!(
            size.is_power_of_two(),
            "size must be a power of 2, got {size}"
        );
        // Prevent unbounded memory accesses per instruction
        if self.cost > 2 * std::cmp::max(self.max_execution_cost, DEFAULT_MAX_COST) {
            self.panic_cost_exceeded();
        }
    }

    #[inline(always)]
    fn should_suspend<F>(exec_state: &mut VmExecState<F, GuestMemory, Self>) -> bool {
        if exec_state.ctx.cost > exec_state.ctx.max_execution_cost {
            true
        } else {
            exec_state.ctx.instret += 1;
            false
        }
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
