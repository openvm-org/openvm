use openvm_instructions::SysPhantom;

use crate::{arch::VmExecState, system::memory::online::GuestMemory};

pub mod metered;
pub mod metered_cost;
mod preflight;
mod pure;

pub use metered::{
    ctx::{MeteredCtx, MeteredCtxConfig, MeteredCtxInputs},
    segment_ctx::{Segment, SegmentationConfig, SegmentationLimits},
};
pub use metered_cost::MeteredCostCtx;
pub use preflight::PreflightCtx;
pub use pure::ExecutionCtx;

/// Hooks used by shared instruction handlers to notify an execution mode about observable state.
/// Default no-op hooks let each mode implement only the events it records.
pub trait ExecutionCtxTrait: Sized {
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: u32, is_write: bool);

    #[inline(always)]
    fn on_memory_read(
        &mut self,
        _memory: &GuestMemory,
        _address_space: u32,
        _byte_ptr: u32,
        _byte_len: u32,
    ) {
    }

    #[inline(always)]
    fn on_memory_write_start(
        &mut self,
        _memory: &GuestMemory,
        _address_space: u32,
        _byte_ptr: u32,
        _byte_len: u32,
    ) {
    }

    #[inline(always)]
    fn on_memory_write_end(&mut self, _memory: &GuestMemory) {}

    #[inline(always)]
    fn on_instruction_start(_exec_state: &mut VmExecState<GuestMemory, Self>, _pc: u32) {}

    #[inline(always)]
    fn advance_timestamp(&mut self, _slots: u32) {}

    #[inline(always)]
    fn on_system_phantom(
        _exec_state: &mut VmExecState<GuestMemory, Self>,
        _pc: u32,
        _phantom: SysPhantom,
    ) {
    }

    fn should_suspend(exec_state: &mut VmExecState<GuestMemory, Self>) -> bool;

    fn on_terminate(_exec_state: &mut VmExecState<GuestMemory, Self>) {}
}

pub trait MeteredExecutionCtxTrait: ExecutionCtxTrait {
    fn on_height_change(&mut self, chip_idx: usize, height_delta: u32);
}
