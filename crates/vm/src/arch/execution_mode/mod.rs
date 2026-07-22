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

pub trait ExecutionCtxTrait: Sized {
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: u32, is_write: bool);

    fn should_suspend(exec_state: &mut VmExecState<GuestMemory, Self>) -> bool;

    fn on_terminate(_exec_state: &mut VmExecState<GuestMemory, Self>) {}
}

pub trait MeteredExecutionCtxTrait: ExecutionCtxTrait {
    fn on_height_change(&mut self, chip_idx: usize, height_delta: u32);
}
