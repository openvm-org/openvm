use crate::{arch::VmExecState, system::memory::online::GuestMemory};

pub mod metered;
pub mod metered_cost;
mod preflight;
mod pure;

pub use metered::{ctx::MeteredCtx, segment_ctx::Segment};
pub use metered_cost::MeteredCostCtx;
pub use preflight::PreflightCtx;
pub use pure::ExecutionCtx;

pub trait ExecutionCtxTrait: Sized {
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: u32);
    fn should_suspend<F>(vm_state: &mut VmExecState<F, GuestMemory, Self>) -> bool;
    fn on_terminate<F>(_vm_state: &mut VmExecState<F, GuestMemory, Self>) {}
}

pub trait MeteredExecutionCtxTrait: ExecutionCtxTrait {
    fn on_height_change(&mut self, chip_idx: usize, height_delta: u32);
}
