use crate::{arch::VmExecState, system::memory::online::GuestMemory};

pub mod metered;
pub mod normal;
pub mod preflight;

pub trait ExecutionCtxTrait: Sized {
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: u32);
    fn should_suspend<F>(vm_state: &mut VmExecState<F, GuestMemory, Self>) -> bool;
    fn on_terminate<F>(_vm_state: &mut VmExecState<F, GuestMemory, Self>) {}
}

pub trait MeteredExecutionCtxTrait: ExecutionCtxTrait {
    fn on_height_change(&mut self, chip_idx: usize, height_delta: u32);
}
