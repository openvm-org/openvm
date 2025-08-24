use crate::{
    arch::{execution_mode::ExecutionCtxTrait, VmExecState},
    system::memory::online::GuestMemory,
};

pub struct ExecutionCtx {
    instret_end: u64,
}

impl ExecutionCtx {
    pub fn new(instret_end: Option<u64>) -> Self {
        ExecutionCtx {
            instret_end: if let Some(end) = instret_end {
                end
            } else {
                u64::MAX
            },
        }
    }
}

impl ExecutionCtxTrait for ExecutionCtx {
    #[inline(always)]
    fn on_memory_operation(&mut self, _address_space: u32, _ptr: u32, _size: u32) {}
    #[inline(always)]
    fn should_suspend<F>(vm_state: &mut VmExecState<F, GuestMemory, Self>) -> bool {
        vm_state.instret >= vm_state.ctx.instret_end
    }
}
