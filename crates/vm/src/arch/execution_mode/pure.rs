use crate::{
    arch::{execution_mode::ExecutionCtxTrait, VmExecState},
    system::memory::online::GuestMemory,
};

#[repr(C)]
pub struct ExecutionCtx {
    pub instret_left: u64,
}

impl ExecutionCtx {
    pub fn new(instret_left: Option<u64>) -> Self {
        ExecutionCtx {
            instret_left: if let Some(end) = instret_left {
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
    fn should_suspend<F>(
        _instret: u64,
        _pc: u32,
        _instret_left: u64,
        exec_state: &mut VmExecState<F, GuestMemory, Self>,
    ) -> bool {
        exec_state.ctx.instret_left -= 1;
        exec_state.ctx.instret_left == 0
    }
}
