use crate::{
    arch::{execution_mode::ExecutionCtxTrait, VmExecState},
    system::memory::online::GuestMemory,
};

#[repr(C)]
pub struct ExecutionCtx {
    pub instret_left: u64,
}

impl ExecutionCtx {
    pub(crate) fn new(instret_left: Option<u64>) -> Self {
        ExecutionCtx {
            instret_left: instret_left.unwrap_or(u64::MAX),
        }
    }
}

impl ExecutionCtxTrait for ExecutionCtx {
    #[inline(always)]
    fn on_memory_operation(&mut self, _address_space: u32, _ptr: u32, _size: u32, _is_write: bool) {
    }

    #[inline(always)]
    fn should_suspend(exec_state: &mut VmExecState<GuestMemory, Self>) -> bool {
        if exec_state.ctx.instret_left == 0 {
            true
        } else {
            exec_state.ctx.instret_left -= 1;
            false
        }
    }
}
