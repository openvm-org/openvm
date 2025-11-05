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
    fn should_suspend<F>(exec_state: &mut VmExecState<F, GuestMemory, Self>) -> bool {
        // ATTENTION: Please make sure to update the corresponding logic in the
        // `asm_bridge` crate and `aot.rs`` when you change this function.
        if exec_state.ctx.instret_left == 0 {
            true
        } else {
            exec_state.ctx.instret_left -= 1;
            false
        }
    }
}
