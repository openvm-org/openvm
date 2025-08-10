use crate::{
    arch::{execution_mode::E1ExecutionCtx, VmExecState},
    system::memory::online::GuestMemory,
};

pub struct E1Ctx {
    instret_end: u64,
}

impl E1Ctx {
    pub fn new(instret_end: Option<u64>) -> Self {
        E1Ctx {
            instret_end: if let Some(end) = instret_end {
                end
            } else {
                u64::MAX
            },
        }
    }
}

impl Default for E1Ctx {
    fn default() -> Self {
        Self::new(None)
    }
}

impl E1ExecutionCtx for E1Ctx {
    #[inline(always)]
    fn on_memory_operation(&mut self, _address_space: u32, _ptr: u32, _size: u32) {}
    #[inline(always)]
    fn should_suspend<F>(vm_state: &mut VmExecState<F, GuestMemory, Self>) -> bool {
        vm_state.instret >= vm_state.ctx.instret_end
    }
}
