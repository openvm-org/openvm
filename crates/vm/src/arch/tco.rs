#![cfg(target_arch = "x86_64")]
use core::arch::global_asm;
use openvm_stark_backend::p3_field::PrimeField32;
use super::{execution_mode::E1ExecutionCtx, VmSegmentState, execution::PreComputeInstruction};
use crate::system::memory::online::GuestMemory;

type ErasedExecuteFunc = extern "C-unwind" fn(*const u8, *mut u8);

unsafe fn execute_one_instruction_impl<F: PrimeField32, Ctx: E1ExecutionCtx>(
    handler: unsafe fn(&[u8], &mut VmSegmentState<F, GuestMemory, Ctx>),
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, Ctx>,
) {
    handler(pre_compute, vm_state);
}

#[no_mangle]
extern "C-unwind" fn tco_execute_one_instruction(
    handler_ptr: *const u8,
    pre_compute_ptr: *const u8,
    pre_compute_len: usize,
    vm_state_ptr: *mut u8,
) {
    unsafe {
        let handler: unsafe fn(&[u8], &mut VmSegmentState<p3_baby_bear::BabyBear, GuestMemory, super::execution_mode::e1::E1Ctx>) = 
            std::mem::transmute(handler_ptr);
        let pre_compute = std::slice::from_raw_parts(pre_compute_ptr, pre_compute_len);
        let vm_state = &mut *(vm_state_ptr as *mut VmSegmentState<p3_baby_bear::BabyBear, GuestMemory, super::execution_mode::e1::E1Ctx>);
        
        handler(pre_compute, vm_state);
    }
}

macro_rules! tco_stub {
    ($asm_name:ident) => {
        global_asm!(
            concat!(
                ".text\n",
                ".globl ", stringify!($asm_name), "\n",
                ".type  ", stringify!($asm_name), ",@function\n",
                stringify!($asm_name), ":\n",
                "  .cfi_startproc\n",
                "  push rbp\n",
                "  .cfi_def_cfa_offset 16\n",
                "  .cfi_offset rbp, -16\n",
                "  mov rbp, rsp\n",
                "  .cfi_def_cfa_register rbp\n",
                "  call tco_execute_one_instruction\n",
                "  pop rbp\n",
                "  ret\n",
                "  .cfi_endproc\n",
            ),
        );
    };
}

tco_stub!(tco_instruction_handler);

extern "C-unwind" {
    fn tco_instruction_handler(
        handler_ptr: *const u8,
        pre_compute_ptr: *const u8,
        pre_compute_len: usize,
        vm_state_ptr: *mut u8,
    );
}

pub unsafe fn execute_instruction_with_tco<F: PrimeField32, Ctx: E1ExecutionCtx>(
    inst: &PreComputeInstruction<F, Ctx>,
    vm_state: &mut VmSegmentState<F, GuestMemory, Ctx>,
) {
    tco_instruction_handler(
        inst.handler as *const u8,
        inst.pre_compute.as_ptr(),
        inst.pre_compute.len(),
        vm_state as *mut VmSegmentState<F, GuestMemory, Ctx> as *mut u8,
    );
}
