use std::ffi::c_void;
use core::arch::global_asm;

use openvm_circuit::arch::VmState;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_circuit::arch::MemoryConfig;
use openvm_circuit::arch::SystemConfig;

use openvm_circuit::arch::VmExecState;
use openvm_circuit::arch::execution_mode::ExecutionCtx;  
use openvm_circuit::system::memory::online::GuestMemory;

mod rv32im;

pub use rv32im::*;

type F = BabyBear;

#[no_mangle]
pub extern "C" fn TEST_FN(base: *mut c_void) {
    println!("TEST_FN called with base: {:?} but do nothing", base);
}

pub fn read_memory<const LEN: usize>(
    base: *mut c_void, 
    address_space: u32, 
    ptr: u32,
) -> [u8; LEN] {
    // TODO: this is a temporary fix
    if address_space == 0 {
        let mut result: [u8; LEN] = [0; LEN];
        result[..4].copy_from_slice(&ptr.to_le_bytes());
        return result
    }
    let vm_exec_state_ref = unsafe {
        &mut *(base as *mut VmExecState<F, GuestMemory, ExecutionCtx>)
    };
    vm_exec_state_ref.vm_read::<u8, LEN>(address_space, ptr)
}

pub fn write_memory<const LEN: usize>(
    base: *mut c_void, 
    data: [u8; LEN], 
    address_space: u32, 
    ptr: u32,
) {
    let vm_exec_state_ref = unsafe {
        &mut *(base as *mut VmExecState<F, GuestMemory, ExecutionCtx>)
    };
    vm_exec_state_ref.vm_write::<u8, LEN>(address_space, ptr, &data);
}

global_asm!(include_str!("asm_run.s"));

extern "C" {
    fn asm_run_internal(base: *mut c_void);
}

#[no_mangle]
pub unsafe extern "C" fn asm_run(base: *mut c_void) {
    asm_run_internal(base);
}