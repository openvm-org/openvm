use std::ffi::c_void;
use core::arch::global_asm;

use openvm_circuit::arch::VmState;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_circuit::arch::MemoryConfig;
use openvm_circuit::arch::SystemConfig;

use openvm_circuit::arch::VmExecState;
use openvm_circuit::arch::execution_mode::ExecutionCtx;  
use openvm_circuit::system::memory::online::GuestMemory;
use openvm_circuit::arch::interpreter::PreComputeInstruction;

type F = BabyBear;

#[no_mangle]
pub extern "C" fn TEST_FN(
    base: *mut c_void,
    vec_ptr: *const c_void,
) {
    println!("TEST_FN");
}

global_asm!(include_str!("asm_run.s"));

extern "C" {
    // rdi=base, rsi=vec_ptr
    fn asm_run_internal(
        base: *mut c_void,
        vec_ptr: *const c_void,
    );
}

#[no_mangle]
pub unsafe extern "C" fn asm_run(
    base: *mut c_void,
    vec_ptr: *const c_void,
) {
    asm_run_internal(base, vec_ptr);
}

#[no_mangle]
pub extern "C" fn extern_handler(
    base: *mut c_void,
    vec_ptr: *const c_void,
) {
    unsafe {
        type F = BabyBear;
        type Ctx = ExecutionCtx;
        
        let typed_ptr = vec_ptr as *const Vec<PreComputeInstruction<'_, F, Ctx>>;
        let pre_compute_insns = &*typed_ptr;

        let fi = &pre_compute_insns[1];
        
        let mut instret: Box<u64> = Box::new(1);
        let mut pc: Box<u32> = Box::new(1);
        let arg : u64 = 10; // arg variable, change later

        let vm_exec_state_ref = unsafe {
            &mut *(base as *mut VmExecState<F, GuestMemory, ExecutionCtx>)
        };
        (fi.handler)(fi.pre_compute, &mut instret, &mut pc, arg, vm_exec_state_ref);

        println!("handler called with args {:?}", fi.pre_compute);
    };
}