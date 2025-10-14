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
use openvm_circuit::arch::execution_mode::MeteredCtx;

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
pub extern "C" fn set_instret_and_pc(
    base: *mut c_void,
    instret: u64,
    pc_val: u32,
) {
    type F = BabyBear;
    type Ctx = ExecutionCtx;
    let vm_exec_state_ref = unsafe {
        &mut *(base as *mut VmExecState<F, GuestMemory, Ctx>)
    };
    vm_exec_state_ref.vm_state.set_instret_and_pc(instret, pc_val);
}

#[no_mangle]
pub extern "C" fn extern_handler(
    base: *mut c_void,
    vec_ptr: *const c_void,
    pc_val: u32,
) -> u32 {
    type F = BabyBear;
    type Ctx = ExecutionCtx;

    let mut instret: Box<u64> = Box::new(0); // placeholder to call the handler function
    let mut pc: Box<u32> = Box::new(pc_val);

    let vm_exec_state_ref = unsafe {
        &mut *(base as *mut VmExecState<F, GuestMemory, Ctx>)
    };

    unsafe {
        let typed_ptr = vec_ptr as *const Vec<PreComputeInstruction<'_, F, Ctx>>;
        let pre_compute_insns = &*typed_ptr;
        let fi = &pre_compute_insns[(pc_val / 4) as usize];
        let arg : u64 = 0; // todo: change this later
        (fi.handler)(fi.pre_compute, &mut instret, &mut pc, arg, vm_exec_state_ref);
    };

    match vm_exec_state_ref.exit_code {
        Ok(None) => {
            return *pc;
            // execution continues 
        }
        _ => {
            // special indicator that we must terminate
            // this won't collide with actual pc value because pc values are always multiple of 4
            return 1;
        }
    }
}

#[no_mangle]
pub extern "C" fn metered_extern_handler(
    base: *mut c_void,
    vec_ptr: *const c_void,
    pc_val: u32,
) -> u32 {
    let mut instret: Box<u64> = Box::new(0);
    let mut pc: Box<u32> = Box::new(pc_val);

    unsafe {
        type F = BabyBear;
        type Ctx = MeteredCtx;
        let typed_ptr = vec_ptr as *const Vec<PreComputeInstruction<'_, F, Ctx>>;
        let pre_compute_insns = &*typed_ptr;
        let fi = &pre_compute_insns[(pc_val / 4) as usize];
        let arg : u64 = 0; // todo: change this later

        let vm_exec_state_ref = unsafe {
            &mut *(base as *mut VmExecState<F, GuestMemory, Ctx>)
        };
        (fi.handler)(fi.pre_compute, &mut instret, &mut pc, arg, vm_exec_state_ref);
    };

    return *pc;
}