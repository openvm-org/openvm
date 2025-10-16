use core::arch::global_asm;
use std::ffi::c_void;

use openvm_circuit::{
    arch::{
        execution_mode::{ExecutionCtx, MeteredCtx},
        interpreter::PreComputeInstruction,
        MemoryConfig, SystemConfig, VmExecState, VmState,
    },
    system::memory::online::GuestMemory,
};
use openvm_instructions::program::DEFAULT_PC_STEP;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

// asm_run.s contains the assembly to run metered execution
global_asm!(include_str!("asm_run.s"));

/*
rbx = vm_exec_state
rbp = pre_compute_insns
r13 = from_state_pc
r14 = from_state_instret
*/

extern "C" {
    fn asm_run_internal(
        vm_exec_state_ptr: *mut c_void,       // rdi = vm_exec_state
        pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
        from_state_pc: u32,                   // rdx = from_state.pc
        from_state_instret: u64,              // rcx = from_state.instret
    );
}

#[no_mangle]
pub unsafe extern "C" fn asm_run(
    vm_exec_state_ptr: *mut c_void,
    pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
    from_state_pc: u32,
    from_state_instret: u64,
) {
    asm_run_internal(
        vm_exec_state_ptr,
        pre_compute_insns_ptr,
        from_state_pc,
        from_state_instret,
    );
}

type F = BabyBear;

// at the end of the execution, you want to store the instret and pc from the x86 registers
// to update the vm state's pc and instret
// works for metered execution
#[no_mangle]
pub extern "C" fn metered_set_instret_and_pc(
    vm_exec_state_ptr: *mut c_void,       // rdi = vm_exec_state
    pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
    final_pc: u32,                        // rdx = final_pc
    final_instret: u64,                   // rcx = final_instret
) {
    type Ctx = MeteredCtx;
    // reference to vm_exec_state
    let vm_exec_state_ref =
        unsafe { &mut *(vm_exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };
    vm_exec_state_ref
        .vm_state
        .set_instret_and_pc(final_instret, final_pc);
}

#[no_mangle]
pub extern "C" fn metered_extern_handler(
    vm_exec_state_ptr: *mut c_void,
    pre_compute_insns_ptr: *const c_void,
    cur_pc: u32,
    cur_instret: u64,
) -> u32 {
    println!("[AOT] cur_pc {} cur_instret {}", cur_pc, cur_instret);
    type F = BabyBear;
    type Ctx = MeteredCtx;

    let mut instret: Box<u64> = Box::new(cur_instret); // placeholder to call the handler function
    let mut pc: Box<u32> = Box::new(cur_pc);

    let vm_exec_state_ref =
        unsafe { &mut *(vm_exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };

    // pointer to the first element of `pre_compute_insns`
    let pre_compute_insns_base_ptr =
        pre_compute_insns_ptr as *const PreComputeInstruction<'static, F, Ctx>;
    let pc_idx = (cur_pc / DEFAULT_PC_STEP) as usize;

    let pre_compute_insns = unsafe { &*pre_compute_insns_base_ptr.add(pc_idx) };

    let ctx = &vm_exec_state_ref.ctx;
    // `arg` is a runtime constant that we want to keep in register
    // - For metered execution it is `segment_check_insns`
    let arg = ctx.segmentation_ctx.segment_check_insns;

    unsafe {
        (pre_compute_insns.handler)(
            pre_compute_insns.pre_compute,
            &mut instret,
            &mut pc,
            arg,
            vm_exec_state_ref,
        );
    };

    match vm_exec_state_ref.exit_code {
        Ok(None) => {
            // execution continues
            *pc
        },
        _ => {
            // special indicator that we must terminate
            // this won't collide with actual pc value because pc values are always multiple of 4
            1
        }
    }
}

#[no_mangle]
pub extern "C" fn should_suspend(
    instret: u64, 
    _pc: u32, 
    exec_state_ptr: *mut c_void
) -> u32 {
    type Ctx = MeteredCtx;

    let exec_state_ref = unsafe { &mut *(exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };

    let segment_check_insns = exec_state_ref.ctx.segmentation_ctx.segment_check_insns;

    if exec_state_ref
        .ctx
        .check_and_segment(instret, segment_check_insns)
        && *exec_state_ref.ctx.suspend_on_segment()
    {
        1
    } else {
        0
    }
}
