use std::{
    ffi::c_void,
    sync::atomic::{AtomicUsize, Ordering},
};

use openvm_circuit::{
    arch::{
        execution_mode::MeteredCtx, interpreter::PreComputeInstruction, ExecutionError, VmExecState,
    },
    system::memory::online::GuestMemory,
};
use openvm_instructions::program::DEFAULT_PC_STEP;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

/*
rbx = vm_exec_state
rbp = pre_compute_insns
r13 = from_state_pc
*/

extern "C" {
    fn asm_run_internal(
        vm_exec_state_ptr: *mut c_void,       // rdi = vm_exec_state
        pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
        from_state_pc: u32,                   // rdx = from_state.pc
        pre_compute_insns_len: u64,
    );
}

/// Runs the VM execution from assembly
///
/// # Safety
///
///
/// This function is unsafe because:
/// - `vm_exec_state_ptr` must be valid
/// - `pre_compute_insns` must point to valid pre-compute instructions
#[no_mangle]
pub unsafe extern "C" fn asm_run(
    vm_exec_state_ptr: *mut c_void,
    pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
    from_state_pc: u32,
    pre_compute_insns_len: u64,
) {
    PRE_COMPUTE_INSNS_LEN.store(pre_compute_insns_len as usize, Ordering::Relaxed);
    asm_run_internal(
        vm_exec_state_ptr,
        pre_compute_insns_ptr,
        from_state_pc,
        pre_compute_insns_len,
    );
    PRE_COMPUTE_INSNS_LEN.store(0, Ordering::Relaxed);
}

type F = BabyBear;
type Ctx = MeteredCtx;
static PRE_COMPUTE_INSNS_LEN: AtomicUsize = AtomicUsize::new(0);

// at the end of the execution, you want to store the instret and pc from the x86 registers
// to update the vm state's pc and instret
// works for metered execution
#[no_mangle]
pub unsafe extern "C" fn metered_set_pc(
    vm_exec_state_ptr: *mut c_void,        // rdi = vm_exec_state
    _pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
    final_pc: u32,                         // rdx = final_pc
) {
    // reference to vm_exec_state
    let vm_exec_state_ref =
        unsafe { &mut *(vm_exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };
    vm_exec_state_ref.vm_state.set_pc(final_pc);
}

#[no_mangle]
pub unsafe extern "C" fn metered_extern_handler(
    vm_exec_state_ptr: *mut c_void,
    pre_compute_insns_ptr: *const c_void,
    cur_pc: u32,
) -> u32 {
    let vm_exec_state_ref =
        unsafe { &mut *(vm_exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };
    vm_exec_state_ref.set_pc(cur_pc);

    // pointer to the first element of `pre_compute_insns`
    let pre_compute_insns_base_ptr =
        pre_compute_insns_ptr as *const PreComputeInstruction<'static, F, Ctx>;
    let pc_idx = (cur_pc / DEFAULT_PC_STEP) as usize;
    let pre_compute_len = PRE_COMPUTE_INSNS_LEN.load(Ordering::Relaxed);
    let pre_compute_insns_slice =
        unsafe { std::slice::from_raw_parts(pre_compute_insns_base_ptr, pre_compute_len) };

    let Some(pre_compute_insns) = pre_compute_insns_slice.get(pc_idx) else {
        vm_exec_state_ref.exit_code = Err(ExecutionError::PcOutOfBounds(cur_pc));
        return 1;
    };

    unsafe {
        (pre_compute_insns.handler)(pre_compute_insns.pre_compute, vm_exec_state_ref);
    };

    let pc = vm_exec_state_ref.vm_state.pc();
    if vm_exec_state_ref
        .exit_code
        .as_ref()
        .is_ok_and(|exit_code| exit_code.is_none())
    {
        // execution continues
        pc
    } else {
        // special indicator that we must terminate
        // this won't collide with actual pc value because pc values are always multiple of 4
        1
    }
}

#[no_mangle]
pub unsafe extern "C" fn should_suspend(exec_state_ptr: *mut c_void) -> u32 {
    // TODO: this is inconsistent with the Rust implementation. Fix it later.
    let exec_state_ref = 
    unsafe { &mut *(exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };

    if exec_state_ref.ctx.check_and_segment() && *exec_state_ref.ctx.suspend_on_segment() {
        1
    } else {
        0
    }
}
