use std::ffi::c_void;

use openvm_circuit::{
    arch::{execution_mode::ExecutionCtx, interpreter::PreComputeInstruction, VmExecState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::program::DEFAULT_PC_STEP;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

extern "C" {
    fn asm_run_internal(
        vm_exec_state_ptr: *mut c_void,       // rdi = vm_exec_state
        pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
        from_state_pc: u32,                   // rdx = from_state.pc
        instret_left: u64,
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
    instret_left: u64,
) {
    asm_run_internal(
        vm_exec_state_ptr,
        pre_compute_insns_ptr,
        from_state_pc,
        instret_left,
    );
}

type F = BabyBear;
type Ctx = ExecutionCtx;

/// At the end of the assembly execution, store the instret and pc from the x86 registers
/// to the vm state's pc and instret for the pure execution mode
#[no_mangle]
pub extern "C" fn set_pc(
    vm_exec_state_ptr: *mut c_void,        // rdi = vm_exec_state
    _pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
    final_pc: u32,                         // rdx = final_pc
) {
    // reference to vm_exec_state
    let vm_exec_state_ref =
        unsafe { &mut *(vm_exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };
    vm_exec_state_ref.vm_state.set_pc(final_pc);
}

/// extern handler for the pure execution mode
/// calls the correct function handler based on `cur_pc`
///
/// returns 1 if we should terminate and *pc otherwise
/// this is safe because *pc is always a multiple of 4
#[no_mangle]
pub extern "C" fn extern_handler(
    vm_exec_state_ptr: *mut c_void,
    pre_compute_insns_ptr: *const c_void,
    cur_pc: u32,
) -> u32 {
    // reference to vm_exec_state
    let vm_exec_state_ref =
        unsafe { &mut *(vm_exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };
    vm_exec_state_ref.set_pc(cur_pc);

    // pointer to the first element of `pre_compute_insns`
    let pre_compute_insns_base_ptr =
        pre_compute_insns_ptr as *const PreComputeInstruction<'static, F, Ctx>;
    let pc_idx = (cur_pc / DEFAULT_PC_STEP) as usize;

    let pre_compute_insns = unsafe { &*pre_compute_insns_base_ptr.add(pc_idx) };

    unsafe {
        (pre_compute_insns.handler)(pre_compute_insns.pre_compute, vm_exec_state_ref);
    };

    let pc = vm_exec_state_ref.vm_state.pc();
    match vm_exec_state_ref.exit_code {
        Ok(None) => pc,
        _ => {
            // special indicator that we must terminate
            // this won't collide with actual pc value because pc values are always multiple of 4
            pc + 1
        }
    }
}

#[no_mangle]
pub extern "C" fn should_suspend(_instret: u64, _pc: u32, exec_state_ptr: *mut c_void) -> u32 {
    // reference to vm_exec_state
    let vm_exec_state_ref =
        unsafe { &mut *(exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };

    let instret_left = vm_exec_state_ref.ctx.instret_left;

    if instret_left == 0 {
        1 // should suspend is `true`
    } else {
        vm_exec_state_ref.ctx.instret_left -= 1;
        0 // should suspend is `false`
    }
}

#[no_mangle]
pub extern "C" fn get_vm_register_addr(exec_state_ptr: *mut c_void) -> *mut u64 {
    let vm_exec_state_ref =
        unsafe { &mut *(exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };
    let ptr = &vm_exec_state_ref.vm_state.memory.memory.mem[1];
    ptr.as_ptr() as *mut u64 // mut u64 because we want to write 8 bytes at a time
}

#[no_mangle]
pub extern "C" fn get_vm_address_space_addr(
    exec_state_ptr: *mut c_void,
    addr_space: u64,
) -> *mut u64 {
    let vm_exec_state_ref =
        unsafe { &mut *(exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };
    let ptr = &vm_exec_state_ref.vm_state.memory.memory.mem[addr_space as usize];
    ptr.as_ptr() as *mut u64 // mut u64 because we want to write 8 bytes at a time
}

#[allow(dead_code)]
#[no_mangle]
pub extern "C" fn debug_vm_register_addr(mmap_ptr: *mut u32) {
    let _first_val = unsafe { *mmap_ptr };
    let _second_val = unsafe { *(mmap_ptr.wrapping_add(1)) };
}
