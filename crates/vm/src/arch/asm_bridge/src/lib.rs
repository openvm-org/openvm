use std::ffi::c_void;

use openvm_circuit::{
    arch::{execution_mode::ExecutionCtx, interpreter::PreComputeInstruction, VmExecState, VmState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::program::DEFAULT_PC_STEP;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use std::mem::offset_of;

extern "C" {
    fn asm_run_internal(
        vm_exec_state_ptr: *mut c_void,       // rdi = vm_exec_state
        pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
        from_state_pc: u32,                   // rdx = from_state.pc
        from_state_instret: u64,              // rcx = from_state.instret
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
    from_state_instret: u64,
) {
    println!("asm_run is called with from_state_pc {} from_state_instret {}", from_state_pc, from_state_instret);

    asm_run_internal(
        vm_exec_state_ptr,
        pre_compute_insns_ptr,
        from_state_pc,
        from_state_instret,
    );
}

type F = BabyBear;
type Ctx = ExecutionCtx;

/// At the end of the assembly execution, store the instret and pc from the x86 registers
/// to the vm state's pc and instret for the pure execution mode
#[no_mangle]
pub extern "C" fn set_instret_and_pc(
    vm_exec_state_ptr: *mut c_void,        // rdi = vm_exec_state
    _pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
    final_pc: u32,                         // rdx = final_pc
    final_instret: u64,                    // rcx = final_instret
) {
    println!("set_instret_and_pc is called with final_pc {} final_instret {} vm_exec_state_ptr {:p} _pre_compute_insns_ptr {:p}", final_pc, final_instret, vm_exec_state_ptr, _pre_compute_insns_ptr);

    // reference to vm_exec_state
    let vm_exec_state_ref =
        unsafe { &mut *(vm_exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };

    vm_exec_state_ref
        .vm_state
        .set_instret_and_pc(final_instret, final_pc);
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
    cur_instret: u64,
) -> u32 {
    if cur_instret <= 1000 {
        println!("extern_handler is called with instret {} pc {} vm_exec_state_ptr {:p} pre_compute_insns_ptr {:p}", cur_instret, cur_pc, vm_exec_state_ptr, pre_compute_insns_ptr);
    }

    // this is boxed for safety so that when `execute_e12_impl` runs when called by the handler
    // it would be able to dereference instret and pc correctly
    let mut instret: Box<u64> = Box::new(cur_instret);
    let mut pc: Box<u32> = Box::new(cur_pc);

    // reference to vm_exec_state
    let vm_exec_state_ref =
        unsafe { &mut *(vm_exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };

    // pointer to the first element of `pre_compute_insns`
    let pre_compute_insns_base_ptr =
        pre_compute_insns_ptr as *const PreComputeInstruction<'static, F, Ctx>;
    let pc_idx = (cur_pc / DEFAULT_PC_STEP) as usize;

    let pre_compute_insns = unsafe { &*pre_compute_insns_base_ptr.add(pc_idx) };

    let ctx = &vm_exec_state_ref.ctx;
    // `arg` is a runtime constant that we want to keep in register
    // - For pure execution it is `instret_end`
    let arg = ctx.instret_end;

    if cur_instret <= 1000 {
        println!("[AT CALL]: vm_exec_state_ptr valid? {:p}", vm_exec_state_ptr);
        println!("[AT CALL] Address of exit_code field: {:p}", &vm_exec_state_ref.exit_code as *const _);
        println!("[AT CALL] Address of ctx field: {:p}", &vm_exec_state_ref.ctx as *const _);
        println!("[AT CALL] Address of vm_state field: {:p}", &vm_exec_state_ref.vm_state as *const _);
    }

    unsafe {
        (pre_compute_insns.handler)(
            pre_compute_insns.pre_compute,
            &mut instret,
            &mut pc,
            arg,
            vm_exec_state_ref,
        );
    };

    let vm_exec_state_ref =
        unsafe { &*(vm_exec_state_ptr as *const VmExecState<F, GuestMemory, Ctx>) };
    
    match vm_exec_state_ref.exit_code {
        Ok(None) => {
            // execution continues
            *pc
        }
        _ => {
            // special indicator that we must terminate
            // this won't collide with actual pc value because pc values are always multiple of 4
            1
        }
    }
}

#[no_mangle]
pub extern "C" fn should_suspend(instret: u64, _pc: u32, exec_state_ptr: *mut c_void) -> u32 {
    // println!("should_suspend is called with instret {} pc {} exec_state_ptr {:p}", instret, _pc, exec_state_ptr);

    // reference to vm_exec_state
    let vm_exec_state_ref =
        unsafe { &mut *(exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };

    let instret_end = vm_exec_state_ref.ctx.instret_end;

    // println!("instret {}", instret);
    // println!("instret_end {}", instret_end);
    // println!("pc at vm_state {}", vm_exec_state_ref.vm_state.pc());

    if instret >= instret_end {
        1 // should suspend is `true`
    } else {
        0 // should suspend is `false`
    }
}
