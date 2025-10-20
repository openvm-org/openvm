use std::ffi::c_void;

use openvm_circuit::{
    arch::{execution_mode::MeteredCostCtx, VmExecState},
    system::memory::online::GuestMemory,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

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

/// Runs the VM execution from assembly
///
/// # Safety
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
    asm_run_internal(
        vm_exec_state_ptr,
        pre_compute_insns_ptr,
        from_state_pc,
        from_state_instret,
    );
}

type F = BabyBear;
type Ctx = MeteredCostCtx;

// at the end of the execution, you want to store the instret and pc from the x86 registers
// to update the vm state's pc and instret
// works for metered cost execution
#[no_mangle]
pub extern "C" fn metered_cost_set_instret_and_pc(
    vm_exec_state_ptr: *mut c_void,        // rdi = vm_exec_state
    _pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
    final_pc: u32,                         // rdx = final_pc
    final_instret: u64,                    // rcx = final_instret
) {
    // reference to vm_exec_state
    asm_bridge_utils::set_instret_and_pc_generic::<Ctx>(vm_exec_state_ptr, final_pc, final_instret);
}

/// # Safety
/// - vm_exec_state_ptr must point to VmExecState<F, GuestMemory, MeteredCostCtx>.
/// - pre_compute_insns_ptr must be a valid, contiguous array of
///   PreComputeInstruction<'static, F, MeteredCostCtx>.
/// - cur_pc must be a valid PC for the current program.
#[no_mangle]
pub unsafe extern "C" fn metered_cost_extern_handler(
    vm_exec_state_ptr: *mut c_void,
    pre_compute_insns_ptr: *const c_void,
    cur_pc: u32,
    cur_instret: u64,
) -> u32 {
    unsafe {
        let (vm_ptr, pre_ptr, ctx_ptr) = asm_bridge_utils::extern_prep_generic::<Ctx>(
            vm_exec_state_ptr,
            pre_compute_insns_ptr,
            cur_pc,
        );
        let arg = (*ctx_ptr).max_execution_cost;
        asm_bridge_utils::extern_finish_generic::<Ctx>(vm_ptr, pre_ptr, cur_pc, cur_instret, arg)
    }
}

#[no_mangle]
pub extern "C" fn should_suspend(_instret: u64, _pc: u32, exec_state_ptr: *mut c_void) -> u32 {
    let exec_state_ref = unsafe { &mut *(exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };

    let max_execution_cost = exec_state_ref.ctx.max_execution_cost;

    if exec_state_ref.ctx.cost > max_execution_cost {
        1
    } else {
        0
    }
}
