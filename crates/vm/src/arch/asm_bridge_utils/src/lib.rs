use openvm_circuit::arch::interpreter::PreComputeInstruction;
use openvm_circuit::arch::{ExecutionCtxTrait, VmExecState};
use openvm_circuit::system::memory::online::GuestMemory;
use openvm_instructions::program::DEFAULT_PC_STEP;
use openvm_stark_sdk::p3_baby_bear::BabyBear as F;
use std::ffi::c_void;

pub fn set_instret_and_pc_generic<Ctx: ExecutionCtxTrait>(
    vm_exec_state_ptr: *mut c_void,
    final_pc: u32,
    final_instret: u64,
) {
    let vm_exec_state_ref =
        unsafe { &mut *(vm_exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };
    vm_exec_state_ref
        .vm_state
        .set_instret_and_pc(final_instret, final_pc);
}

/// # Safety
/// - vm_exec_state_ptr must point to a valid VmExecState<F, GuestMemory, Ctx>.
/// - pre_compute_insns_ptr must be a valid, contiguous array of
///   PreComputeInstruction<'static, F, Ctx>.
/// - cur_pc must be in-bounds (pc/DEFAULT_PC_STEP is a valid index).
pub unsafe fn extern_prep_generic<Ctx: ExecutionCtxTrait>(
    vm_exec_state_ptr: *mut c_void,
    pre_compute_insns_ptr: *const c_void,
    cur_pc: u32,
) -> (
    *mut VmExecState<F, GuestMemory, Ctx>,
    *const PreComputeInstruction<'static, F, Ctx>,
    *const Ctx,
) {
    let vm_ptr = vm_exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>;
    let base = pre_compute_insns_ptr as *const PreComputeInstruction<'static, F, Ctx>;
    let pc_idx = (cur_pc / DEFAULT_PC_STEP) as usize;
    let pre_ptr = base.add(pc_idx);
    let ctx_ptr = &(*vm_ptr).ctx as *const Ctx;
    (vm_ptr, pre_ptr, ctx_ptr)
}

/// # Safety
/// - vm_exec_state_ptr and pre_compute_insn_ptr must come from extern_prep_generic
///   for the same Ctx and program.
/// - cur_pc/cur_instret must correspond to the provided state.
/// - arg must be the correct runtime parameter for Ctx (e.g., max_execution_cost).
pub unsafe fn extern_finish_generic<Ctx: ExecutionCtxTrait>(
    vm_exec_state_ptr: *mut VmExecState<F, GuestMemory, Ctx>,
    pre_compute_insn_ptr: *const PreComputeInstruction<'static, F, Ctx>,
    cur_pc: u32,
    cur_instret: u64,
    arg: u64,
) -> u32 {
    let vm = &mut *vm_exec_state_ptr;
    let pre = &*pre_compute_insn_ptr;

    let mut instret: Box<u64> = Box::new(cur_instret);
    let mut pc: Box<u32> = Box::new(cur_pc);

    (pre.handler)(pre.pre_compute, &mut instret, &mut pc, arg, vm);

    match vm.exit_code {
        Ok(None) => *pc,
        _ => 1,
    }
}
