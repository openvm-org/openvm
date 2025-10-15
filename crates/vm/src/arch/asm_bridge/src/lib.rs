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
use openvm_instructions::program::DEFAULT_PC_STEP;

// asm_run.s contains the assembly to run pure execution
global_asm!(include_str!("asm_run.s"));

extern "C" {
    fn asm_run_internal(
        vm_exec_state_ptr: *mut c_void, // rdi = vm_exec_state
        pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
        from_state_pc: u32, // rdx = from_state.pc
        from_state_instret: u64, // rcx = from_state.instret
        pc_base: u32, // r8 = pc_base
    );
}

#[no_mangle]
pub unsafe extern "C" fn asm_run(
    vm_exec_state_ptr: *mut c_void,
    pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
    from_state_pc: u32,
    from_state_instret: u64,
    pc_base: u32, // r8 = pc_base
) {
    asm_run_internal(
        vm_exec_state_ptr, 
        pre_compute_insns_ptr, 
        from_state_pc, 
        from_state_instret,
        pc_base
    );
}

type F = BabyBear;

// at the end of the execution, store the instret and pc from the x86 registers
// to update the vm state's pc and instret for the pure execution mode
#[no_mangle]
pub extern "C" fn set_instret_and_pc(
    vm_exec_state_ptr: *mut c_void, // rdi = vm_exec_state
    pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
    final_pc: u32, // rdx = final_pc 
    final_instret: u64, // rcx = final_instret
    pc_base: u32, // r8 = pc_base
) {
    type Ctx = ExecutionCtx;
    // reference to vm_exec_state
    let vm_exec_state_ref = unsafe { 
        &mut *(vm_exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>)
    };
    vm_exec_state_ref.vm_state.set_instret_and_pc(final_instret, final_pc);
}

// extern handler for the pure execution mode
// calls the correct function handler based on `cur_pc`
#[no_mangle]
pub extern "C" fn extern_handler(
    vm_exec_state_ptr: *mut c_void,
    pre_compute_insns_ptr: *const c_void,
    cur_pc: u32, 
    cur_instret: u64,
    pc_base: u32,
) -> u32 {

    println!("handler called with args cur_pc {} cur_instret {} pc_base {}", cur_pc, cur_instret, pc_base);

    type F = BabyBear;
    type Ctx = ExecutionCtx;

    // this is boxed for safety so that when `execute_e12_impl` runs when called by the handler
    // it would be able to dereference instret and pc correctly
    let mut instret: Box<u64> = Box::new(cur_instret);
    let mut pc: Box<u32> = Box::new(cur_pc);

    // reference to vm_exec_state
    let vm_exec_state_ref = unsafe {
        &mut *(vm_exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>)
    };

    // pointer to the first element of `pre_compute_insns`
    let pre_compute_insns_base_ptr = pre_compute_insns_ptr as *const PreComputeInstruction<'static, BabyBear, Ctx>;
    let pc_idx = (cur_pc / DEFAULT_PC_STEP) as usize;

    let pre_compute_insns = unsafe {
        &*pre_compute_insns_base_ptr.add(pc_idx)
    };

    let ctx = &vm_exec_state_ref.ctx;
    // `arg` is a runtime constant that we want to keep in register
    // - For pure execution it is `instret_end`
    let arg = ctx.instret_end; 

    unsafe {
        (pre_compute_insns.handler)(
            pre_compute_insns.pre_compute, 
            &mut instret, 
            &mut pc, 
            arg, 
            vm_exec_state_ref
        );
    };

    println!("new pc after executing instruction {}", *pc);

    match vm_exec_state_ref.exit_code {
        Ok(None) => { // execution continues 
            return *pc;
        }
        _ => { // special indicator that we must terminate
            // this won't collide with actual pc value because pc values are always multiple of 4
            return 1;
        }
    }
}