#[cfg(feature = "aot")]
pub use aot::*;
#[cfg(feature = "aot")]
mod aot {
    use std::ffi::c_void;

    use openvm_circuit::{
        arch::{
            aot::AotMeteredVmExecState, execution_mode::MeteredCtx,
            interpreter::PreComputeInstruction, VmExecState,
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
            aot_vm_exec_state_ptr: *mut c_void,   // rdi = aot_vm_exec_state
            pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
            from_state_pc: u32,                   // rdx = from_state.pc
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
        aot_vm_exec_state_ptr: *mut c_void,
        pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
        from_state_pc: u32,
    ) {
        asm_run_internal(aot_vm_exec_state_ptr, pre_compute_insns_ptr, from_state_pc);
    }

    type F = BabyBear;
    type Ctx = MeteredCtx;

    // at the end of the execution, you want to store the instret and pc from the x86 registers
    // to update the vm state's pc and instret
    // works for metered execution
    #[no_mangle]
    pub extern "C" fn metered_set_pc(
        aot_vm_exec_state_ptr: *mut c_void,    // rdi = vm_exec_state
        _pre_compute_insns_ptr: *const c_void, // rsi = pre_compute_insns
        final_pc: u32,                         // rdx = final_pc
    ) {
        // reference to vm_exec_state
        let aot_vm_exec_state_ref =
            unsafe { &mut *(aot_vm_exec_state_ptr as *mut AotMeteredVmExecState) };
        unsafe { (aot_vm_exec_state_ref.set_pc)(aot_vm_exec_state_ref.vm_exec_state, final_pc) };
    }

    #[no_mangle]
    pub extern "C" fn metered_extern_handler(
        aot_vm_exec_state_ptr: *mut c_void,
        pre_compute_insns_ptr: *const c_void,
        cur_pc: u32,
    ) -> u32 {
        let aot_vm_exec_state_ref =
            unsafe { &mut *(aot_vm_exec_state_ptr as *mut AotMeteredVmExecState) };
        let vm_exec_state_ptr = aot_vm_exec_state_ref.vm_exec_state;
        unsafe { (aot_vm_exec_state_ref.set_pc)(vm_exec_state_ptr, cur_pc) };

        // pointer to the first element of `pre_compute_insns`
        let pre_compute_insns_base_ptr =
            pre_compute_insns_ptr as *const PreComputeInstruction<'static, F, Ctx>;
        let pc_idx = (cur_pc / DEFAULT_PC_STEP) as usize;

        let pre_compute_insns = unsafe { &*pre_compute_insns_base_ptr.add(pc_idx) };

        let vm_exec_state_ref =
            unsafe { &mut *(vm_exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };

        unsafe {
            (pre_compute_insns.handler)(pre_compute_insns.pre_compute, vm_exec_state_ref);
        };
        if unsafe { (aot_vm_exec_state_ref.is_exit_code_ok_none)(vm_exec_state_ptr) } {
            // Unsafe but `vm_state` is the first field of `VmExecState` and `pc` is the first field
            // of `VmState`. Since all structs are `repr(C)`, `pc` should alawys in the
            // same offset across different compilation.
            vm_exec_state_ref.pc()
        } else {
            1
        }
    }

    #[no_mangle]
    pub extern "C" fn should_suspend(aot_vm_exec_state_ptr: *mut c_void) -> u32 {
        let aot_vm_exec_state_ref =
            unsafe { &mut *(aot_vm_exec_state_ptr as *mut AotMeteredVmExecState) };
        if unsafe { (aot_vm_exec_state_ref.should_suspend)(aot_vm_exec_state_ref.vm_exec_state) } {
            1
        } else {
            0
        }
    }
}
