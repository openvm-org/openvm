use std::ffi::c_void;

use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{AotInstance, AsmRunFn};
use crate::{
    arch::{
        aot::{asm_to_lib, extern_handler, set_pc_shim, should_suspend_shim},
        execution_mode::{MeteredCtx, Segment},
        interpreter::{
            alloc_pre_compute_buf, get_metered_pre_compute_instructions,
            get_metered_pre_compute_max_size, split_pre_compute_buf,
        },
        ExecutionError, ExecutorInventory, MeteredExecutionCtxTrait, MeteredExecutor,
        StaticProgramError, Streams, VmExecState, VmState,
    },
    system::memory::online::GuestMemory,
};

impl<F, Ctx> AotInstance<F, Ctx>
where
    F: PrimeField32,
    Ctx: MeteredExecutionCtxTrait,
{
    /// Creates a new instance for metered execution.
    pub fn new_metered<E>(
        inventory: &ExecutorInventory<E>,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<Self, StaticProgramError>
    where
        E: MeteredExecutor<F>,
    {
        let start = std::time::Instant::now();
        let lib = asm_to_lib(&Self::generate_metered_asm())?;

        let program = &exe.program;
        let pre_compute_max_size = get_metered_pre_compute_max_size(program, inventory);
        let mut pre_compute_buf = alloc_pre_compute_buf(program, pre_compute_max_size);
        let mut split_pre_compute_buf =
            split_pre_compute_buf(program, &mut pre_compute_buf, pre_compute_max_size);

        let pre_compute_insns = get_metered_pre_compute_instructions::<F, Ctx, E>(
            program,
            inventory,
            executor_idx_to_air_idx,
            &mut split_pre_compute_buf,
        )?;

        let init_memory = exe.init_memory.clone();

        tracing::trace!(
            "Time taken to initialize AotInstance metered execution: {}ms",
            start.elapsed().as_millis()
        );

        Ok(Self {
            system_config: inventory.config().clone(),
            pre_compute_buf,
            pre_compute_insns,
            pc_start: exe.pc_start,
            init_memory,
            lib,
        })
    }

    fn generate_metered_asm() -> String {
        // Assumption: these functions are created at compile time so their pointers don't change
        // over time.
        let should_suspend_ptr = format!("{:p}", should_suspend_shim::<F, Ctx> as *const ());
        let metered_extern_handler_ptr =
            format!("{:p}", extern_handler::<F, Ctx, false> as *const ());
        let set_pc_ptr = format!("{:p}", set_pc_shim::<F, Ctx> as *const ());
        ASM_TEMPLATE
            .replace("{should_suspend_ptr}", &should_suspend_ptr)
            .replace("{metered_extern_handler_ptr}", &metered_extern_handler_ptr)
            .replace("{set_pc_ptr}", &set_pc_ptr)
    }
}

impl<F> AotInstance<F, MeteredCtx>
where
    F: PrimeField32,
{
    /// Metered exeecution for the given `inputs`. Execution begins from the initial
    /// state specified by the `VmExe`. This function executes the program until termination.
    ///
    /// Returns the segmentation boundary data and the final VM state when execution stops.
    ///
    /// Assumes the program doesn't jump to out of bounds pc
    pub fn execute_metered(
        &self,
        inputs: impl Into<Streams<F>>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<F, GuestMemory>), ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_metered_from_state(vm_state, ctx)
    }

    /// Metered execution for the given `VmState`. This function executes the program until
    /// termination
    ///
    /// Returns the segmentation boundary data and the final VM state when execution stops.
    ///
    /// Assume program doesn't jump to out of bounds pc
    pub fn execute_metered_from_state(
        &self,
        from_state: VmState<F, GuestMemory>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<F, GuestMemory>), ExecutionError> {
        let vm_exec_state = VmExecState::new(from_state, ctx);
        let vm_exec_state = self.execute_metered_until_suspend(vm_exec_state)?;
        // handle execution error
        match vm_exec_state.exit_code {
            Ok(_) => Ok((
                vm_exec_state.ctx.segmentation_ctx.segments,
                vm_exec_state.vm_state,
            )),
            Err(e) => Err(e),
        }
    }

    // TODO: implement execute_metered_until_suspend for AOT if needed
    pub fn execute_metered_until_suspend(
        &self,
        vm_exec_state: VmExecState<F, GuestMemory, MeteredCtx>,
    ) -> Result<VmExecState<F, GuestMemory, MeteredCtx>, ExecutionError> {
        let from_state_pc = vm_exec_state.vm_state.pc();
        let mut vm_exec_state: Box<VmExecState<F, GuestMemory, MeteredCtx>> =
            Box::new(vm_exec_state);

        unsafe {
            let asm_run: libloading::Symbol<AsmRunFn> = self
                .lib
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");

            let pre_compute_insns_ptr = self.pre_compute_insns.as_ptr();
            let vm_exec_state_ptr =
                vm_exec_state.as_mut() as *mut VmExecState<F, GuestMemory, MeteredCtx>;

            asm_run(
                vm_exec_state_ptr.cast(),
                pre_compute_insns_ptr as *const c_void,
                from_state_pc,
                0, /* TODO: this is a placeholder because in the pure case asm_run needs to take
                    * in 5 args. Fix later */
            );
        }
        Ok(*vm_exec_state)
    }
}

/*
rbx = aot_vm_exec_state_ptr
rbp = pre_compute_insns_ptr
r13 = cur_pc
r12 is currently unused

Assembly code explanation

asm_run:
    push rbp     ; push callee saved register
    push rbx
    push r12
    push r13
    push r13     ; A dummy push to ensure the stack is 16 bytes aligned
    mov rbx, rdi         ; rbx = rdi = aot_vm_exec_state_ptr
    mov rbp, rsi         ; rbp = rsi = pre_compute_insns_ptr
    mov r13, rdx         ; r13 = rdx = from_state_pc

asm_execute:
    mov rdi, rbx         ; rdi = aot_vm_exec_state_ptr
    mov {should_suspend_ptr} ; rax = should_suspend
    call rax                 ; should_suspend(aot_vm_exec_state_ptr)
    cmp rax, 1           ; if return value of should_suspend is 1
    je asm_run_end       ; jump to asm_run_end
    mov rdi, rbx         ; rdi = aot_vm_exec_state_ptr
    mov rsi, rbp         ; rsi = pre_compute_insns_ptr
    mov rdx, r13         ; rdx = cur_pc
    mov rax, {metered_extern_handler_ptr}            ; rax = extern_metered_handler
    call rax             ; extern_metered_handler(aot_vm_exec_state_ptr, pre_compute_insns_ptr, cur_pc)
    cmp rax, 1           ; if return value of metered_extern_handler is 1
    je asm_run_end       ; jump to asm_run_end
    mov r13, rax         ; cur_pc = return value of metered_extern_handler
    jmp asm_execute      ; jump to asm_execute

asm_run_end:
    mov rdi, rbx          ; rdi = aot_vm_exec_state
    mov rsi, r13          ; rsi = cur_pc
    mov rax, {set_pc_ptr} ; rax = set_pc
    call rax              ;vm_exec_state.set_pc(cur_pc)
    xor rax, rax          ; set return value to 0
    pop r13
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
*/

const ASM_TEMPLATE: &str = r#".intel_syntax noprefix
.code64
.section .text
.global asm_run
asm_run:
    push rbp     
    push rbx
    push r12
    push r13  
    push r13
    mov rbx, rdi  
    mov rbp, rsi  
    mov r13, rdx 

asm_execute:
    mov rdi, rbx
    mov rax, {should_suspend_ptr}
    call rax
    cmp rax, 1          
    je asm_run_end      
    mov rdi, rbx        
    mov rsi, rbp        
    mov rdx, r13
    mov rax, {metered_extern_handler_ptr}
    call rax
    cmp rax, 1          
    je asm_run_end      
    mov r13, rax        
    jmp asm_execute     

asm_run_end:
    mov rdi, rbx
    mov rsi, r13
    mov rax, {set_pc_ptr}  
    call rax
    xor rax, rax        
    pop r13
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
"#;
