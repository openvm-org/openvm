use std::{ffi::c_void, mem::offset_of};

use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{
    common::{
        sync_gpr_to_xmm, sync_xmm_to_gpr, REG_D, REG_EXEC_STATE_PTR, REG_FIRST_ARG, REG_FOURTH_ARG,
        REG_INSNS_PTR, REG_INSTRET_END, REG_PC, REG_RETURN_VAL, REG_SECOND_ARG, REG_THIRD_ARG,
    },
    AotInstance, AsmRunFn,
};
use crate::{
    arch::{
        aot::{
            asm_to_lib, extern_handler, get_vm_address_space_addr, set_pc_shim, should_suspend_shim,
        },
        execution_mode::{
            metered::segment_ctx::SegmentationCtx, ExecutionCtx, MeteredCtx, Segment,
        },
        interpreter::{
            alloc_pre_compute_buf, get_metered_pre_compute_instructions,
            get_metered_pre_compute_max_size, split_pre_compute_buf,
        },
        AotError, ExecutionError, ExecutorInventory, MeteredExecutor, StaticProgramError, Streams,
        VmExecState, VmState,
    },
    system::memory::online::GuestMemory,
};
static_assertions::assert_impl_all!(AotInstance<p3_baby_bear::BabyBear, MeteredCtx>: Send, Sync);

impl<F> AotInstance<F, MeteredCtx>
where
    F: PrimeField32,
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
        let asm_source = Self::create_metered_asm(exe, inventory)?;
        let lib = asm_to_lib(&asm_source)?;

        let program = &exe.program;
        let pre_compute_max_size = get_metered_pre_compute_max_size(program, inventory);
        let mut pre_compute_buf = alloc_pre_compute_buf(program, pre_compute_max_size);
        let mut split_pre_compute_buf =
            split_pre_compute_buf(program, &mut pre_compute_buf, pre_compute_max_size);
        let pre_compute_insns = get_metered_pre_compute_instructions::<F, MeteredCtx, E>(
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
    pub fn create_metered_asm<E>(
        exe: &VmExe<F>,
        inventory: &ExecutorInventory<E>,
    ) -> Result<String, StaticProgramError>
    where
        E: MeteredExecutor<F>,
    {
        let mut asm_str = String::new();
        let instret_until_end_offset = offset_of!(VmExecState<F, GuestMemory, MeteredCtx>, ctx)
            + offset_of!(MeteredCtx, segmentation_ctx)
            + offset_of!(SegmentationCtx, instrets_until_check);
        let sync_reg_to_instret_until_end = || {
            format!(
                "    mov QWORD PTR [{REG_EXEC_STATE_PTR} + {instret_until_end_offset}], {REG_INSTRET_END}\n"
            )
        };
        let sync_instret_until_end_to_reg = || {
            format!(
                "    mov {REG_INSTRET_END}, [{REG_EXEC_STATE_PTR} + {instret_until_end_offset}]\n"
            )
        };
        // generate the assembly based on exe.program

        // header part
        asm_str += ".intel_syntax noprefix\n";
        asm_str += ".code64\n";
        asm_str += ".section .text\n";
        asm_str += ".global asm_run\n";

        // asm_run_internal part
        asm_str += "asm_run:\n";

        asm_str += &Self::push_external_registers();

        asm_str += &format!("   mov {REG_EXEC_STATE_PTR}, {REG_FIRST_ARG}\n");
        asm_str += &format!("   mov {REG_INSNS_PTR}, {REG_SECOND_ARG}\n");
        asm_str += &format!("   mov {REG_PC}, {REG_THIRD_ARG}\n");
        asm_str += &format!("   mov {REG_INSTRET_END}, {REG_FOURTH_ARG}\n");

        let get_vm_address_space_addr_ptr = format!(
            "{:p}",
            get_vm_address_space_addr::<F, MeteredCtx> as *const ()
        );

        asm_str += &Self::push_internal_registers();

        // Temporarily use r14 as the pointer to get_vm_address_space_addr
        asm_str += &format!("    mov r14, {get_vm_address_space_addr_ptr}\n");
        asm_str += "    mov rdi, rbx\n";
        asm_str += "    mov rsi, 1\n";
        asm_str += "    call r14\n";
        // Store the start of register address space in r15
        asm_str += "    mov r15, rax\n";
        // Store the start of address space 2 in high 64 bits of xmm0
        asm_str += "    mov rdi, rbx\n";
        asm_str += "    mov rsi, 2\n";
        asm_str += "    call r14\n";
        asm_str += "    pinsrq  xmm0, rax, 1\n";
        // Store the start of address space 3 in high 64 bits of xmm1
        asm_str += "    mov rdi, rbx\n";
        asm_str += "    mov rsi, 3\n";
        asm_str += "    call r14\n";
        asm_str += "    pinsrq  xmm1, rax, 1\n";
        // Store the start of address space 4 in high 64 bits of xmm2
        asm_str += "    mov rdi, rbx\n";
        asm_str += "    mov rsi, 4\n";
        asm_str += "    call r14\n";
        asm_str += "    pinsrq  xmm2, rax, 1\n";

        asm_str += &Self::pop_internal_registers();

        asm_str += &Self::rv32_regs_to_xmm();

        asm_str += &format!("   lea {REG_D}, [rip + map_pc_base]\n");
        asm_str += &format!("   movsxd {REG_PC}, [{REG_D} + {REG_PC}]\n");
        asm_str += &format!("   add {REG_PC}, {REG_D}\n");
        asm_str += &format!("   jmp {REG_PC}\n");

        // asm_execute_pc_{pc_num}
        // do fallback first for now but expand per instruction
        // other function, then check rax = 1 value, for should suspend

        let pc_base = exe.program.pc_base;

        for i in 0..(pc_base / 4) {
            asm_str += &format!("asm_execute_pc_{}:", i * 4);
            asm_str += "\n";
        }

        let extern_handler_ptr =
            format!("{:p}", extern_handler::<F, MeteredCtx, true> as *const ());
        let set_pc_ptr = format!("{:p}", set_pc_shim::<F, MeteredCtx> as *const ());
        let should_suspend_ptr = format!("{:p}", should_suspend_shim::<F, MeteredCtx> as *const ()); //needs state_ptr
                                                                                                     // need to pass in config: SystemConfig

        // before, 1 too high, now 1 too low for instret
        for (pc, instruction, _) in exe.program.enumerate_by_pc() {
            /* Preprocessing step, to check if we should suspend or not */
            asm_str += &format!("asm_execute_pc_{pc}:\n");

            asm_str += &format!("    cmp {REG_INSTRET_END}, 0\n");
            asm_str += &format!("    jne instret_positive_{pc}\n"); // if instret > 0, skip slow path

            asm_str += &Self::xmm_to_rv32_regs();
            asm_str += &Self::push_address_space_start();
            asm_str += &Self::push_internal_registers();
            asm_str += &sync_reg_to_instret_until_end();
            asm_str += &format!("    movabs {REG_D}, {should_suspend_ptr}\n");
            asm_str += &format!("    mov {REG_FIRST_ARG}, {REG_EXEC_STATE_PTR}\n");
            asm_str += &format!("    call {REG_D}\n");
            asm_str += &format!("    test al, al\n");

            asm_str += &Self::pop_internal_registers();
            asm_str += &Self::pop_address_space_start();
            asm_str += &sync_instret_until_end_to_reg();
            asm_str += &Self::rv32_regs_to_xmm();
            asm_str += &format!("    jnz asm_run_end_{pc}\n");
            asm_str += &format!("    jmp execute_instruction_{pc}\n");

            asm_str += &format!("instret_positive_{pc}:\n");
            asm_str += &format!("    dec {REG_INSTRET_END}\n");
            asm_str += &sync_reg_to_instret_until_end();

            // continue with execution, as should_suspend returned false
            asm_str += &format!("execute_instruction_{pc}:\n");

            if instruction.opcode.as_usize() == 0 {
                // terminal opcode has no associated executor, so can handle with default fallback
                asm_str += &Self::xmm_to_rv32_regs();
                asm_str += &Self::push_address_space_start();
                asm_str += &Self::push_internal_registers();
                asm_str += &sync_reg_to_instret_until_end();
                asm_str += &format!("   mov {REG_FIRST_ARG}, {REG_EXEC_STATE_PTR}\n");
                asm_str += &format!("   mov {REG_SECOND_ARG}, {REG_INSNS_PTR}\n");
                asm_str += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
                asm_str += &format!("   mov {REG_D}, {extern_handler_ptr}\n");
                asm_str += &format!("   call {REG_D}\n");
                asm_str += &format!("   mov {REG_PC}, {REG_D}\n");
                asm_str += &format!("   AND {REG_D}, 1\n");
                asm_str += &format!("   cmp {REG_D}, 1\n");

                asm_str += &Self::pop_internal_registers();
                asm_str += &Self::pop_address_space_start();
                asm_str += &format!("   mov {REG_FIRST_ARG}, {REG_EXEC_STATE_PTR}\n");
                asm_str += &format!("   mov {REG_SECOND_ARG}, {pc}\n");
                asm_str += &format!("   mov {REG_D}, {set_pc_ptr}\n");
                asm_str += &format!("   call {REG_D}\n");
                asm_str += &format!(
                    "   mov {REG_RETURN_VAL}, {}\n",
                    instruction.c.as_canonical_u32()
                );
                asm_str += &Self::pop_external_registers();
                asm_str += "    ret\n";

                continue;
            }

            let executor = inventory
                .get_executor(instruction.opcode)
                .expect("executor not found for opcode");

            if executor.is_aot_metered_supported(&instruction) {
                let segment = executor
                    .generate_x86_metered_asm(&instruction, pc, inventory.config())
                    .map_err(|err| match err {
                        AotError::InvalidInstruction => StaticProgramError::InvalidInstruction(pc),
                        AotError::NotSupported => StaticProgramError::DisabledOperation {
                            pc,
                            opcode: instruction.opcode,
                        },
                        AotError::NoExecutorFound(opcode) => {
                            StaticProgramError::ExecutorNotFound { opcode }
                        }
                        AotError::Other(_message) => StaticProgramError::InvalidInstruction(pc),
                    })?;
                asm_str += &segment;
            } else {
                asm_str += &Self::xmm_to_rv32_regs();
                asm_str += &Self::push_address_space_start();
                asm_str += &Self::push_internal_registers();
                asm_str += &sync_reg_to_instret_until_end();
                asm_str += "    mov rdi, rbx\n";
                asm_str += "    mov rsi, rbp\n";
                asm_str += &format!("    mov rdx, {pc}\n");
                asm_str += &format!("    mov rax, {extern_handler_ptr}\n");
                asm_str += "    call rax\n";

                asm_str += "    mov r13, rax\n"; // move the return value of the extern_handler into r13
                asm_str += "    AND rax, 1\n"; // check if the return value is 1
                asm_str += "    cmp rax, 1\n"; // compare the return value with 1
                asm_str += &Self::pop_internal_registers(); // pop the internal registers from the stack
                asm_str += &Self::pop_address_space_start();
                asm_str += &Self::rv32_regs_to_xmm(); // read the memory from the memory location of the RV32 registers in `GuestMemory`
                                                      // registers, to the appropriate XMM registers
                asm_str += &format!("   je asm_run_end_{pc}\n");

                asm_str += "    lea rdx, [rip + map_pc_base]\n"; // load the base address of the map_pc_base section
                asm_str += "    movsxd rcx, [rdx + r13]\n"; // load the offset of the next instruction (r13 is the next pc)
                asm_str += "    add rcx, rdx\n"; // add the base address and the offset
                asm_str += "    jmp rcx\n"; // jump to the next instruction (rcx is the next instruction)
                asm_str += "\n";
            }
        }

        // asm_run_end part
        for (pc, _instruction, _) in exe.program.enumerate_by_pc() {
            asm_str += &format!("asm_run_end_{pc}:\n");
            asm_str += &Self::xmm_to_rv32_regs();
            asm_str += &format!("    mov {REG_FIRST_ARG}, rbx\n");
            asm_str += &format!("    mov {REG_SECOND_ARG}, {pc}\n");
            asm_str += &format!("    mov {REG_D}, {set_pc_ptr}\n");
            asm_str += &format!("    call {REG_D}\n");
            asm_str += &Self::pop_external_registers();
            asm_str += &format!("    xor {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
            asm_str += "    ret\n";
        }

        // map_pc_base part
        asm_str += ".section .rodata\n";
        asm_str += "map_pc_base:\n";

        for i in 0..(pc_base / 4) {
            asm_str += &format!("   .long asm_execute_pc_{} - map_pc_base\n", i * 4);
        }

        for (pc, _instruction, _) in exe.program.enumerate_by_pc() {
            asm_str += &format!("   .long asm_execute_pc_{pc} - map_pc_base\n");
        }

        Ok(asm_str)
    }
    fn generate_metered_asm() -> String {
        // Assumption: these functions are created at compile time so their pointers don't change
        // over time.
        let should_suspend_ptr = format!("{:p}", should_suspend_shim::<F, MeteredCtx> as *const ());
        let metered_extern_handler_ptr =
            format!("{:p}", extern_handler::<F, MeteredCtx, false> as *const ());
        let set_pc_ptr = format!("{:p}", set_pc_shim::<F, MeteredCtx> as *const ());
        ASM_TEMPLATE
            .replace("{should_suspend_ptr}", &should_suspend_ptr)
            .replace("{metered_extern_handler_ptr}", &metered_extern_handler_ptr)
            .replace("{set_pc_ptr}", &set_pc_ptr)
    }
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

            let instret_until_end = vm_exec_state.ctx.segmentation_ctx.instrets_until_check;
            asm_run(
                vm_exec_state_ptr.cast(),
                pre_compute_insns_ptr as *const c_void,
                from_state_pc,
                instret_until_end,
            );
        }
        Ok(*vm_exec_state)
    }
}

/*
rbx = aot_vm_exec_state_ptr
rbp = pre_compute_insns_ptr
r13 = cur_pc
r12 = instret_until_end counter

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
    cmp al, 1            ; if return value of should_suspend is 1. CAREFUL: when a function returns a boolean, only the lower 8 bits are set.
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
    cmp al, 1          
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
