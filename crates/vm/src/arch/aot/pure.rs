use std::ffi::c_void;

use openvm_instructions::{exe::VmExe, program::DEFAULT_PC_STEP};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{common::*, AotInstance, AsmRunFn};
#[cfg(feature = "metrics")]
use crate::arch::execution_metrics::{ExecutionMetric, ExecutionMetricTimer};
use crate::{
    arch::{
        aot::{
            asm_to_lib, extern_handler, get_vm_address_space_addr, get_vm_pc_ptr,
            set_instret_left_shim, set_pc_shim,
        },
        execution_mode::ExecutionCtx,
        interpreter::{
            alloc_pre_compute_buf, get_pre_compute_instructions, get_pre_compute_max_size,
            split_pre_compute_buf, PreComputeInstruction,
        },
        AotError, ExecutionError, Executor, ExecutorInventory, ExitCode, StaticProgramError,
        Streams, VmExecState, VmState,
    },
    system::memory::online::GuestMemory,
};

static_assertions::assert_impl_all!(
    AotInstance<'static, ExecutionCtx>: Send,
    Sync
);

impl<'a> AotInstance<'a, ExecutionCtx> {
    fn create_pure_asm<F, E>(
        exe: &VmExe<F>,
        inventory: &ExecutorInventory<E>,
        pre_compute_insns_ptr: *const PreComputeInstruction<ExecutionCtx>,
    ) -> Result<String, StaticProgramError>
    where
        F: PrimeField32,
        E: Executor<F>,
    {
        let mut asm_str = String::new();
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
        asm_str += &format!("   mov {REG_B}, {REG_THIRD_ARG}\n");
        asm_str += &format!("   mov {REG_INSTRET_END}, {REG_FOURTH_ARG}\n");

        let get_vm_address_space_addr_ptr = format!(
            "{:p}",
            get_vm_address_space_addr::<ExecutionCtx> as *const ()
        );

        let get_vm_pc_ptr = format!("{:p}", get_vm_pc_ptr::<ExecutionCtx> as *const ());

        asm_str += &Self::push_internal_registers();

        // Temporarily use r14 as the pointer to get_vm_address_space_addr
        asm_str += &format!("    mov r14, {get_vm_address_space_addr_ptr}\n");
        asm_str += "    mov rdi, rbx\n";
        asm_str += "    mov rsi, 2\n";
        asm_str += "    call r14\n";
        // Store the start of address space 2 in r15
        asm_str += "    mov r15, rax\n";
        // Store the start of register address space in high 64 bits of xmm0
        asm_str += "    mov rdi, rbx\n";
        asm_str += "    mov rsi, 1\n";
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
        // Store the pointer to where `pc` is stored in the vmstate in high 64 bits of xmm3
        asm_str += "    mov rdi, rbx\n";
        asm_str += &format!("   mov {REG_D}, {get_vm_pc_ptr}\n");
        asm_str += &format!("   call {REG_D}\n");
        asm_str += "    pinsrq  xmm3, rax, 1\n"; // write `eax` to the third lane of xmm3

        asm_str += &Self::pop_internal_registers();

        asm_str += &Self::rv32_regs_to_xmm();

        asm_str += &format!("   lea {REG_C}, [rip + map_pc_base]\n");
        asm_str += &format!("   pextrq {REG_A}, xmm3, 1\n"); // extract the upper 64 bits of the xmm3 register to REG_A
        asm_str += &format!("   mov {REG_A_W}, dword ptr [{REG_A}]\n");
        asm_str += &format!("   movsxd {REG_A}, [{REG_C} + {REG_A}]\n");
        asm_str += &format!("   add {REG_A}, {REG_C}\n");
        asm_str += &format!("   jmp {REG_A}\n");

        // asm_execute_pc_{pc_num}
        // do fallback first for now but expand per instruction

        let pc_base = exe.program.pc_base;
        let base_idx = (pc_base / DEFAULT_PC_STEP) as usize;
        let num_pc_slots = base_idx + exe.program.instructions_and_debug_infos.len();

        let extern_handler_ptr = format!("{:p}", extern_handler::<ExecutionCtx, true> as *const ());
        let set_pc_ptr = format!("{:p}", set_pc_shim::<ExecutionCtx> as *const ());
        let pre_compute_insns_ptr = format!("{:p}", pre_compute_insns_ptr as *const ());
        let instret_left_ptr = format!("{:p}", set_instret_left_shim as *const ());

        for pc_idx in 0..num_pc_slots {
            /* Preprocessing step, to check if we should suspend or not */
            let pc = pc_idx as u32 * DEFAULT_PC_STEP;
            let instruction = pc_idx.checked_sub(base_idx).and_then(|idx| {
                exe.program.instructions_and_debug_infos[idx]
                    .as_ref()
                    .map(|(instruction, _)| instruction)
            });

            let Some(instruction) = instruction else {
                asm_str += &format!("asm_execute_pc_{pc}:\n");
                asm_str += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
                asm_str += "   jmp asm_dead_pc_handler\n";
                continue;
            };

            asm_str += &format!("asm_execute_pc_{pc}:\n");

            // Check if we should suspend or not

            asm_str += &format!("    cmp {REG_INSTRET_END}, 0\n");
            asm_str += &format!("    je asm_run_end_{pc}\n");
            asm_str += &format!("    dec {REG_INSTRET_END}\n");

            if instruction.opcode.as_usize() == 0 {
                // terminal opcode has no associated executor, so can handle with default
                // fallback
                asm_str += &Self::xmm_to_rv32_regs();
                asm_str += &Self::push_address_space_start();
                asm_str += &Self::push_internal_registers();

                asm_str += &format!("   mov {REG_FIRST_ARG}, {REG_EXEC_STATE_PTR}\n");
                asm_str += &format!("   mov {REG_SECOND_ARG}, {REG_INSTRET_END}\n");
                asm_str += &format!("   mov {REG_D}, {instret_left_ptr}\n");
                asm_str += &format!("   call {REG_D}\n");

                asm_str += &format!("   mov {REG_FIRST_ARG}, {REG_EXEC_STATE_PTR}\n");
                asm_str += &format!("   mov {REG_SECOND_ARG}, {pre_compute_insns_ptr}\n");
                asm_str += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
                asm_str += &format!("   mov {REG_D}, {extern_handler_ptr}\n");
                asm_str += &format!("   call {REG_D}\n");
                asm_str += &format!("   cmp {REG_RETURN_VAL}, 1\n");

                asm_str += &Self::pop_internal_registers();
                asm_str += &Self::pop_address_space_start();
                asm_str += &format!("   mov {REG_FIRST_ARG}, {REG_EXEC_STATE_PTR}\n");
                asm_str += &format!("   mov {REG_SECOND_ARG}, {pc}\n");
                asm_str += &format!("   mov {REG_D}, {set_pc_ptr}\n");
                asm_str += &format!("   call {REG_D}\n");
                asm_str += &format!("   xor {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
                asm_str += &Self::pop_external_registers();
                asm_str += "    ret\n";

                continue;
            }

            let executor = inventory
                .get_executor(instruction.opcode)
                .expect("executor not found for opcode");

            if executor.is_aot_supported(instruction) {
                let segment =
                    executor
                        .generate_x86_asm(instruction, pc)
                        .map_err(|err| match err {
                            AotError::InvalidInstruction => {
                                StaticProgramError::InvalidInstruction(pc)
                            }
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
                continue;
            }

            {
                asm_str += &Self::xmm_to_rv32_regs();
                asm_str += &Self::push_address_space_start();
                asm_str += &Self::push_internal_registers();

                asm_str += &format!("   mov {REG_FIRST_ARG}, {REG_EXEC_STATE_PTR}\n");
                asm_str += &format!("   mov {REG_SECOND_ARG}, {pre_compute_insns_ptr}\n");
                asm_str += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
                asm_str += &format!("   mov {REG_D}, {extern_handler_ptr}\n");
                asm_str += &format!("   call {REG_D}\n");
                asm_str += &format!("   cmp {REG_RETURN_VAL}, 1\n");

                asm_str += &Self::pop_internal_registers();
                asm_str += &Self::pop_address_space_start();
                asm_str += &Self::rv32_regs_to_xmm();
                asm_str += &format!("   je asm_run_end_{pc}\n");
                asm_str += &format!("   lea {REG_C}, [rip + map_pc_base]\n");
                asm_str += &format!("   pextrq {REG_A}, xmm3, 1\n"); // extract the upper 64 bits of the xmm3 register to REG_A
                asm_str += &format!("   mov {REG_A_W}, dword ptr [{REG_A}]\n");
                asm_str += &format!("   movsxd {REG_A}, [{REG_C} + {REG_A}]\n");
                asm_str += &format!("   add {REG_A}, {REG_C}\n");
                asm_str += &format!("   jmp {REG_A}\n");
            }
        }

        asm_str += &Self::dead_pc_handler_asm(
            &extern_handler_ptr,
            &pre_compute_insns_ptr,
            &set_pc_ptr,
            &instret_left_ptr,
        );

        // asm_run_end part. Only real instructions reach a per-pc suspend target; dead slots
        // use the shared exit path inside the dead-pc handler instead.
        for pc_idx in 0..num_pc_slots {
            let pc = pc_idx as u32 * DEFAULT_PC_STEP;
            let is_real = pc_idx
                .checked_sub(base_idx)
                .is_some_and(|idx| exe.program.instructions_and_debug_infos[idx].is_some());
            if !is_real {
                continue;
            }
            asm_str += &format!("asm_run_end_{pc}:\n");
            asm_str += &format!("   mov {REG_FIRST_ARG}, {REG_EXEC_STATE_PTR}\n");
            asm_str += &format!("   mov {REG_SECOND_ARG}, {REG_INSTRET_END}\n");
            asm_str += &format!("   mov {REG_D}, {instret_left_ptr}\n");
            asm_str += &format!("   call {REG_D}\n");
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

        for pc_idx in 0..num_pc_slots {
            let pc = pc_idx as u32 * DEFAULT_PC_STEP;
            asm_str += &format!("   .long asm_execute_pc_{pc} - map_pc_base\n");
        }

        // std::fs::write("/tmp/asm_dump.s", &asm_str).expect("failed to write asm_str");

        Ok(asm_str)
    }

    fn dead_pc_handler_asm(
        extern_handler_ptr: &str,
        pre_compute_insns_ptr: &str,
        set_pc_ptr: &str,
        instret_left_ptr: &str,
    ) -> String {
        let mut asm_str = String::new();

        // Dead-slot stubs enter with the dead pc held in REG_THIRD_ARG (= REG_C).
        asm_str += "asm_dead_pc_handler:\n";
        asm_str += &format!("    cmp {REG_INSTRET_END}, 0\n");
        asm_str += "    je asm_dead_pc_exit\n";
        asm_str += &format!("    dec {REG_INSTRET_END}\n");
        asm_str += &Self::xmm_to_rv32_regs();
        asm_str += &Self::push_address_space_start();
        asm_str += &Self::push_internal_registers();
        asm_str += &format!("   mov {REG_FIRST_ARG}, {REG_EXEC_STATE_PTR}\n");
        asm_str += &format!("   mov {REG_SECOND_ARG}, {pre_compute_insns_ptr}\n");
        asm_str += &format!("   mov {REG_D}, {extern_handler_ptr}\n");
        asm_str += &format!("   call {REG_D}\n");
        asm_str += &Self::pop_internal_registers();
        asm_str += &Self::pop_address_space_start();
        asm_str += &Self::rv32_regs_to_xmm();

        asm_str += "asm_dead_pc_exit:\n";
        asm_str += &Self::xmm_to_rv32_regs();
        asm_str += &format!("   mov {REG_FIRST_ARG}, {REG_EXEC_STATE_PTR}\n");
        asm_str += &format!("   mov {REG_SECOND_ARG}, {REG_C}\n");
        asm_str += &format!("   mov {REG_D}, {set_pc_ptr}\n");
        asm_str += &format!("   call {REG_D}\n");
        asm_str += &format!("   mov {REG_FIRST_ARG}, {REG_EXEC_STATE_PTR}\n");
        asm_str += &format!("   mov {REG_SECOND_ARG}, {REG_INSTRET_END}\n");
        asm_str += &format!("   mov {REG_D}, {instret_left_ptr}\n");
        asm_str += &format!("   call {REG_D}\n");
        asm_str += &Self::pop_external_registers();
        asm_str += &format!("   xor {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm_str += "    ret\n";

        asm_str
    }

    /// Creates a new instance for pure execution
    pub fn new<F, E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
    ) -> Result<Self, StaticProgramError>
    where
        F: PrimeField32,
        E: Executor<F>,
    {
        let program = &exe.program;
        let pre_compute_max_size = get_pre_compute_max_size(program, inventory);
        let mut pre_compute_buf = alloc_pre_compute_buf(program, pre_compute_max_size);
        let mut split_pre_compute_buf =
            split_pre_compute_buf(program, &mut pre_compute_buf, pre_compute_max_size);
        let pre_compute_insns = get_pre_compute_instructions::<F, ExecutionCtx, E>(
            program,
            inventory,
            &mut split_pre_compute_buf,
        )?;

        let asm_source = Self::create_pure_asm(exe, inventory, pre_compute_insns.as_ptr())?;
        let lib = asm_to_lib(&asm_source)?;

        let init_memory = exe.init_memory.clone();

        Ok(Self {
            system_config: inventory.config(),
            pre_compute_buf,
            pre_compute_insns,
            pc_start: exe.pc_start,
            init_memory,
            lib,
        })
    }

    /// Pure AOT execution, without metering, for the given `inputs`.
    /// this function executes the program until termination
    /// Returns the final VM state when execution stops.
    pub fn execute(
        &self,
        inputs: impl Into<Streams>,
        num_insns: Option<u64>,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        let vm_state =
            VmState::initial(self.system_config, &self.init_memory, self.pc_start, inputs);
        self.execute_from_state(vm_state, num_insns)
    }

    // Runs pure execution with AOT starting with `from_state` VmState
    // Runs for `num_insns` instructions if `num_insns` is not None
    // Otherwise executes until termination
    pub fn execute_from_state(
        &self,
        from_state: VmState<GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        let from_state_pc = from_state.pc();
        let ctx = ExecutionCtx::new(num_insns);
        let instret_left = ctx.instret_left;

        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::Pure);
        #[cfg(feature = "metrics")]
        let start_instret_left = instret_left;

        let mut vm_exec_state: Box<VmExecState<GuestMemory, ExecutionCtx>> =
            Box::new(VmExecState::new(from_state, ctx));

        tracing::info_span!("execute_pure").in_scope(|| unsafe {
            let asm_run: libloading::Symbol<AsmRunFn> = self
                .lib
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");

            let vm_exec_state_ptr =
                vm_exec_state.as_mut() as *mut VmExecState<GuestMemory, ExecutionCtx>;
            let pre_compute_insns_ptr = self.pre_compute_insns.as_ptr();

            asm_run(
                vm_exec_state_ptr.cast(),
                pre_compute_insns_ptr as *const c_void,
                from_state_pc,
                instret_left,
            );
        });

        #[cfg(feature = "metrics")]
        {
            let insns = start_instret_left - vm_exec_state.ctx.instret_left;
            metrics.record(insns);
        }

        if num_insns.is_some() {
            check_exit_code(vm_exec_state.exit_code)?;
        } else {
            check_termination(vm_exec_state.exit_code)?;
        }

        Ok(vm_exec_state.vm_state)
    }
}

/// Errors if exit code is either error or terminated with non-successful exit code.
fn check_exit_code(exit_code: Result<Option<u32>, ExecutionError>) -> Result<(), ExecutionError> {
    let exit_code = exit_code?;
    if let Some(exit_code) = exit_code {
        // This means execution did terminate
        if exit_code != ExitCode::Success as u32 {
            return Err(ExecutionError::FailedWithExitCode(exit_code));
        }
    }
    Ok(())
}

/// Same as [check_exit_code] but errors if program did not terminate.
fn check_termination(exit_code: Result<Option<u32>, ExecutionError>) -> Result<(), ExecutionError> {
    let did_terminate = matches!(exit_code.as_ref(), Ok(Some(_)));
    check_exit_code(exit_code)?;
    match did_terminate {
        true => Ok(()),
        false => Err(ExecutionError::DidNotTerminate),
    }
}
