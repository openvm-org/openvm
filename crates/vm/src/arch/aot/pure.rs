use std::{ffi::c_void, fs, process::Command};

use libloading::Library;
use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;
use rand::Rng;

use super::{AotInstance, AsmRunFn};
use crate::{
    arch::{
        execution_mode::{ExecutionCtx, ExecutionCtxTrait},
        interpreter::{
            alloc_pre_compute_buf, get_pre_compute_instructions, get_pre_compute_max_size,
            split_pre_compute_buf, PreComputeInstruction,
        },
        AotError, ExecutionError, Executor, ExecutorInventory, ExitCode, StaticProgramError,
        Streams, VmExecState, VmState,
    },
    system::memory::online::GuestMemory,
};

impl<'a, F, Ctx> AotInstance<'a, F, Ctx>
where
    F: PrimeField32,
    Ctx: ExecutionCtxTrait,
{
    pub fn create_pure_asm<E>(
        exe: &VmExe<F>,
        inventory: &ExecutorInventory<E>,
    ) -> Result<String, StaticProgramError>
    where
        E: Executor<F>,
    {
        let mut asm_str = String::new();
        // generate the assembly based on exe.program

        // header part
        asm_str += ".intel_syntax noprefix\n";
        asm_str += ".code64\n";
        asm_str += ".section .text\n";
        asm_str += ".global asm_run_internal\n";

        // asm_run_internal part
        asm_str += "asm_run_internal:\n";
        asm_str += &Self::push_external_registers();
        asm_str += "    mov rbx, rdi\n";
        asm_str += "    mov rbp, rsi\n";
        asm_str += "    mov r13, rdx\n";
        asm_str += "    mov r12, rcx\n";

        asm_str += &Self::push_internal_registers();
        // Store the start of register address space in r15
        asm_str += "    mov rdi, rbx\n";
        asm_str += "    call get_vm_register_addr\n";
        asm_str += "    mov r15, rax\n";
        // Store the start of address space 2 in high 64 bits of xmm0
        asm_str += "    mov rdi, rbx\n";
        asm_str += "    mov rsi, 2\n";
        asm_str += "    call get_vm_address_space_addr\n";
        asm_str += "    pinsrq  xmm0, rax, 1\n";
        // Store the start of address space 3 in high 64 bits of xmm1
        asm_str += "    mov rdi, rbx\n";
        asm_str += "    mov rsi, 3\n";
        asm_str += "    call get_vm_address_space_addr\n";
        asm_str += "    pinsrq  xmm1, rax, 1\n";
        // Store the start of address space 4 in high 64 bits of xmm2
        asm_str += "    mov rdi, rbx\n";
        asm_str += "    mov rsi, 4\n";
        asm_str += "    call get_vm_address_space_addr\n";
        asm_str += "    pinsrq  xmm2, rax, 1\n";
        asm_str += &Self::pop_internal_registers();

        asm_str += &Self::initialize_xmm_regs();

        asm_str += "    lea rdx, [rip + map_pc_base]\n";
        asm_str += "    movsxd rcx, [rdx + r13]\n";
        asm_str += "    add rcx, rdx\n";
        asm_str += "    jmp rcx\n";

        // asm_execute_pc_{pc_num}
        // do fallback first for now but expand per instruction

        let pc_base = exe.program.pc_base;

        for i in 0..(pc_base / 4) {
            asm_str += &format!("asm_execute_pc_{}:", i * 4);
            asm_str += "\n";
            asm_str += "\n";
        }

        for (pc, instruction, _) in exe.program.enumerate_by_pc() {
            /* Preprocessing step, to check if we should suspend or not */
            asm_str += &format!("asm_execute_pc_{pc}:\n");

            // Check if we should suspend or not
            asm_str += "    cmp r12, 0\n";
            asm_str += &format!("    je asm_run_end_{pc}\n");
            asm_str += "    dec r12\n";

            if instruction.opcode.as_usize() == 0 {
                // terminal opcode has no associated executor, so can handle with default fallback
                asm_str += &Self::xmm_to_rv32_regs();
                asm_str += &Self::push_address_space_start();
                asm_str += &Self::push_internal_registers();
                asm_str += "    mov rdi, rbx\n";
                asm_str += "    mov rsi, rbp\n";
                asm_str += &format!("    mov rdx, {pc}\n");
                asm_str += "    call extern_handler\n";
                asm_str += "    mov r13, rax\n"; // move the return value of the extern_handler into r13
                asm_str += "    AND rax, 1\n"; // check if the return value is 1
                asm_str += "    cmp rax, 1\n"; // compare the return value with 1
                asm_str += &Self::pop_internal_registers(); // pop the internal registers from the stack
                asm_str += &Self::pop_address_space_start();
                // read the memory from the memory location of the RV32 registers in `GuestMemory`
                // registers, to the appropriate XMM registers
                asm_str += &format!("   je asm_run_end_{pc}\n");
                asm_str += "    lea rdx, [rip + map_pc_base]\n"; // load the base address of the map_pc_base section
                asm_str += "    movsxd rcx, [rdx + r13]\n"; // load the offset of the next instruction (r13 is the next pc)
                asm_str += "    add rcx, rdx\n"; // add the base address and the offset
                asm_str += "    jmp rcx\n"; // jump to the next instruction (rcx is the next instruction)
                asm_str += "\n";
                continue;
            }

            let executor = inventory
                .get_executor(instruction.opcode)
                .expect("executor not found for opcode");

            if executor.is_aot_supported(&instruction) {
                let segment =
                    executor
                        .generate_x86_asm(&instruction, pc)
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
            } else {
                asm_str += &Self::xmm_to_rv32_regs();
                asm_str += &Self::push_address_space_start();
                asm_str += &executor.fallback_to_interpreter(
                    &Self::push_internal_registers(),
                    &Self::pop_internal_registers(),
                    &(Self::pop_address_space_start() + &Self::rv32_regs_to_xmm()),
                    &instruction,
                    pc,
                );
            }
        }

        // asm_run_end part
        for (pc, _instruction, _) in exe.program.enumerate_by_pc() {
            asm_str += &format!("asm_run_end_{pc}:\n");
            asm_str += "    mov rdi, rbx\n";
            asm_str += "    mov rsi, rbp\n";
            asm_str += &format!("    mov rdx, {pc}\n");
            asm_str += "    call set_pc\n";
            asm_str += "    xor rax, rax\n";
            asm_str += &Self::pop_external_registers();
            asm_str += "    ret\n";
            asm_str += "\n";
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
    /// Creates a new instance for pure execution
    pub fn new<E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
    ) -> Result<Self, StaticProgramError>
    where
        E: Executor<F>,
    {
        let _default_name = String::from("asm_x86_run");
        let random_name = format!("asm_x86_run_{}", rand::thread_rng().gen_range(0..1000000));
        Self::new_with_asm_name(inventory, exe, &random_name)
    }

    /// Creates a new instance for pure execution
    /// Specify the name of the asm file
    pub fn new_with_asm_name<E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
        asm_name: &String, // name of the asm file we write into
    ) -> Result<Self, StaticProgramError>
    where
        E: Executor<F>,
    {
        // source asm_bridge directory
        // this is fixed
        // can unwrap because its fixed and guaranteed to exist
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let root_dir = std::path::Path::new(manifest_dir)
            .parent()
            .unwrap()
            .parent()
            .unwrap();

        let src_asm_bridge_dir = std::path::Path::new(manifest_dir).join("src/arch/asm_bridge");
        let src_asm_bridge_dir_str = src_asm_bridge_dir.to_str().unwrap();

        let asm_source = Self::create_pure_asm(exe, inventory)?;
        fs::write(
            format!("{src_asm_bridge_dir_str}/src/{asm_name}.s"),
            asm_source,
        )
        .expect("Failed to write generated assembly");

        // ar rcs libasm_runtime.a asm_run.o
        // cargo rustc -- -L /home/ubuntu/openvm/crates/vm/src/arch/asm_bridge -l static=asm_runtime

        // run the below command from the `src_asm_bridge_dir` directory
        // as src/asm_run.s -o asm_run.o
        let status = Command::new("as")
            .current_dir(&src_asm_bridge_dir)
            .args([&format!("src/{asm_name}.s"), "-o", &format!("{asm_name}.o")])
            .status()
            .expect("Failed to assemble the file into an object file");

        assert!(
            status.success(),
            "as src/<asm_name>.s -o <asm_name>.o failed with exit code: {:?}",
            status.code()
        );

        let status = Command::new("ar")
            .current_dir(&src_asm_bridge_dir)
            .args(["rcs", &format!("lib{asm_name}.a"), &format!("{asm_name}.o")])
            .status()
            .expect("Create a static library");

        assert!(
            status.success(),
            "ar rcs lib<asm_name>.a <asm_name>.o failed with exit code: {:?}",
            status.code()
        );

        // library goes to `workspace_dir/target/{asm_name}/release/libasm_bridge.so`

        let status = Command::new("cargo")
            .current_dir(&src_asm_bridge_dir)
            .args([
                "rustc",
                "--release",
                &format!(
                    "--target-dir={}/target/{}",
                    root_dir.to_str().unwrap(),
                    asm_name
                ),
                "--",
                "-L",
                src_asm_bridge_dir_str,
                "-l",
                &format!("static={asm_name}"),
            ])
            .status()
            .expect("Creating the dynamic library");

        assert!(
            status.success(),
            "Cargo build failed with exit code: {:?}",
            status.code()
        );

        let lib_path = root_dir
            .join("target")
            .join(asm_name)
            .join("release")
            .join("libasm_bridge.so");

        let lib = unsafe { Library::new(&lib_path).expect("Failed to load library") };
        // Cleanup artifacts after library is loaded into memory
        let _ = fs::remove_file(format!("{src_asm_bridge_dir_str}/src/{asm_name}.s"));
        let _ = fs::remove_file(format!("{src_asm_bridge_dir_str}/{asm_name}.o"));
        let _ = fs::remove_file(format!("{src_asm_bridge_dir_str}/lib{asm_name}.a"));
        let _ = fs::remove_dir_all(root_dir.join("target").join(asm_name));

        let program = &exe.program;
        let pre_compute_max_size = get_pre_compute_max_size(program, inventory);
        let mut pre_compute_buf = alloc_pre_compute_buf(program, pre_compute_max_size);
        let mut split_pre_compute_buf =
            split_pre_compute_buf(program, &mut pre_compute_buf, pre_compute_max_size);
        let pre_compute_insns = get_pre_compute_instructions::<F, Ctx, E>(
            program,
            inventory,
            &mut split_pre_compute_buf,
        )?;
        let pre_compute_insns_box: Box<[PreComputeInstruction<'a, F, Ctx>]> =
            pre_compute_insns.into_boxed_slice();

        let init_memory = exe.init_memory.clone();

        Ok(Self {
            system_config: inventory.config().clone(),
            pre_compute_buf,
            pre_compute_insns_box,
            pc_start: exe.pc_start,
            init_memory,
            lib,
        })
    }

    pub fn create_initial_vm_state(&self, inputs: impl Into<Streams<F>>) -> VmState<F> {
        VmState::initial(
            &self.system_config,
            &self.init_memory,
            self.pc_start,
            inputs,
        )
    }
}

impl<F> AotInstance<'_, F, ExecutionCtx>
where
    F: PrimeField32,
{
    /// Pure AOT execution, without metering, for the given `inputs`.
    /// this function executes the program until termination
    /// Returns the final VM state when execution stops.
    pub fn execute(
        &self,
        inputs: impl Into<Streams<F>>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F, GuestMemory>, ExecutionError> {
        let vm_state = VmState::initial(
            &self.system_config,
            &self.init_memory,
            self.pc_start,
            inputs,
        );
        self.execute_from_state(vm_state, num_insns)
    }

    // Runs pure execution with AOT starting with `from_state` VmState
    // Runs for `num_insns` instructions if `num_insns` is not None
    // Otherwise executes until termination
    pub fn execute_from_state(
        &self,
        from_state: VmState<F, GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F, GuestMemory>, ExecutionError> {
        let from_state_pc = from_state.pc();
        let ctx = ExecutionCtx::new(num_insns);
        let instret_left = ctx.instret_left;

        let mut vm_exec_state: Box<VmExecState<F, GuestMemory, ExecutionCtx>> =
            Box::new(VmExecState::new(from_state, ctx));

        unsafe {
            let asm_run: libloading::Symbol<AsmRunFn> = self
                .lib
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");

            let vm_exec_state_ptr =
                &mut *vm_exec_state as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            let pre_compute_insns_ptr = self.pre_compute_insns_box.as_ptr();

            asm_run(
                vm_exec_state_ptr as *mut c_void,
                pre_compute_insns_ptr as *const c_void,
                from_state_pc,
                instret_left,
            );
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
