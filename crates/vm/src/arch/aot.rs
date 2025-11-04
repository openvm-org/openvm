#![cfg(feature = "aot")]
use std::{ffi::c_void, fs, process::Command};

use libloading::Library;
use openvm_instructions::exe::{SparseMemoryImage, VmExe};
use openvm_stark_backend::p3_field::PrimeField32;
use rand::Rng;

use crate::{
    arch::{
        execution_mode::{ExecutionCtx, MeteredCtx, Segment},
        interpreter::{
            alloc_pre_compute_buf, get_metered_pre_compute_instructions,
            get_metered_pre_compute_max_size, get_pre_compute_instructions,
            get_pre_compute_max_size, split_pre_compute_buf, AlignedBuf, PreComputeInstruction,
        },
        AotError, ExecutionCtxTrait, ExecutionError, Executor, ExecutorInventory, ExitCode,
        MeteredExecutionCtxTrait, MeteredExecutor, StaticProgramError, Streams, SystemConfig,
        VmExecState, VmState,
    },
    system::memory::online::GuestMemory,
};

const REG_A: &str = "rcx";
const REG_A_W: &str = "ecx";

const REG_B: &str = "rax";
const REG_B_W: &str = "eax";

const REG_C: &str = "r10";
const REG_C_W: &str = "r10d";

const REG_AUX: &str = "r11";
const REG_EXEC_STATE_PTR: &str = "rbx";
const REG_INSNS_PTR: &str = "rbp";
const REG_PC: &str = "r13";
const REG_INSTRET: &str = "r14";

/// The assembly bridge build process requires the following tools:
/// GNU Binutils (provides `as` and `ar`)
/// Rust toolchain
/// Verify installation by `as --version`, `ar --version` and `cargo --version`
/// Refer to AOT.md for further clarification about AOT
///  
pub struct AotInstance<'a, F, Ctx> {
    init_memory: SparseMemoryImage,
    system_config: SystemConfig,
    // SAFETY: this is not actually dead code, but `pre_compute_insns` contains raw pointer refers
    // to this buffer.
    #[allow(dead_code)]
    pre_compute_buf: AlignedBuf,
    lib: Library,
    pre_compute_insns_box: Box<[PreComputeInstruction<'a, F, Ctx>]>,
    pc_start: u32,
}

type AsmRunFn = unsafe extern "C" fn(
    vm_exec_state_ptr: *mut c_void,
    pre_compute_insns_ptr: *const c_void,
    from_state_pc: u32,
    from_state_instret: u64,
    instret_end: u64,
);

impl<'a, F, Ctx> AotInstance<'a, F, Ctx>
where
    F: PrimeField32,
    Ctx: ExecutionCtxTrait,
{
    fn push_external_registers() -> String {
        let mut asm_str = String::new();
        asm_str += "    push rbp\n";
        asm_str += "    push rbx\n";
        asm_str += "    push r12\n";
        asm_str += "    push r13\n";
        asm_str += "    push r14\n";
        asm_str += "    push r15\n";

        asm_str
    }

    fn pop_external_registers() -> String {
        let mut asm_str = String::new();
        asm_str += "    pop r15\n";
        asm_str += "    pop r14\n";
        asm_str += "    pop r13\n";
        asm_str += "    pop r12\n";
        asm_str += "    pop rbx\n";
        asm_str += "    pop rbp\n";

        asm_str
    }

    fn debug_cur_string(str: &String) {
        println!("DEBUG");
        println!("{}", str);
    }

    fn push_xmm_regs() -> String {
        let mut asm_str = String::new();
        asm_str += "    sub rsp, 16*16";
        asm_str += "    movaps [rsp + 0*16], xmm0\n";
        asm_str += "    movaps [rsp + 1*16], xmm1\n";
        asm_str += "    movaps [rsp + 2*16], xmm2\n";
        asm_str += "    movaps [rsp + 3*16], xmm3\n";
        asm_str += "    movaps [rsp + 4*16], xmm4\n";
        asm_str += "    movaps [rsp + 5*16], xmm5\n";
        asm_str += "    movaps [rsp + 6*16], xmm6\n";
        asm_str += "    movaps [rsp + 7*16], xmm7\n";
        asm_str += "    movaps [rsp + 8*16], xmm8\n";
        asm_str += "    movaps [rsp + 9*16], xmm9\n";
        asm_str += "    movaps [rsp + 10*16], xmm10\n";
        asm_str += "    movaps [rsp + 11*16], xmm11\n";
        asm_str += "    movaps [rsp + 12*16], xmm12\n";
        asm_str += "    movaps [rsp + 13*16], xmm13\n";
        asm_str += "    movaps [rsp + 14*16], xmm14\n";
        asm_str += "    movaps [rsp + 15*16], xmm15\n";

        asm_str
    }
    fn pop_xmm_regs() -> String {
        let mut asm_str = String::new();
        asm_str += "    movaps xmm0, [rsp + 0*16]\n";
        asm_str += "    movaps xmm1, [rsp + 1*16]\n";
        asm_str += "    movaps xmm2, [rsp + 2*16]\n";
        asm_str += "    movaps xmm3, [rsp + 3*16]\n";
        asm_str += "    movaps xmm4, [rsp + 4*16]\n";
        asm_str += "    movaps xmm5, [rsp + 5*16]\n";
        asm_str += "    movaps xmm6, [rsp + 6*16]\n";
        asm_str += "    movaps xmm7, [rsp + 7*16]\n";
        asm_str += "    movaps xmm8, [rsp + 8*16]\n";
        asm_str += "    movaps xmm9, [rsp + 9*16]\n";
        asm_str += "    movaps xmm10, [rsp + 10*16]\n";
        asm_str += "    movaps xmm11, [rsp + 11*16]\n";
        asm_str += "    movaps xmm12, [rsp + 12*16]\n";
        asm_str += "    movaps xmm13, [rsp + 13*16]\n";
        asm_str += "    movaps xmm14, [rsp + 14*16]\n";
        asm_str += "    movaps xmm15, [rsp + 15*16]\n";
        asm_str += "    add rsp, 16*16\n";

        asm_str
    }

    fn push_internal_registers() -> String {
        let mut asm_str = String::new();
        asm_str += "    push rax\n";
        asm_str += "    push rcx\n";
        asm_str += "    push rdx\n";
        asm_str += "    push r8\n";
        asm_str += "    push r9\n";
        asm_str += "    push r10\n";
        asm_str += "    push r11\n";
        // asm_str += &Self::push_xmm_regs();

        asm_str
    }

    fn pop_internal_registers() -> String {
        let mut asm_str = String::new();
        asm_str += "    pop r11\n";
        asm_str += "    pop r10\n";
        asm_str += "    pop r9\n";
        asm_str += "    pop r8\n";
        asm_str += "    pop rdx\n";
        asm_str += "    pop rcx\n";
        asm_str += "    pop rax\n";
        // asm_str += &Self::pop_xmm_regs();

        asm_str
    }

    /*
    fn sync_vm_registers() -> String  {
        let mut asm_str = String::new();
        for r in 0..16 {
            asm_str += &format!("");
            asm_str += &format!("   mov xmm{}, \n", r);
        }
    }
    */

    // r15 stores vm_register_address
    fn rv32_regs_to_xmm() -> String {
        let mut asm_str = String::new();

        for r in 0..16 {
            asm_str += &format!("   mov rdi, [r15 + 8*{}]\n", r);
            asm_str += &format!("   pinsrq xmm{}, rdi, 0\n", r);
        }

        asm_str
    }

    fn pop_address_space_start() -> String {
        let mut asm_str = String::new();
        // For byte alignment
        asm_str += "   pop rdi\n";
        asm_str += "   pop rdi\n";
        asm_str += "   pinsrq xmm2, rdi, 1\n";
        asm_str += "   pop rdi\n";
        asm_str += "   pinsrq xmm1, rdi, 1\n";
        asm_str += "   pop rdi\n";
        asm_str += "   pinsrq xmm0, rdi, 1\n";
        asm_str
    }

    fn xmm_to_rv32_regs() -> String {
        let mut asm_str = String::new();

        for r in 0..16 {
            // at each iteration we save register 2r and 2r+1 of the guest mem to xmm
            asm_str += &format!("   movq [r15 + 8*{}], xmm{}\n", r, r);
        }

        asm_str
    }

    fn push_address_space_start() -> String {
        let mut asm_str = String::new();

        asm_str += "   pextrq rdi, xmm0, 1\n";
        asm_str += "   push rdi\n";
        asm_str += "   pextrq rdi, xmm1, 1\n";
        asm_str += "   push rdi\n";
        asm_str += "   pextrq rdi, xmm2, 1\n";
        asm_str += "   push rdi\n";
        // For byte alignment
        asm_str += "   push rdi\n";

        asm_str
    }

    fn initialize_xmm_regs() -> String {
        let mut asm_str = String::new();
        asm_str += "    mov rax, 0\n";
        for r in 0..16 {
            asm_str += &format!("   pinsrq xmm{}, rax, 0\n", r);
        }

        asm_str
    }

    pub fn to_i16(c: F) -> i16 {
        let c_u24 = (c.as_canonical_u64() & 0xFFFFFF) as u32;
        let c_i24 = ((c_u24 << 8) as i32) >> 8;
        c_i24 as i16
    }

    pub fn create_asm<E>(
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
        asm_str += "    mov r14, rcx\n";
        asm_str += "    mov r12, r8\n";

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
            asm_str += &format!("asm_execute_pc_{}:\n", pc);

            // Check if we should suspend or not

            asm_str += "    cmp r14, r12\n";
            asm_str += "    jae asm_run_end\n";

            if instruction.opcode.as_usize() == 0 {
                // terminal opcode has no associated executor, so can handle with default fallback
                asm_str += &Self::xmm_to_rv32_regs();
                asm_str += &Self::push_address_space_start();
                asm_str += &Self::push_internal_registers();
                asm_str += "    mov rdi, rbx\n";
                asm_str += "    mov rsi, rbp\n";
                asm_str += "    mov rdx, r13\n";
                asm_str += "    mov rcx, r14\n";
                asm_str += "    call extern_handler\n";
                asm_str += "    add r14, 1\n"; // increment the instret
                asm_str += "    mov r13, rax\n"; // move the return value of the extern_handler into r13
                asm_str += "    AND rax, 1\n"; // check if the return value is 1
                asm_str += "    cmp rax, 1\n"; // compare the return value with 1
                asm_str += &Self::pop_internal_registers(); // pop the internal registers from the stack
                asm_str += &Self::pop_address_space_start();
                // read the memory from the memory location of the RV32 registers in `GuestMemory`
                // registers, to the appropriate XMM registers
                asm_str += "    je asm_run_end\n"; // jump to end, if the return value is 1 (indicates that the program should
                                                   // terminate)
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
                );
            }
        }

        // asm_run_end part
        asm_str += "asm_run_end:\n";
        asm_str += "    sub r13, 1\n";
        asm_str += "    mov rdi, rbx\n";
        asm_str += "    mov rsi, rbp\n";
        asm_str += "    mov rdx, r13\n";
        asm_str += "    mov rcx, r14\n";
        asm_str += "    call set_instret_and_pc\n";
        asm_str += "    xor rax, rax\n";
        asm_str += &Self::pop_external_registers();
        asm_str += "    ret\n";
        asm_str += "\n";

        // map_pc_base part
        asm_str += ".section .rodata\n";
        asm_str += "map_pc_base:\n";

        for i in 0..(pc_base / 4) {
            asm_str += &format!("   .long asm_execute_pc_{} - map_pc_base\n", i * 4);
        }

        for (pc, _instruction, _) in exe.program.enumerate_by_pc() {
            asm_str += &format!("   .long asm_execute_pc_{} - map_pc_base\n", pc);
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

        let asm_source = Self::create_asm(&exe, &inventory)?;
        fs::write(
            format!("{}/src/{}.s", src_asm_bridge_dir_str, asm_name),
            asm_source,
        )
        .expect("Failed to write generated assembly");

        // ar rcs libasm_runtime.a asm_run.o
        // cargo rustc -- -L /home/ubuntu/openvm/crates/vm/src/arch/asm_bridge -l static=asm_runtime

        // run the below command from the `src_asm_bridge_dir` directory
        // as src/asm_run.s -o asm_run.o
        let status = Command::new("as")
            .current_dir(&src_asm_bridge_dir)
            .args([
                &format!("src/{}.s", asm_name),
                "-o",
                &format!("{}.o", asm_name),
            ])
            .status()
            .expect("Failed to assemble the file into an object file");

        assert!(
            status.success(),
            "as src/<asm_name>.s -o <asm_name>.o failed with exit code: {:?}",
            status.code()
        );

        let status = Command::new("ar")
            .current_dir(&src_asm_bridge_dir)
            .args([
                "rcs",
                &format!("lib{}.a", asm_name),
                &format!("{}.o", asm_name),
            ])
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
                &format!("static={}", asm_name),
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
        let _ = fs::remove_file(format!("{}/src/{}.s", src_asm_bridge_dir_str, asm_name));
        let _ = fs::remove_file(format!("{}/{}.o", src_asm_bridge_dir_str, asm_name));
        let _ = fs::remove_file(format!("{}/lib{}.a", src_asm_bridge_dir_str, asm_name));
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
        let from_state_instret = from_state.instret();
        let from_state_pc = from_state.pc();
        let ctx = ExecutionCtx::new(num_insns);
        let instret_end = ctx.instret_end;

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

            #[cfg(target_arch = "x86_64")]
            let start_cycles = core::arch::x86_64::_rdtsc();
            let start_time = std::time::Instant::now();

            asm_run(
                vm_exec_state_ptr as *mut c_void,
                pre_compute_insns_ptr as *const c_void,
                from_state_pc,
                from_state_instret,
                instret_end,
            );

            #[cfg(target_arch = "x86_64")]
            let end_cycles = core::arch::x86_64::_rdtsc();
            let duration = start_time.elapsed();

            let instructions_executed = vm_exec_state.vm_state.instret() - from_state_instret;
            
            #[cfg(target_arch = "x86_64")]
            {
                let cycles = end_cycles - start_cycles;
                let ipc = instructions_executed as f64 / cycles as f64;
                let ips = instructions_executed as f64 / duration.as_secs_f64();
                println!("[aot profiling] instructions: {}", instructions_executed);
                println!("[aot profiling] cycles: {}", cycles);
                println!("[aot profiling] IPC: {:.3}", ipc);
                println!("[aot profiling] IPS: {:.2e}", ips);
                println!("[aot profiling] duration: {:?}", duration);
            }
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

impl<'a, F, Ctx> AotInstance<'a, F, Ctx>
where
    F: PrimeField32,
    Ctx: MeteredExecutionCtxTrait,
{
    /// Creates a new instance for metered execution.
    pub fn new_metered<E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<Self, StaticProgramError>
    where
        E: MeteredExecutor<F>,
    {
        let default_name = String::from("asm_x86_run");
        Self::new_metered_with_asm_name(inventory, exe, executor_idx_to_air_idx, &default_name)
    }

    /// Creates a new interpreter instance for metered execution.
    pub fn new_metered_with_asm_name<E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
        asm_name: &String,
    ) -> Result<Self, StaticProgramError>
    where
        E: MeteredExecutor<F>,
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

        let src_asm_bridge_dir =
            std::path::Path::new(manifest_dir).join("src/arch/asm_bridge_metered");
        let src_asm_bridge_dir_str = src_asm_bridge_dir.to_str().unwrap();

        // ar rcs libasm_runtime.a asm_run.o
        // cargo rustc -- -L /home/ubuntu/openvm/crates/vm/src/arch/asm_bridge -l static=asm_runtime

        // run the below command from the `src_asm_bridge_dir` directory
        // as src/asm_run.s -o asm_run.o
        let status = Command::new("as")
            .current_dir(&src_asm_bridge_dir)
            .args([
                &format!("src/{}.s", asm_name),
                "-o",
                &format!("{}.o", asm_name),
            ])
            .status()
            .expect("Failed to assemble the file into an object file");

        assert!(
            status.success(),
            "as src/<asm_name>.s -o <asm_name>.o failed with exit code: {:?}",
            status.code()
        );

        let status = Command::new("ar")
            .current_dir(&src_asm_bridge_dir)
            .args([
                "rcs",
                &format!("lib{}.a", asm_name),
                &format!("{}.o", asm_name),
            ])
            .status()
            .expect("Create a static library");

        assert!(
            status.success(),
            "ar rcs lib<asm_name>.a <asm_name>.o failed with exit code: {:?}",
            status.code()
        );

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
                &format!("static={}", asm_name),
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
            .join("libasm_bridge_metered.so");
        let lib = unsafe { Library::new(&lib_path).expect("Failed to load library") };

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
}

impl<F> AotInstance<'_, F, MeteredCtx>
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
        let from_state_instret = from_state.instret();
        let from_state_pc = from_state.pc();

        let mut vm_exec_state: Box<VmExecState<F, GuestMemory, MeteredCtx>> =
            Box::new(VmExecState::new(from_state, ctx));

        unsafe {
            let asm_run: libloading::Symbol<AsmRunFn> = self
                .lib
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");

            let vm_exec_state_ptr =
                &mut *vm_exec_state as *mut VmExecState<F, GuestMemory, MeteredCtx>;
            let pre_compute_insns_ptr = self.pre_compute_insns_box.as_ptr();

            asm_run(
                vm_exec_state_ptr as *mut c_void,
                pre_compute_insns_ptr as *const c_void,
                from_state_pc,
                from_state_instret,
                0, /* TODO: this is a placeholder because in the pure case asm_run needs to take
                    * in 5 args. Fix later */
            );
        }

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
}
