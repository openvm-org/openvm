#![cfg(feature = "aot")]
use std::{ffi::c_void, process::Command, fs};
use rand::Rng;

use libloading::Library;
use openvm_instructions::exe::{SparseMemoryImage, VmExe};
use openvm_stark_backend::p3_field::PrimeField32;

use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_instructions::LocalOpcode;

use crate::{
    arch::{
        execution_mode::{ExecutionCtx, MeteredCtx, Segment},
        interpreter::{
            alloc_pre_compute_buf, get_metered_pre_compute_instructions,
            get_metered_pre_compute_max_size, get_pre_compute_instructions,
            get_pre_compute_max_size, split_pre_compute_buf, AlignedBuf, PreComputeInstruction,
        },
        ExecutionCtxTrait, ExecutionError, Executor, ExecutorInventory, ExitCode,
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
    
    fn debug_cur_sting(str: &String) {
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
            asm_str += &format!("   movq xmm{}, [r15 + 8*{}]\n", r, r);
        }

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

    fn initialize_xmm_regs() -> String {
        let mut asm_str = String::new();

        for r in 0..16 {
            asm_str += &format!("   pxor xmm{}, xmm{}\n", r, r);
        }

        asm_str
    }

    pub fn to_i16(c: F) -> i16 {
        let c_u24 = (c.as_canonical_u64() & 0xFFFFFF) as u32;
        let c_i24 = ((c_u24 << 8) as i32) >> 8; 
        c_i24 as i16
    }

    pub fn create_asm(exe: &VmExe<F>) -> String {
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

        asm_str += &Self::push_internal_registers();
        asm_str += "    mov rdi, rbx\n";
        asm_str += "    call get_vm_register_addr\n";
        asm_str += "    mov r15, rax\n";
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

            if instruction.opcode == BaseAluOpcode::ADD.global_opcode() {
                // Spec: ADD_RV32	a,b,c,1,e	[a:4]_1 = [b:4]_1 + [c:4]_e. 
                // Overflow is ignored and the lower 32-bits are written to the destination.

                asm_str += &format!("asm_execute_pc_{}:\n", pc);

                asm_str += &Self::xmm_to_rv32_regs();
                asm_str += &Self::push_internal_registers();

                asm_str += "    mov rdi, r14\n";
                asm_str += "    mov rsi, r13\n";
                asm_str += "    mov rdx, rbx\n";
                asm_str += "    call should_suspend\n";
                asm_str += "    cmp rax, 1\n";

                asm_str += &Self::pop_internal_registers();
                asm_str += &Self::rv32_regs_to_xmm();
                asm_str += "    je asm_run_end\n";             
    
                
                let a : i16 = Self::to_i16(instruction.a);
                let b : i16 = Self::to_i16(instruction.b);
                let c : i16 = Self::to_i16(instruction.c);
                let e : i16 = Self::to_i16(instruction.e);

                assert_eq!(a % 4, 0);
                assert_eq!(b % 4, 0);    

                // perform the operation
                if e == 0 {
                    // reg_a = xmm_b + c (immediate)
                    // 
                    
                    /*
                    Plan for e = 0

                    REG_A = a / 4
                    REG_B = b / 4
                    
                    XMM_MAP_REG_A 
                    XMM_MAP_REG_B

                    1. move from the XMM_MAP_REG_B to x86 reg_a
                    2. add x86 reg_a by c
                    3. set xmm register XMM_MAP_REG_A to REG_A_W
                    */

                    let xmm_map_reg_a = if (a/4) % 2 == 0 {
                        a/8
                    } else {
                        ((a/4)-1)/2
                    };

                    let xmm_map_reg_b = if (b/4) % 2 == 0 {
                        b/8
                    } else {
                        ((b/4)-1)/2
                    };

                    if (b/4)%2 == 0 {
                        // get the [0:32) bits of xmm_map_reg_b
                        asm_str += &format!("   vmovd {REG_A}, xmm{}\n", xmm_map_reg_b);                                            
                    } else {
                        // get the [32:64) bits of xmm_map_reg_b
                        asm_str += &format!("   vpextrd {REG_A_W}, xmm{}, 1\n", xmm_map_reg_b);
                    }
                    
                    // REG_A += c
                    asm_str += &format!("   add {REG_A}, {c}\n");

                    
                    if (a/4)%2 == 0 {
                        // make the [0:32) bits of xmm_map_reg_a equal to REG_A_W without modifying the other bits
                        asm_str += &format!("   vpinsrd xmm{}, xmm{}, {REG_A_W}, 0\n", xmm_map_reg_a, xmm_map_reg_a);
                    } else {
                        // make the [32:64) bits of xmm_map_reg_a equal to REG_A_W without modifying the other bits
                        asm_str += &format!("   vpinsrd xmm{}, xmm{}, {REG_A_W}, 1\n", xmm_map_reg_a, xmm_map_reg_a);
                    }
                    
                } else {
                    // reg_a = xmm_b + reg_c

                    let xmm_map_reg_a = if (a/4) % 2 == 0 {
                        a/8
                    } else {
                        ((a/4)-1)/2
                    };

                    let xmm_map_reg_b = if (b/4) % 2 == 0 {
                        b/8
                    } else {
                        ((b/4)-1)/2
                    };

                    assert_eq!(c % 4, 0);
                    let xmm_map_reg_c = if (c/4) % 2 == 0 {
                        c/8
                    } else {
                        ((c/4)-1)/2
                    };
                    
                    if (b/4)%2 == 0 {
                        // get the [0:32) bits of xmm_map_reg_b
                        asm_str += &format!("   vmovd {REG_A}, xmm{}\n", xmm_map_reg_b);                                            
                    } else {
                        // get the [32:64) bits of xmm_map_reg_b
                        asm_str += &format!("   vpextrd {REG_A_W}, xmm{}, 1\n", xmm_map_reg_b);
                    }

                    if (c/4) % 2 == 0 {
                        // get the [0:32) bits of xmm_map_reg_c
                        asm_str += &format!("   vmovd {REG_C}, xmm{}\n", xmm_map_reg_c); 
                    } else {
                        // get the [32:64) bits of xmm_map_reg_b
                        asm_str += &format!("   vpextrd {REG_C_W}, xmm{}, 1\n", xmm_map_reg_c);
                    }

                      

                    // reg_a += reg_c
                    asm_str += &format!("   add {REG_A}, {REG_C}\n");

                    // place back from general register to xmm register
                    if (a/4)%2 == 0 {
                        asm_str += &format!("   vpinsrd xmm{}, xmm{}, {REG_A_W}, 0\n", xmm_map_reg_a, xmm_map_reg_a);
                    } else {
                        asm_str += &format!("   vpinsrd xmm{}, xmm{}, {REG_A_W}, 1\n", xmm_map_reg_a, xmm_map_reg_a);
                    }
                }

                // increment pc
                asm_str += &format!("   add {REG_PC}, 4\n");
                asm_str += &format!("   add {REG_INSTRET}, 1\n");

                // let it fall to the next instruction 

                continue;
            }

           asm_str += &format!("asm_execute_pc_{}:\n", pc);

            /*
            Invariant to be maintained before and after the call

            rbx -> vm_exec_state_ptr
            rbp -> pre_compute_insns_ptr
            r13 -> cur_pc 
            r14 -> cur_instret
            */
    
            /*
            call should_suspend with parameters
            - cur_instret
            - cur_pc 
            - vm_exec_state_ptr
            */

            asm_str += &Self::xmm_to_rv32_regs();
            asm_str += &Self::push_internal_registers();

            asm_str += "    mov rdi, r14\n";
            asm_str += "    mov rsi, r13\n";
            asm_str += "    mov rdx, rbx\n";

            /*
            should_suspend may change 
            rcx, rdx, r8, r9, r10, r11

            rax holds the return value which is the next pc
            */

            asm_str += "    call should_suspend\n";
            asm_str += "    cmp rax, 1\n";

            asm_str += &Self::pop_internal_registers();
    
            asm_str += "    je asm_run_end\n";
    
            asm_str += &Self::push_internal_registers();
    
            asm_str += "    mov rdi, rbx\n";
            asm_str += "    mov rsi, rbp\n";
            asm_str += "    mov rdx, r13\n";
            asm_str += "    mov rcx, r14\n";
            
            asm_str += "    call extern_handler\n";
            asm_str += "    add r14, 1\n";
            asm_str += "    mov r13, rax\n";
            asm_str += "    AND rax, 1\n";
            asm_str += "    cmp rax, 1\n";
            
            asm_str += &Self::pop_internal_registers();
            asm_str += &Self::rv32_regs_to_xmm();
            asm_str += "    je asm_run_end\n";
            asm_str += "    lea rdx, [rip + map_pc_base]\n";   
            asm_str += "    movsxd rcx, [rdx + r13]\n";               
            asm_str += "    add rcx, rdx\n";
            asm_str += "    jmp rcx\n";
            asm_str += "\n";

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
    
        for (pc, instruction, _) in exe.program.enumerate_by_pc() {
            asm_str += &format!("   .long asm_execute_pc_{} - map_pc_base\n", pc);
        }
    
        return asm_str;
    }
    

    /// Creates a new instance for pure execution
    pub fn new<E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
    ) -> Result<Self, StaticProgramError>
    where
        E: Executor<F>,
    {
        let default_name = String::from("asm_x86_run");
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

        fs::write(format!("{}/src/{}.s", src_asm_bridge_dir_str, asm_name), Self::create_asm(&exe));

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
        &mut self,
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
        &mut self,
        from_state: VmState<F, GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F, GuestMemory>, ExecutionError> {
        let from_state_instret = from_state.instret();
        let from_state_pc = from_state.pc();
        let ctx = ExecutionCtx::new(num_insns);

        let mut vm_exec_state: Box<VmExecState<F, GuestMemory, ExecutionCtx>> =
            Box::new(VmExecState::new(from_state, ctx));

        let instret_end = vm_exec_state.ctx.instret_end;

        println!("what is instret_end {}", instret_end);

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
                from_state_instret,
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
        &mut self,
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
