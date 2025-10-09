use crate::arch::instructions::VmOpcode;
use crate::{
    arch::{SystemConfig, VmState},
    system::memory::online::GuestMemory,
};
use libloading::{Library, Symbol};
use openvm_instructions::instruction;
use openvm_instructions::LocalOpcode;
use openvm_instructions::{exe::VmExe, instruction::Instruction};
use openvm_rv32im_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, Rv32LoadStoreOpcode,
};
use openvm_stark_backend::p3_field::FieldAlgebra;
use p3_baby_bear::{BabyBear, BabyBearParameters};
use p3_field::PrimeField32;
use std::fs::File;
use std::io::Write;
use std::pin::Pin;
use std::{env, env::args, fs, path::PathBuf, process::Command};
use tracing::subscriber::SetGlobalDefaultError;

use crate::arch::state;
use crate::arch::MemoryConfig;
use crate::arch::execution_mode::ExecutionCtx;
use crate::arch::VmExecState;
use crate::arch::interpreter::{get_pre_compute_max_size, alloc_pre_compute_buf, split_pre_compute_buf, get_pre_compute_instructions};

use std::ffi::c_void;
use memmap2::MmapOptions;
use memmap2::MmapMut;
use crate::arch::Executor;
use crate::arch::ExecutorInventory;
use crate::arch::ExecutionCtxTrait;
use crate::arch::VmExecutor;
use crate::arch::interpreter::PreComputeInstruction;

use std::ffi::CString;
use crate::arch::Streams;

pub struct AotInstance {
    mmap: MmapMut,
}

type F = BabyBear;

impl AotInstance {
    pub fn new<E>(
        exe: &VmExe<F>,
    ) -> Self {
        Self::create_assembly(exe);

        let status = Command::new("cargo")
            .args(&["build", "--release"])
            .current_dir("asm_bridge/")
            .status()
            .expect("Failed to execute cargo");
        
        if !status.success() {
            panic!("Cargo build failed");
        }

        let len = std::mem::size_of::<VmExecState<F, GuestMemory, ExecutionCtx>>();
        let mut mmap = unsafe {
            MmapOptions::new().len(len).map_anon().expect("mmap")
        };  

        // create a vm_exec_state
        let memory_config = Default::default();
        let system_config = SystemConfig::default_from_memory(memory_config);
        let init_memory = Default::default();
        let vm_state: VmState<F> = VmState::initial(&system_config, &init_memory, 0, vec![]);
        let exec_ctx = ExecutionCtx::new(None); 
        let mut vm_exec_state: VmExecState<F, GuestMemory, ExecutionCtx> = VmExecState::new(vm_state, exec_ctx);

        /*
        let inventory = ExecutorInventory::new(system_config);
        */

        unsafe {
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            std::ptr::write(ptr, vm_exec_state);
        }     

        /*
        let program = &exe.program;
        let pre_compute_max_size = get_pre_compute_max_size(program, &inventory);
        let mut pre_compute_buf = alloc_pre_compute_buf(program, pre_compute_max_size);
        let mut split_pre_compute_buf = split_pre_compute_buf(program, &mut pre_compute_buf, pre_compute_max_size);
        */
        
        // let pre_compute_insns = get_pre_compute_instructions::<F, Ctx, E>(
        //     program,
        //     &inventory,
        //     &mut split_pre_compute_buf,
        // )?;

        Self {
            mmap,
        }
    }    
    
    pub fn create_assembly(exe: &VmExe<F>) {
        // save assembly to asm_bridge/src/asm_run.s
        let mut asm = String::new();
        asm += ".intel_syntax noprefix\n";
        asm += ".code64\n";
        asm += ".section .text\n";
        asm += ".extern TEST_FN\n";
        asm += ".extern ADD_RV32\n";
        asm += ".global asm_run_internal\n";
        asm += "\n";

        asm += "asm_run_internal:\n";
        asm += "    mov rbx, rdi\n";
        asm += "    sub rsp, 8\n";
        asm += "    call TEST_FN\n";
        asm += "\n";

        for (pc, instruction, _) in exe.program.enumerate_by_pc() {
            asm += &format!("pc_{:x}:\n", pc);
            let opcode = instruction.opcode; 

            match opcode {
                x if x == BaseAluOpcode::ADD.global_opcode() => {
                    asm += "    mov rdi, rbx\n";
                    asm += &format!("    mov rsi, {}\n", instruction.a);
                    asm += &format!("    mov rdx, {}\n", instruction.b);
                    asm += &format!("    mov rcx, {}\n", instruction.c);
                    asm += &format!("    mov r8, {}\n", instruction.d);
                    asm += &format!("    mov r9, {}\n", instruction.e);
                    asm += &format!("    call ADD_RV32\n");
                }
                x if x == BaseAluOpcode::SUB.global_opcode() => {
                    asm += "    mov rdi, rbx\n";
                    asm += &format!("    mov rsi, {}\n", instruction.a);
                    asm += &format!("    mov rdx, {}\n", instruction.b);
                    asm += &format!("    mov rcx, {}\n", instruction.c);
                    asm += &format!("    mov r8, {}\n", instruction.d);
                    asm += &format!("    mov r9, {}\n", instruction.e);
                    asm += &format!("    call SUB_RV32\n");
                }
                _ => {
                    println!("instruction {pc}'s opcode is not implemented yet");
                }
            }
            asm += "\n";
        }

        asm += "asm_run_end:\n";
        asm += "    add rsp, 8\n";
        asm += "    xor eax, eax\n";
        asm += "    ret\n";

        fs::write("asm_bridge/src/asm_run.s", asm).expect("Failed to write file");
    }

    // for testing
    pub fn default_asm() {
        let mut asm = String::new();
        asm += ".intel_syntax noprefix\n";
        asm += ".code64\n";
        asm += ".section .text\n";
        asm += ".extern TEST_FN\n";
        asm += ".global asm_run_internal\n";
        asm += "\n";
        asm += "asm_run_internal:\n";
        asm += "    push rbp\n";
        asm += "    mov rbp, rsp\n";
        asm += "    call TEST_FN\n";
        asm += "    pop rbp\n";
        asm += "    xor eax, eax\n";
        asm += "    ret\n";

        fs::write("asm_bridge/src/asm_run.s", asm).expect("Failed to write file");
    }

    // execute finds the dynamic library, calls the asm_run function, and returns a reference to the vm_exec_state
    pub fn execute(&mut self, 
        inputs: impl Into<Streams<F>>,
        num_insns: Option<u64>
    ) -> &mut VmExecState<F, GuestMemory, ExecutionCtx> {
        unsafe {
            let lib = Library::new("/home/ubuntu/openvm-test/target/release/libasm_bridge.so").expect("Failed to load library");
            let asm_run: Symbol<unsafe extern "C" fn(*mut c_void)> = 
                lib.get(b"asm_run").expect("Failed to find asm_run symbol");
            
            asm_run(self.mmap.as_mut_ptr() as *mut c_void);
        }
        
        let vm_exec_state_ref = unsafe {
            &mut *(self.mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>)
        };

        vm_exec_state_ref
    }
}
