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

use libc::{
    shm_open, 
    shm_unlink, 
    mmap, 
    munmap, 
    ftruncate, 
    close,
    O_CREAT, 
    O_RDWR, 
    O_RDONLY, 
    PROT_READ, 
    PROT_WRITE, 
    MAP_SHARED,
    MAP_FAILED
};

use std::ffi::CString;

pub struct AotInstance<F: PrimeField32> {
    exe: VmExe<F>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MemoryUpdate {
    address_space: u32, 
    pointer: u32, 
    value: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryLog {
    count: usize,
    updates: [MemoryUpdate; 10],
}

impl<F: PrimeField32> AotInstance<F> {
    /* 
    compile with 
    as aot_asm.s -o aot_asm.o
    gcc -no-pie aot_asm.o -L. -lopenvm_circuit_rust_bridge -o program
    */
    pub fn new(exe: &VmExe<F>) -> Self {
        /* Write out the assembly

        let _ = File::create("aot_asm.s").expect("Unable to create file");
        std::fs::write("aot_asm.s", &asm_string).expect("Unable to write file");
        */
        let output = Command::new("as")
            .args(["aot_asm.s", "-o", "aot_asm.o"])
            .output();

        let output = Command::new("gcc")
            .args([
                "-no-pie",
                "aot_asm.o",
                "-L.",
                "-lopenvm_circuit_rust_bridge",
                "-o",
                "program",
            ])
            .output();

        Self { 
            exe: exe.clone()
        }
    }

    pub fn generate_asm() {

    }

    pub fn execute(&self) -> VmExecState<F, GuestMemory, ExecutionCtx> {
        // parent process set-up the shared memory
        let c_name = CString::new("/shmem").unwrap();

        /* 
        initialize vm state and execution context
        goal is to create a VmExecState<F>
        */

        let memory_config = Default::default();
        let system_config = SystemConfig::default_from_memory(memory_config);
        let init_memory = Default::default();
        let vm_state: VmState<F> = VmState::initial(&system_config, &init_memory, 0, vec![]);
        let exec_ctx = ExecutionCtx::new(None); // Pure execution context
        let mut vm_exec_state: VmExecState<F, GuestMemory, ExecutionCtx> = VmExecState::new(vm_state, exec_ctx);
        
        unsafe {
            let size = std::mem::size_of::<MemoryLog>();
            let fd = shm_open(c_name.as_ptr(), O_CREAT | O_RDWR, 0o666);
            ftruncate(fd, size as i64);

            let ptr = mmap(
                std::ptr::null_mut(), 
                size, 
                PROT_READ | PROT_WRITE, 
                MAP_SHARED, 
                fd, 
                0
            );

            let log_ptr = ptr as *mut MemoryLog;
            (*log_ptr).count = 0;

            munmap(ptr, size);
            close(fd);
        }

        unsafe {
            let _ = Command::new("./program").status();
        }

        unsafe {
            let size = std::mem::size_of::<MemoryLog>();
    
            let fd = shm_open(c_name.as_ptr(), O_CREAT | O_RDWR, 0o666);
            ftruncate(fd, size as i64);
    
            let ptr = mmap(
                std::ptr::null_mut(), 
                size, 
                PROT_READ | PROT_WRITE, 
                MAP_SHARED, 
                fd, 
                0
            );
    
            let log_ptr = ptr as *mut MemoryLog;

            /*
            Do conversion from MemoryLog to VmState
            */

            let count = (*log_ptr).count;
            for i in 0..count {
                let update = (*log_ptr).updates[i];
                let data : &[u8; 1] = &[update.value as u8];
                println!("update: {:?}", update);
                vm_exec_state.vm_write::<u8, 1>(update.address_space, update.pointer, data);
            }
    
            munmap(ptr, size);
            close(fd);
        }

        return vm_exec_state;
    }

    pub fn clean_up(&self) {
        unsafe {
            let c_name = CString::new("/shmem").unwrap();
            shm_unlink(c_name.as_ptr());
        }
    }
    
}
