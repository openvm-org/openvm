use openvm_instructions::exe::VmExe;
use openvm_circuit::arch::VmExecState;
use openvm_circuit::arch::VmState; 
use openvm_instructions::program::Program;
use openvm_instructions::instruction::Instruction;
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_instructions::LocalOpcode;
use openvm_circuit::arch::MemoryConfig;
use p3_baby_bear::BabyBear;
use openvm_circuit::arch::SystemConfig;
use openvm_circuit::system::memory::online::GuestMemory;
use openvm_instructions::exe::SparseMemoryImage;
use openvm_circuit::arch::execution_mode::ExecutionCtx;
use openvm_circuit::arch::interpreter::{get_pre_compute_max_size, alloc_pre_compute_buf, split_pre_compute_buf, get_pre_compute_instructions};
use openvm_circuit::arch::ExecutorInventory;
use openvm_circuit::arch::VmExecutor;
use openvm_rv32im_circuit::Rv32IExecutor;
use openvm_circuit::derive::VmConfig;
use openvm_rv32im_circuit::Rv32IConfig;
use openvm_rv32im_circuit::Rv32BaseAluExecutor;
use openvm_rv32im_circuit::adapters::Rv32BaseAluAdapterExecutor;
use strum::{EnumCount, EnumIter, FromRepr, IntoEnumIterator};
use std::process::Command;
use memmap2::MmapOptions;
use openvm_circuit::arch::StaticProgramError;
use openvm_circuit::arch::interpreter::AlignedBuf;
use std::fs;
use memmap2::MmapMut;
use openvm_circuit::arch::Streams;
use std::ffi::c_void;
use libloading::{Library, Symbol};

use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine};

use openvm_rv32im_circuit::{
    Rv32I, Rv32Io, Rv32M
};

use openvm_stark_backend::config::Val;

// FFI-safe descriptor for one mutable slice
#[repr(C)]
pub struct SliceMeta {
    pub ptr: *mut u8,
    pub len: usize,
}

type F = BabyBear;
type Ctx = ExecutionCtx;
type Executor = Rv32IExecutor;

pub struct AotInstance {
    mmap: MmapMut,
    // Own the underlying memory so pointers remain valid.
    pre_compute_buf: AlignedBuf,
    // Store stable, FFI-safe metadata (boxed slice == fixed address, no reallocation).
    split_meta: Box<[SliceMeta]>,
}

pub const DEFAULT_PC_STEP: u32 = 4;

pub fn get_pc_index(pc: u32) -> usize {
    (pc / DEFAULT_PC_STEP) as usize
}

impl AotInstance {
    pub fn new(
        system_config: SystemConfig,
        inventory: &ExecutorInventory<Executor>,
        exe: &VmExe<F>,
    ) -> Result<Self, StaticProgramError> {
        Self::create_assembly(exe);

        let status = Command::new("cargo")
            .args(&["build", "--release"])
            .current_dir("asm_bridge")
            .status()
            .expect("Failed to execute cargo");

        if !status.success() {
            panic!("Cargo build failed");
        }

        let len = std::mem::size_of::<VmExecState<F, GuestMemory, ExecutionCtx>>();
        let mut mmap = unsafe {
            MmapOptions::new().len(len).map_anon().expect("mmap")
        };  

        let init_memory : SparseMemoryImage = Default::default();

        let vm_state: VmState<F> = VmState::initial(&system_config, &init_memory, 0, vec![]);

        let exec_ctx = ExecutionCtx::new(None); 

        let mut vm_exec_state: VmExecState<F, GuestMemory, ExecutionCtx> = VmExecState::new(vm_state, exec_ctx);

        unsafe {
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            std::ptr::write(ptr, vm_exec_state);
        }  

        let program = &exe.program; 
        
        let pre_compute_max_size = get_pre_compute_max_size(program, &inventory);
        let mut pre_compute_buf = alloc_pre_compute_buf(program, pre_compute_max_size);
        let mut split_views = split_pre_compute_buf(program, &mut pre_compute_buf, pre_compute_max_size);

        let _pre_compute_insns = get_pre_compute_instructions::<F, Ctx, Executor>(
            program,
            inventory,
            &mut split_views,
        )?;

        let split_meta_vec: Vec<SliceMeta> = split_views
            .iter_mut()
            .map(|s| SliceMeta {
                ptr: s.as_mut_ptr(),
                len: s.len(),
            })
            .collect();
        let split_meta = split_meta_vec.into_boxed_slice();
        Ok(Self {
            mmap,
            pre_compute_buf,
            split_meta
        })
    }    

    /// Pointer you can pass to C/asm. Valid as long as `self` lives and
    /// you don't mutate `split_meta`/`pre_compute_buf` in a way that reallocates.
    pub fn split_meta_ptr(&mut self) -> *mut SliceMeta {
        self.split_meta.as_mut_ptr()
    }
    pub fn split_meta_len(&self) -> usize {
        self.split_meta.len()
    }

    /*
    pub fn new_without_inventory(
        exe: &VmExe<F>,
    ) -> Result<Self, StaticProgramError> {
        Self::create_assembly(exe);

        let status = Command::new("cargo")
            .args(&["build", "--release"])
            .current_dir("asm_bridge")
            .status()
            .expect("Failed to execute cargo");

        if !status.success() {
            panic!("Cargo build failed");
        }

        let len = std::mem::size_of::<VmExecState<F, GuestMemory, ExecutionCtx>>();
        let mut mmap = unsafe {
            MmapOptions::new().len(len).map_anon().expect("mmap")
        };  

        let memory_config = Default::default();
        let system_config = SystemConfig::default_from_memory(memory_config);
        let init_memory = Default::default();
        let exec_ctx = ExecutionCtx::new(None); 
        let vm_state: VmState<F> = VmState::initial(&system_config, &init_memory, 0, vec![]);
        let mut vm_exec_state: VmExecState<F, GuestMemory, ExecutionCtx> = VmExecState::new(vm_state, exec_ctx);

        unsafe {
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            std::ptr::write(ptr, vm_exec_state);
        }  

        let lib = unsafe {
            Library::new("/home/ubuntu/openvm-test/target/release/libasm_bridge.so").expect("Failed to load library")
        };

        Ok(Self {
            mmap: mmap, 
            asm_lib: Some(lib),
        })
    }    
    */

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

    pub fn execute(
        &mut self
    ) {
        unsafe {
            let lib = Library::new("/home/ubuntu/openvm-test/target/release/libasm_bridge.so").expect("Failed to load library");
            let asm_run: Symbol<unsafe extern "C" fn(*mut c_void)> = 
                lib.get(b"asm_run").expect("Failed to find asm_run symbol");
            
            asm_run(self.mmap.as_mut_ptr() as *mut c_void);
        };

        println!("helo it's ok here!");
    }
}

impl Drop for AotInstance {
    fn drop(&mut self) {
        println!("AOT INSTANCE IS DROPPED");
        unsafe {
            // Manually drop the VmExecState before unmapping
            let ptr = self.mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            std::ptr::drop_in_place(ptr);
        }
        // mmap drops automatically after this
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let program = Program::<F>::from_instructions(&[
        Instruction::from_isize(
            BaseAluOpcode::ADD.global_opcode(),
            4,
            4,
            4,
            1,
            0,
        )
    ]);

    let exe = VmExe {
        program, 
        pc_start: 0,
        fn_bounds: Default::default(),
        init_memory: Default::default(),
    };

    let config = Rv32IConfig::default();
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let executor = VmExecutor::<F, _>::new(config)?;
    let _ = executor.instance(&exe);    

    let memory_config : MemoryConfig = Default::default();
    let system_config = SystemConfig::default_from_memory(memory_config);
    let mut inventory : ExecutorInventory<Executor> = ExecutorInventory::new(system_config.clone());

    let base_alu = Rv32BaseAluExecutor::new(Rv32BaseAluAdapterExecutor, BaseAluOpcode::CLASS_OFFSET);
    inventory.add_executor(base_alu, BaseAluOpcode::iter().map(|x| x.global_opcode()))?;

    let mut aot_instance = AotInstance::new(system_config.clone(), &inventory, &exe)?;

    let ptr = aot_instance.split_meta_ptr();
    println!("ptr:{:?}, count: {}", ptr, aot_instance.split_meta_len());
    aot_instance.execute();
    Ok(())
} 