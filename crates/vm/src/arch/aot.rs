use openvm_instructions::exe::VmExe;
use crate::arch::VmExecState;
use crate::arch::VmState; 
use openvm_instructions::program::Program;
use openvm_instructions::instruction::Instruction;
use openvm_rv32im_transpiler::{
    BaseAluOpcode,
    BranchEqualOpcode
};
use openvm_instructions::LocalOpcode;
use crate::arch::MemoryConfig;
use p3_baby_bear::BabyBear;
use openvm_stark_backend::p3_field::PrimeField32;
use crate::arch::SystemConfig;
use crate::system::memory::online::GuestMemory;
use openvm_instructions::exe::SparseMemoryImage;
use crate::arch::execution_mode::ExecutionCtx;
use crate::arch::interpreter::{get_pre_compute_max_size, alloc_pre_compute_buf, split_pre_compute_buf, get_pre_compute_instructions, get_metered_pre_compute_max_size, get_metered_pre_compute_instructions};
use crate::arch::ExecutorInventory;
use crate::arch::VmExecutor;
use crate::derive::VmConfig;
use strum::{EnumCount, EnumIter, FromRepr, IntoEnumIterator};
use std::process::Command;
use memmap2::MmapOptions;
use crate::arch::StaticProgramError;
use crate::arch::interpreter::AlignedBuf;
use std::fs;
use memmap2::MmapMut;
use crate::arch::Streams;
use std::ffi::c_void;
use libloading::{Library, Symbol};
use crate::arch::{
    interpreter::PreComputeInstruction,
    ExecutionError,
};
use crate::arch::Executor;

use openvm_stark_backend::config::Val;
use crate::arch::ExecutionCtxTrait;
use crate::arch::InterpretedInstance;
use crate::arch::execution_mode::MeteredCtx;
use crate::arch::execution_mode::Segment;
use crate::arch::instructions::SystemOpcode::TERMINATE;
use crate::arch::MeteredExecutor;
use crate::arch::MeteredExecutionCtxTrait;

pub struct AotInstance<'a, F, Ctx> {
    init_memory: SparseMemoryImage,
    system_config: SystemConfig,
    pre_compute_buf: AlignedBuf,
    lib: Library,
    table_box: Box<[PreComputeInstruction<'a, F, Ctx>]>
}

use std::sync::Mutex;
use std::thread;

type AsmRunFn = unsafe extern "C" fn(
    exec_state: *mut core::ffi::c_void, 
    vec_ptr: *const c_void,
);

const PURE_EXECUTION : u32 = 0;
const METERED_EXECUTION : u32 = 1;

pub fn create_assembly<F>(exe: &VmExe<F>, execution_mode: u32)
where F: p3_field::Field
 {
    // save assembly to asm_bridge/src/asm_run.s
    let mut asm = String::new();
    
    asm += ".intel_syntax noprefix\n";
    asm += ".code64\n";
    asm += ".section .text\n";
    asm += ".extern extern_handler\n";
    asm += ".global asm_run_internal\n";
    asm += "\n";

    asm += "asm_run_internal:\n";
    asm += "    push rbp\n";
    asm += "    push rbx\n";
    asm += "    push r12\n";
    asm += "    push r13\n";
    asm += "    push r14\n";
    
    asm += "    mov rbx, rdi\n";
    asm += "    mov rbp, rsi\n";
    asm += "    xor r13, r13\n";
    asm += "    xor r14, r14\n";
    asm += "    lea r10, [rip + map_pc_base]\n";
    asm += "    lea r12, [rip + map_pc_end]\n";
    asm += "    sub r12, r10\n";
    asm += "    shr r12, 2\n";

    for (pc, instruction, _) in exe.program.enumerate_by_pc() {
        asm += &format!("pc_{:x}:\n", pc);

            asm += "    mov rdi, rbx\n";
            asm += "    mov rsi, rbp\n";
            asm += "    mov rdx, r13\n";
            
            if execution_mode == METERED_EXECUTION {
                asm += "    call metered_extern_handler\n";
            } else {
                asm += "    call extern_handler\n";
            }
            asm += "    add r14, 1\n";
            asm += "    cmp rax, 1\n";
            asm += "    je asm_run_end\n";
            
            asm += "    mov r13, rax\n";

            asm += "    shr eax, 2\n";
            asm += "    cmp rax, r12\n";
            asm += "    jae asm_run_end\n";

            asm += "    lea r10, [rip + map_pc_base]\n";
            asm += "    movsxd  r11, dword ptr [r10 + rax*4]\n";
            asm += "    add r11, r10\n";
            
            asm += "    jmp r11\n";
            asm += "\n";
    }

    asm += "asm_run_end:\n";
    asm += "    mov rdi, rbx\n";
    asm += "    mov rsi, r14\n";
    asm += "    mov rdx, r13\n";
    asm += "    call set_instret_and_pc\n";
    asm += "    xor eax, eax\n";
    
    asm += "    pop r14\n";
    asm += "    pop r13\n";
    asm += "    pop r12\n";
    asm += "    pop rbx\n";
    asm += "    pop rbp\n";
    asm += "    ret\n";

    asm += ".section .rodata,\"a\",@progbits\n";
    asm += ".p2align 4\n";
    asm += "map_pc_base:\n";

    for (pc, instruction, _) in exe.program.enumerate_by_pc() {
        asm += &format!("    .long (pc_{:x} - map_pc_base)\n", pc);
    }
    asm += "map_pc_end:\n";
    asm += "\n";

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let asm_file_path = std::path::Path::new(manifest_dir).join("src/arch/asm_bridge/src/asm_run.s");
    fs::write(asm_file_path, asm).expect("Failed to write file");
}

// AotInstance only works for F = BabyBear
impl<'a, F, Ctx> AotInstance<'a, F, Ctx>
where 
    F: PrimeField32,
    Ctx: ExecutionCtxTrait
{
    pub fn new<E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
    ) -> Result<Self, StaticProgramError> 
    where
        E: Executor<F>,
    {
        create_assembly(exe, PURE_EXECUTION);

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let lib_path = std::path::Path::new(manifest_dir)
            .parent().unwrap()
            .parent().unwrap()
            .join("target/release/libasm_bridge.so");
        let asm_bridge_dir = std::path::Path::new(manifest_dir).join("src/arch/asm_bridge");
        let status = Command::new("cargo")
            .args(&["build", "--release"])
            .current_dir(&asm_bridge_dir)
            .status()
            .expect("Failed to execute cargo");
        assert!(status.success(), "Cargo build failed with exit code: {:?}", status.code());

        let lib = unsafe {
            Library::new(&lib_path).expect("Failed to load library")
        };

        let program = &exe.program; 
        let pre_compute_max_size = get_pre_compute_max_size(program, inventory);
        let mut pre_compute_buf = alloc_pre_compute_buf(program, pre_compute_max_size);
        let mut split_pre_compute_buf = split_pre_compute_buf(program, &mut pre_compute_buf, pre_compute_max_size);
        let pre_compute_insns = get_pre_compute_instructions::<F, Ctx, E>(
            program,
            inventory,
            &mut split_pre_compute_buf,
        )?;

        let init_memory = exe.init_memory.clone();

        let table_box : Box<[PreComputeInstruction<'a, F, Ctx>]> = pre_compute_insns.into_boxed_slice();
        let buf_ptr = table_box.as_ptr();
        let box_handle_addr = &table_box as *const _;

        /*
        eprintln!(
            "tid={:?} buf={:p} len={} (box_handle_on_stack={:p}) vm_exec_state={:p}",
            std::thread::current().id(),
            buf_ptr,
            table_box.len()
        );
        */

        Ok(Self {
            pre_compute_buf: pre_compute_buf,
            system_config: inventory.config().clone(),
            init_memory: init_memory,
            lib: lib,
            table_box: table_box,
        })
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
            0,
            inputs,
        );
        self.execute_from_state(vm_state, num_insns)
    }

    pub fn execute_from_state(
        &self,
        from_state: VmState<F, GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F, GuestMemory>, ExecutionError> {
        /*
        let len = std::mem::size_of::<VmExecState<F, GuestMemory, ExecutionCtx>>();
        let mut mmap = unsafe {
            MmapOptions::new().len(len).map_anon().expect("mmap")
        };  
        */

        let ctx = ExecutionCtx::new(num_insns);
        let mut vm_exec_state: Box<VmExecState<F, GuestMemory, ExecutionCtx>> = Box::new(VmExecState::new(from_state, ctx));

        /*
        unsafe {
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            std::ptr::write(ptr, vm_exec_state);
        }  
        */

        unsafe {
            let asm_run: libloading::Symbol<AsmRunFn> = self.lib
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");
            /*
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            */

            let state_ptr = &mut *vm_exec_state as *mut VmExecState<F, GuestMemory, ExecutionCtx>;

            /*
            eprintln!(
                "tid={:?} first_arg={:p} second_arg={:p}",
                std::thread::current().id(),
                state_ptr as *mut c_void,
                (&self.table_box).as_ptr() as *const c_void
            );
            */

            asm_run(state_ptr as *mut c_void, (&self.table_box).as_ptr() as *const c_void);
        }

        Ok((*vm_exec_state).vm_state)
    }
}

impl<'a, F, Ctx> AotInstance<'a, F, Ctx>
where 
    F: PrimeField32,
    Ctx: MeteredExecutionCtxTrait,
{
    pub fn new_metered<E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<Self, StaticProgramError>
    where
        E: MeteredExecutor<F>,
    {
        create_assembly(exe, METERED_EXECUTION);

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let lib_path = std::path::Path::new(manifest_dir)
            .parent().unwrap()
            .parent().unwrap()
            .join("target/release/libasm_bridge.so");
        let asm_bridge_dir = std::path::Path::new(manifest_dir).join("src/arch/asm_bridge");
        let status = Command::new("cargo")
            .args(&["build", "--release"])
            .current_dir(&asm_bridge_dir)
            .status()
            .expect("Failed to execute cargo");
        assert!(status.success(), "Cargo build failed with exit code: {:?}", status.code());

        let lib = unsafe {
            Library::new(&lib_path).expect("Failed to load library")
        };

        let program = &exe.program; 
        let pre_compute_max_size = get_metered_pre_compute_max_size(program, inventory);
        let mut pre_compute_buf = alloc_pre_compute_buf(program, pre_compute_max_size);
        let mut split_pre_compute_buf = split_pre_compute_buf(program, &mut pre_compute_buf, pre_compute_max_size);
        let pre_compute_insns = get_metered_pre_compute_instructions::<F, Ctx, E>(
            program,
            inventory,
            executor_idx_to_air_idx,
            &mut split_pre_compute_buf,
        )?;
        let init_memory = exe.init_memory.clone();

        let table_box : Box<[PreComputeInstruction<'a, F, Ctx>]> = pre_compute_insns.into_boxed_slice(); // todo: maybe rename this
        let buf_ptr = table_box.as_ptr();
        let box_handle_addr = &table_box as *const _;

        Ok(Self {
            pre_compute_buf: pre_compute_buf,
            system_config: inventory.config().clone(),
            init_memory: init_memory,
            lib: lib,
            table_box: table_box
        })
    }
}


impl<F> AotInstance<'_, F, MeteredCtx>
where 
    F: PrimeField32,
{
    pub fn execute_metered(
        &self,
        inputs: impl Into<Streams<F>>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<F, GuestMemory>), ExecutionError> {
        let vm_state = VmState::initial(
            &self.system_config,
            &self.init_memory,
            0,
            inputs,
        );
        self.execute_metered_from_state(vm_state, ctx)
    }

    pub fn execute_metered_from_state(
        &self,
        from_state: VmState<F, GuestMemory>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<F, GuestMemory>), ExecutionError> {
        /*
        let len = std::mem::size_of::<VmExecState<F, GuestMemory, MeteredCtx>>();
        let mut mmap = unsafe {
            MmapOptions::new().len(len).map_anon().expect("mmap")
        };
        */  

        let mut vm_exec_state: Box<VmExecState<F, GuestMemory, MeteredCtx>> = Box::new(VmExecState::new(from_state, ctx.clone()));
        /*
        unsafe {
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, MeteredCtx>;
            std::ptr::write(ptr, vm_exec_state);
        } 
        */ 

        unsafe {
            /*
            let vec_ptr = &self.pre_compute_insns as *const Vec<_> as *const c_void;
            */

            let asm_run: libloading::Symbol<AsmRunFn> = self.lib
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");
            /*
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, MeteredCtx>;
            */

            let state_ptr = &mut *vm_exec_state as *mut VmExecState<F, GuestMemory, MeteredCtx>;

            asm_run(state_ptr as *mut c_void, (&self.table_box).as_ptr() as *const c_void);
        }

        Ok((ctx.into_segments(), vm_exec_state.vm_state))
    }
}