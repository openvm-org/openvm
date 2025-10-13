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
use crate::arch::interpreter::{get_pre_compute_max_size, alloc_pre_compute_buf, split_pre_compute_buf, get_pre_compute_instructions};
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

pub struct AotInstance<'a, F, Ctx> {
    init_memory: SparseMemoryImage,
    system_config: SystemConfig,
    pre_compute_buf: AlignedBuf,
    pre_compute_insns: Vec<PreComputeInstruction<'a, F, Ctx>>
}

type AsmRunFn = unsafe extern "C" fn(
    exec_state: *mut core::ffi::c_void, 
    vec_ptr: *const c_void,
);

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
        Self::create_assembly(exe);

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let asm_bridge_dir = std::path::Path::new(manifest_dir).join("src/arch/asm_bridge");
        
        let status = Command::new("cargo")
            .args(&["build", "--release"])
            .current_dir(&asm_bridge_dir)
            .status()
            .expect("Failed to execute cargo");

        assert!(status.success(), "Cargo build failed with exit code: {:?}", status.code());

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

        Ok(Self {
            pre_compute_insns: pre_compute_insns,
            pre_compute_buf: pre_compute_buf,
            system_config: inventory.config().clone(),
            init_memory: init_memory
        })
    }   

    pub fn create_assembly(exe: &VmExe<F>) {
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
        
        asm += "    mov rbx, rdi\n";
        asm += "    mov rbp, rsi\n";
        asm += "    xor r13, r13\n";
        asm += "    lea r10, [rip + map_pc_base]\n";
        asm += "    lea r12, [rip + map_pc_end]\n";
        asm += "    sub r12, r10\n";
        asm += "    shr r12, 2\n";

        for (pc, instruction, _) in exe.program.enumerate_by_pc() {
            asm += &format!("pc_{:x}:\n", pc);

            if instruction.opcode == TERMINATE.global_opcode() {
                println!("pc {} is terminate instruction", pc);
            }

            asm += "    mov rdi, rbx\n";
            asm += "    mov rsi, rbp\n";
            asm += "    mov rdx, r13\n";
            asm += "    call extern_handler\n";

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
        asm += "    xor eax, eax\n";
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
        let len = std::mem::size_of::<VmExecState<F, GuestMemory, ExecutionCtx>>();
        let mut mmap = unsafe {
            MmapOptions::new().len(len).map_anon().expect("mmap")
        };  

        let ctx = ExecutionCtx::new(num_insns);
        let mut vm_exec_state: VmExecState<F, GuestMemory, ExecutionCtx> = VmExecState::new(from_state, ctx);

        unsafe {
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            std::ptr::write(ptr, vm_exec_state);
        }  

        unsafe {
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            // asm_bridge is a workspace member, so its target is at workspace root
            let lib_path = std::path::Path::new(manifest_dir)
                .parent().unwrap()  // go up to crates/
                .parent().unwrap()  // go up to workspace root
                .join("target/release/libasm_bridge.so");
            
            let vec_ptr = &self.pre_compute_insns as *const Vec<_> as *const c_void;
            let lib = Library::new(&lib_path)
                .expect("Failed to load library");
            let asm_run: libloading::Symbol<AsmRunFn> = lib
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            
            asm_run(ptr as *mut c_void, vec_ptr);
        }

        let vm_exec_state_ref = unsafe {
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            &mut *ptr
        };

        let vm_state = vm_exec_state_ref.vm_state.clone();

        unsafe {
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            std::ptr::drop_in_place(ptr);
        }

        Ok(vm_state)
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
        let len = std::mem::size_of::<VmExecState<F, GuestMemory, MeteredCtx>>();
        let mut mmap = unsafe {
            MmapOptions::new().len(len).map_anon().expect("mmap")
        };  

        let mut vm_exec_state: VmExecState<F, GuestMemory, MeteredCtx> = VmExecState::new(from_state, ctx);

        unsafe {
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, MeteredCtx>;
            std::ptr::write(ptr, vm_exec_state);
        }  

        unsafe {
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            // asm_bridge is a workspace member, so its target is at workspace root
            let lib_path = std::path::Path::new(manifest_dir)
                .parent().unwrap()  // go up to crates/
                .parent().unwrap()  // go up to workspace root
                .join("target/release/libasm_bridge.so");
            
            let vec_ptr = &self.pre_compute_insns as *const Vec<_> as *const c_void;
            let lib = Library::new(&lib_path)
                .expect("Failed to load library");
            let asm_run: libloading::Symbol<AsmRunFn> = lib
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, MeteredCtx>;
            
            asm_run(ptr as *mut c_void, vec_ptr);
        }

        let vm_exec_state_ref = unsafe {
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, MeteredCtx>;
            &mut *ptr
        };

        let vm_exec_state = vm_exec_state_ref.try_clone().unwrap();

        unsafe {
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, MeteredCtx>;
            std::ptr::drop_in_place(ptr);
        }

        let VmExecState { vm_state, ctx, .. } = vm_exec_state;
        Ok((ctx.into_segments(), vm_state))
    }
}
