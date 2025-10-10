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
use openvm_circuit::arch::interpreter::PreComputeInstruction;

use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine};

use openvm_rv32im_circuit::{
    Rv32I, Rv32Io, Rv32M
};

use openvm_stark_backend::config::Val;

type F = BabyBear;
type Ctx = ExecutionCtx;
type Executor = Rv32IExecutor;

pub struct AotInstance<'a> {
    pub mmap: MmapMut,
    pre_compute_buf: AlignedBuf,
    pre_compute_insns: Vec<PreComputeInstruction<'a, F, Ctx>>
}

type AsmRunFn = unsafe extern "C" fn(
    exec_state: *mut core::ffi::c_void, 
    vec_ptr: *const c_void,
);

impl<'a> AotInstance<'a> {
    pub fn new(
        system_config: SystemConfig,
        inventory: &'a ExecutorInventory<Executor>,
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
        let mut split_pre_compute_buf = split_pre_compute_buf(program, &mut pre_compute_buf, pre_compute_max_size);

        let pre_compute_insns = get_pre_compute_instructions::<F, Ctx, Executor>(
            program,
            inventory,
            &mut split_pre_compute_buf,
        )?;

        let mut instance = Self {
            mmap: mmap,
            pre_compute_insns: pre_compute_insns,
            pre_compute_buf: pre_compute_buf,
        };

        Ok(instance)
    }   
    
    pub fn execute(
        &mut self
    ) -> Result<(), libloading::Error> {
        unsafe {
            let fi = &self.pre_compute_insns[0];
            let vec_ptr = &self.pre_compute_insns as *const Vec<_> as *const c_void;
            let lib = Library::new("/home/ubuntu/openvm-test/target/release/libasm_bridge.so")
                .expect("Failed to load library");
            let asm_run: libloading::Symbol<AsmRunFn> = lib.get(b"asm_run")?;
            let ptr = self.mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            
            asm_run(ptr as *mut c_void, vec_ptr);
        }
        Ok(())
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

        fs::write("asm_bridge/src/asm_run.s", asm).expect("Failed to write file");
    }
}

impl<'a> Drop for AotInstance<'a> {
    fn drop(&mut self) {
        unsafe {
            let ptr = self.mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            std::ptr::drop_in_place(ptr);
        }
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
        ), 
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
    aot_instance.execute();

    let ptr = aot_instance.mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>;

    let vm_exec_state_ref = unsafe {
        &mut *(ptr)
    };

    Ok(())
} 