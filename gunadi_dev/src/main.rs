use openvm_instructions::exe::VmExe;
use openvm_circuit::arch::VmExecState;
use openvm_circuit::arch::VmState; 
use openvm_instructions::program::Program;
use openvm_instructions::instruction::Instruction;
use openvm_rv32im_transpiler::{
    BaseAluOpcode,
    BranchEqualOpcode
};
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
use openvm_rv32im_circuit::{
    Rv32IExecutor,
    Rv32BranchEqualExecutor
};
use openvm_circuit::derive::VmConfig;
use openvm_rv32im_circuit::Rv32IConfig;
use openvm_rv32im_circuit::Rv32BaseAluExecutor;
use openvm_rv32im_circuit::adapters::{
    Rv32BaseAluAdapterExecutor,
    Rv32BranchAdapterExecutor
};
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
use openvm_circuit::arch::{
    interpreter::PreComputeInstruction,
    ExecutionError,
};
use openvm_circuit::arch::Executor;

use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine};

use openvm_rv32im_circuit::{
    Rv32I, Rv32Io, Rv32M
};

use openvm_stark_backend::config::Val;

use openvm_circuit::arch::ExecutionCtxTrait;

type F = BabyBear;

pub struct AotInstance<'a, Ctx> {
    init_memory: SparseMemoryImage,
    system_config: SystemConfig,
    pre_compute_buf: AlignedBuf,
    pre_compute_insns: Vec<PreComputeInstruction<'a, F, Ctx>>
}

type AsmRunFn = unsafe extern "C" fn(
    exec_state: *mut core::ffi::c_void, 
    vec_ptr: *const c_void,
);

impl<'a, Ctx> AotInstance<'a, Ctx>
where 
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

        let status = Command::new("cargo")
            .args(&["build", "--release"])
            .current_dir("asm_bridge")
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
    
    pub fn execute(
        &mut self,
        inputs: impl Into<Streams<F>>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F, GuestMemory>, ExecutionError> {
        let len = std::mem::size_of::<VmExecState<F, GuestMemory, ExecutionCtx>>();
        let mut mmap = unsafe {
            MmapOptions::new().len(len).map_anon().expect("mmap")
        };  

        let vm_state = VmState::initial(
            &self.system_config,
            &self.init_memory,
            0,
            inputs,
        );
        let ctx = ExecutionCtx::new(None);
        let mut vm_exec_state: VmExecState<F, GuestMemory, ExecutionCtx> = VmExecState::new(vm_state, ctx);

        unsafe {
            let ptr = mmap.as_mut_ptr() as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            std::ptr::write(ptr, vm_exec_state);
        }  

        unsafe {
            let vec_ptr = &self.pre_compute_insns as *const Vec<_> as *const c_void;
            let lib = Library::new("/home/ubuntu/openvm-test/target/release/libasm_bridge.so")
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let program = Program::<F>::from_instructions(&[
        Instruction::from_isize(
            BaseAluOpcode::ADD.global_opcode(),
            1,
            1,
            3,
            1,
            0,
        ), 
        Instruction::from_isize(
            BaseAluOpcode::SUB.global_opcode(),
            1,
            1,
            1,
            1,
            0,
        ),
        Instruction::from_isize(
            BranchEqualOpcode::BNE.global_opcode(),
            1,
            2,
            4,
            1,
            0,
        ),
        Instruction::from_isize(
            BaseAluOpcode::ADD.global_opcode(),
            3,
            3,
            2,
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

    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let memory_config : MemoryConfig = Default::default();
    let system_config = SystemConfig::default_from_memory(memory_config);
    let mut inventory : ExecutorInventory<Rv32IExecutor> = ExecutorInventory::new(system_config.clone());

    let base_alu = Rv32BaseAluExecutor::new(Rv32BaseAluAdapterExecutor, BaseAluOpcode::CLASS_OFFSET);
    inventory.add_executor(base_alu, BaseAluOpcode::iter().map(|x| x.global_opcode()))?;

    let branch_equal = Rv32BranchEqualExecutor::new(Rv32BranchAdapterExecutor,BranchEqualOpcode::CLASS_OFFSET, 4);
    inventory.add_executor(branch_equal, BranchEqualOpcode::iter().map(|x| x.global_opcode()))?;

    let mut aot_instance = AotInstance::<ExecutionCtx>::new(&inventory, &exe)?;
    aot_instance.execute(vec![], None);

    Ok(())
} 