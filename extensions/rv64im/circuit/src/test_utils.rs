use openvm_circuit::arch::{
    execution_mode::ExecutionCtx, InterpreterExecutor, VmExecState, VmState,
};
use openvm_circuit::system::memory::online::{AddressMap, GuestMemory};
use openvm_instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS};
use openvm_instructions::{instruction::Instruction, VmOpcode};
use openvm_stark_backend::p3_field::PrimeField32;

#[cfg(feature = "tco")]
use openvm_circuit::arch::{ExecutorInventory, InterpretedInstance, SystemConfig};
#[cfg(feature = "tco")]
use openvm_instructions::{exe::VmExe, program::Program};

use crate::rv64_mem_config;

pub fn create_exec_state<F: PrimeField32>(
    pc_start: u32,
) -> VmExecState<F, GuestMemory, ExecutionCtx> {
    let mem = GuestMemory::new(AddressMap::from_mem_config(&rv64_mem_config()));
    let vm_state = VmState::new_with_defaults(pc_start, mem, vec![] as Vec<Vec<F>>, 0, 0);
    VmExecState::new(vm_state, ExecutionCtx::new(None))
}

pub fn write_reg<F: PrimeField32>(
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    reg: u32,
    val: u64,
) {
    unsafe {
        state
            .memory
            .write::<u8, 8>(RV32_REGISTER_AS, reg, val.to_le_bytes());
    }
}

pub fn read_reg<F: PrimeField32>(
    state: &VmExecState<F, GuestMemory, ExecutionCtx>,
    reg: u32,
) -> u64 {
    let bytes: [u8; 8] = unsafe { state.memory.read::<u8, 8>(RV32_REGISTER_AS, reg) };
    u64::from_le_bytes(bytes)
}

pub const DATA_MEM_AS: u32 = RV32_MEMORY_AS;

pub fn write_mem<F: PrimeField32>(
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    addr: u32,
    val: [u8; 8],
) {
    unsafe {
        state.memory.write::<u8, 8>(DATA_MEM_AS, addr, val);
    }
}

pub fn read_mem<F: PrimeField32>(
    state: &VmExecState<F, GuestMemory, ExecutionCtx>,
    addr: u32,
) -> [u8; 8] {
    unsafe { state.memory.read::<u8, 8>(DATA_MEM_AS, addr) }
}

pub fn execute_instruction<F, E, I>(
    executor: &E,
    opcodes: I,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
    pc_base: u32,
) -> u32
where
    F: PrimeField32,
    E: InterpreterExecutor<F> + Clone,
    I: IntoIterator<Item = VmOpcode>,
{
    let pc = state.pc();
    #[cfg(not(feature = "tco"))]
    {
        let _ = opcodes;
        let _ = pc_base;
        let size = <E as InterpreterExecutor<F>>::pre_compute_size(executor);
        let mut data = vec![0u8; size];
        let func = executor
            .pre_compute::<ExecutionCtx>(pc, inst, &mut data)
            .expect("pre_compute should succeed");
        unsafe { func(data.as_ptr(), state) };
    }
    #[cfg(feature = "tco")]
    {
        let mut inventory =
            ExecutorInventory::<E>::new(SystemConfig::default_from_memory(rv64_mem_config()));
        inventory
            .add_executor(executor.clone(), opcodes)
            .expect("add_executor should succeed");

        let base_idx = (pc_base / openvm_instructions::program::DEFAULT_PC_STEP) as usize;
        let cur_idx = (pc / openvm_instructions::program::DEFAULT_PC_STEP) as usize;
        let (program, exe_pc_base) = if cur_idx < base_idx {
            (Program::new_without_debug_infos(&[inst.clone()], pc), pc)
        } else {
            let len = cur_idx - base_idx + 1;
            let mut insts: Vec<Option<Instruction<F>>> = vec![None; len];
            insts[len - 1] = Some(inst.clone());
            (
                Program::new_without_debug_infos_with_option(&insts, pc_base),
                pc_base,
            )
        };
        let exe = VmExe::new(program).with_pc_start(exe_pc_base);
        let interpreter =
            InterpretedInstance::new(&inventory, &exe).expect("interpreter build must succeed");

        state.ctx.instret_left = 0;
        let handler = interpreter.get_handler(pc).expect("handler should exist");
        unsafe { handler(&interpreter, state) };
    }
    state.pc()
}
