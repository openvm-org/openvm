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
use openvm_circuit::arch::interpreter::get_pre_compute_max_size;
use openvm_circuit::arch::ExecutorInventory;
use openvm_circuit::arch::VmExecutor;
use openvm_rv32im_circuit::Rv32IExecutor;
use openvm_circuit::derive::VmConfig;
use openvm_rv32im_circuit::Rv32IConfig;


use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine};

use openvm_rv32im_circuit::{
    Rv32I, Rv32Io, Rv32M
};

use openvm_stark_backend::config::Val;

type F = BabyBear;
type Executor = Rv32IExecutor;

pub struct AotInstance {

}

impl AotInstance {
    pub fn new(
        inventory: &ExecutorInventory<Executor>,
        exe: &VmExe<F>,
    ) {
        let memory_config : MemoryConfig = Default::default();
        let system_config = SystemConfig::default_from_memory(memory_config);
        let init_memory : SparseMemoryImage = Default::default();
        let vm_state: VmState<F> = VmState::initial(&system_config, &init_memory, 0, vec![]);
        let exec_ctx = ExecutionCtx::new(None); 
        let mut vm_exec_state: VmExecState<F, GuestMemory, ExecutionCtx> = VmExecState::new(vm_state, exec_ctx);

        let program = &exe.program; 
        let pre_compute_max_size = get_pre_compute_max_size(program, &inventory);

    }    
}

fn main() {
    let program = Program::<F>::from_instructions(&[
        Instruction::from_isize(
            BaseAluOpcode::ADD.global_opcode(),
            4,
            4,
            5,
            0,
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


    let executor = VmExecutor::<F, _>::new(config);
    executor.instance(exe);    
    
    // let inventory : ExecutorInventory<Executor> = ExecutorInventory::new(system_config);

    // let base_alu =
    //     Rv32BaseAluExecutor::new(Rv32BaseAluAdapterExecutor, BaseAluOpcode::CLASS_OFFSET);

    // inventory.add_executor(base_alu, BaseAluOpcode::iter().map(|x| x.global_opcode()));

    // let aot_instance = AotInstance::new(&inventory, &exe);
} 