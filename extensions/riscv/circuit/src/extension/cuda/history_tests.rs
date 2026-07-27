use openvm_circuit::{
    arch::{cuda::postflight::GpuPostflightProgram, VirtualMachine},
    utils::{test_gpu_engine, test_system_config},
};
use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::Program,
    riscv::{RV64_IMM_AS, RV64_REGISTER_AS},
    LocalOpcode, SystemOpcode,
};
use openvm_riscv_transpiler::BaseAluImmOpcode;
use openvm_stark_backend::StarkEngine;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::Rv64ImPreflightGpuTracegen;
use crate::{adapters::RV64_REGISTER_NUM_LIMBS, Rv64IConfig, Rv64IGpuBuilder};

fn register(index: usize) -> usize {
    index * RV64_REGISTER_NUM_LIMBS
}

fn addi(rd: usize, rs1: usize, immediate: usize) -> Instruction<BabyBear> {
    Instruction::from_usize(
        BaseAluImmOpcode::ADDI.global_opcode(),
        [
            register(rd),
            register(rs1),
            immediate,
            RV64_REGISTER_AS as usize,
            RV64_IMM_AS as usize,
        ],
    )
}

#[test]
fn interpreter_history_proves_system_and_rv64_traces() {
    let instructions = [
        addi(1, 0, 7),
        addi(2, 1, 1),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
    ];
    let program = Program::from_instructions(&instructions);
    let exe = VmExe::new(program.clone());
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64IGpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);

    let interpreter = vm.preflight_interpreter(&exe).unwrap();
    let state = vm.create_initial_state(&exe, Vec::<Vec<u8>>::new());
    vm.transport_init_memory_to_device(&state.memory);
    let output = vm.execute_preflight(&interpreter, state).unwrap();

    let device_ctx = &vm.engine.device().device_ctx;
    let gpu_program =
        GpuPostflightProgram::upload(&program, &config.system.memory_config, device_ctx).unwrap();
    let (transcript, replay_plan) = vm.postflight_history(&gpu_program, &output).unwrap();
    let tracegen =
        Rv64ImPreflightGpuTracegen::new(&gpu_program, &transcript, &replay_plan).unwrap();
    let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
    drop(replay_plan);
    drop(transcript);

    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();
}
