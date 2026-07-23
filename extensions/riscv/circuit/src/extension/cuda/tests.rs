use openvm_circuit::{
    arch::{
        rvr::{cuda::GpuRvrProgram, RvrPreflightLimits},
        Arena, GenerationError, VirtualMachine, VmExecutor,
    },
    utils::{test_gpu_engine, test_system_config},
};
use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::Program,
    riscv::{RV64_IMM_AS, RV64_REGISTER_AS},
    LocalOpcode, SystemOpcode,
};
use openvm_riscv_transpiler::{
    BaseAluImmOpcode, BaseAluOpcode, BaseAluWImmOpcode, BaseAluWOpcode, BranchEqualOpcode,
    LessThanImmOpcode, LessThanOpcode, ShiftImmOpcode, ShiftWImmOpcode,
};
use openvm_stark_backend::StarkEngine;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::Rv64IRvrGpuTracegen;
use crate::{adapters::RV64_REGISTER_NUM_LIMBS, Rv64IConfig, Rv64IGpuBuilder};

type F = BabyBear;

fn reg(index: usize) -> usize {
    index * RV64_REGISTER_NUM_LIMBS
}

fn instruction(opcode: impl LocalOpcode, operands: [usize; 5]) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), operands)
}

#[test]
fn rvr_gpu_tracegen_proves_multiple_rv64i_airs_without_extension_arenas() {
    let register_operands = |rd, rs1, rs2| {
        [
            reg(rd),
            reg(rs1),
            reg(rs2),
            RV64_REGISTER_AS as usize,
            RV64_REGISTER_AS as usize,
        ]
    };
    let immediate_operands = |rd, rs1, imm| {
        [
            reg(rd),
            reg(rs1),
            imm,
            RV64_REGISTER_AS as usize,
            RV64_IMM_AS as usize,
        ]
    };
    let instructions = [
        instruction(BaseAluImmOpcode::ADDI, immediate_operands(3, 1, 7)),
        instruction(BaseAluImmOpcode::XORI, immediate_operands(4, 3, 1)),
        instruction(BaseAluOpcode::ADD, register_operands(5, 3, 4)),
        instruction(BaseAluOpcode::SUB, register_operands(11, 5, 1)),
        instruction(LessThanImmOpcode::SLTI, immediate_operands(6, 5, 100)),
        instruction(LessThanImmOpcode::SLTIU, immediate_operands(14, 1, 4)),
        instruction(ShiftImmOpcode::SLLI, immediate_operands(7, 6, 1)),
        instruction(ShiftImmOpcode::SRLI, immediate_operands(15, 5, 1)),
        instruction(ShiftImmOpcode::SRAI, immediate_operands(16, 5, 1)),
        instruction(BaseAluWImmOpcode::ADDIW, immediate_operands(8, 7, 2)),
        instruction(BaseAluWOpcode::ADDW, register_operands(9, 8, 1)),
        instruction(BaseAluWOpcode::SUBW, register_operands(12, 9, 1)),
        instruction(LessThanOpcode::SLTU, register_operands(10, 1, 5)),
        instruction(LessThanOpcode::SLT, register_operands(13, 1, 2)),
        instruction(ShiftWImmOpcode::SLLIW, immediate_operands(17, 8, 1)),
        instruction(ShiftWImmOpcode::SRLIW, immediate_operands(18, 17, 1)),
        instruction(ShiftWImmOpcode::SRAIW, immediate_operands(19, 17, 1)),
        instruction(BaseAluImmOpcode::ORI, immediate_operands(20, 4, 2)),
        instruction(BaseAluImmOpcode::ANDI, immediate_operands(21, 20, 7)),
        Instruction::from_isize(
            BranchEqualOpcode::BEQ.global_opcode(),
            reg(1) as isize,
            reg(2) as isize,
            4,
            RV64_REGISTER_AS as isize,
            RV64_REGISTER_AS as isize,
        ),
        Instruction::from_isize(
            BranchEqualOpcode::BNE.global_opcode(),
            reg(1) as isize,
            reg(1) as isize,
            4,
            RV64_REGISTER_AS as isize,
            RV64_REGISTER_AS as isize,
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let init_memory = [(1usize, 3u64), (2, 4u64)]
        .into_iter()
        .flat_map(|(register, value)| {
            value
                .to_le_bytes()
                .into_iter()
                .enumerate()
                .map(move |(offset, byte)| {
                    ((RV64_REGISTER_AS, (reg(register) + offset) as u32), byte)
                })
        })
        .collect();
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };

    let executor = VmExecutor::new(config.clone()).unwrap();
    let rvr = executor.rvr_preflight_instance(&exe, None).unwrap();
    let rvr_execution = rvr
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(64, 128))
        .unwrap();

    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64IGpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    let mut interpreter = vm.preflight_interpreter(&exe).unwrap();
    let state = vm.create_initial_state(&exe, Vec::<Vec<u8>>::new());
    vm.transport_init_memory_to_device(&state.memory);
    let output = vm
        .execute_preflight(&mut interpreter, state, &vec![64; vm.pk().per_air.len()])
        .unwrap();
    let num_system_airs = config.system.num_airs();
    let mut record_arenas = output.record_arenas;
    let extension_arenas = record_arenas.split_off(num_system_airs);
    assert!(
        extension_arenas.iter().any(|arena| !arena.is_empty()),
        "the interpreter must have produced extension records that this path discards"
    );
    drop(extension_arenas);

    let device_ctx = &vm.engine.device().device_ctx;
    let gpu_program =
        GpuRvrProgram::upload(&program, &config.system.memory_config, device_ctx).unwrap();
    let (gpu_transcript, replay_plan) = gpu_program
        .upload_transcript(&rvr_execution.transcript, rvr_execution.endpoint)
        .unwrap();
    let mut tracegen =
        Rv64IRvrGpuTracegen::new(&gpu_program, &gpu_transcript, &replay_plan).unwrap();
    let proving_ctx = vm
        .generate_proving_ctx_with_extension_tracegen(
            output.system_records,
            record_arenas,
            |insertion_idx, chip| {
                tracegen
                    .generate_for_chip(insertion_idx, chip)
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))
            },
        )
        .unwrap();
    tracegen.finish().unwrap();

    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();
}

#[test]
fn rvr_gpu_tracegen_rejects_an_executed_unported_opcode_before_tracegen() {
    let instructions = [
        instruction(
            BaseAluOpcode::XOR,
            [
                reg(3),
                reg(1),
                reg(2),
                RV64_REGISTER_AS as usize,
                RV64_REGISTER_AS as usize,
            ],
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let init_memory = [(1usize, 3u64), (2, 4u64)]
        .into_iter()
        .flat_map(|(register, value)| {
            value
                .to_le_bytes()
                .into_iter()
                .enumerate()
                .map(move |(offset, byte)| {
                    ((RV64_REGISTER_AS, (reg(register) + offset) as u32), byte)
                })
        })
        .collect();
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let rvr = executor.rvr_preflight_instance(&exe, None).unwrap();
    let execution = rvr
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(8, 16))
        .unwrap();
    let engine = test_gpu_engine();
    let gpu_program = GpuRvrProgram::upload(
        &program,
        &config.system.memory_config,
        &engine.device().device_ctx,
    )
    .unwrap();
    let (gpu_transcript, replay_plan) = gpu_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();

    let error = match Rv64IRvrGpuTracegen::new(&gpu_program, &gpu_transcript, &replay_plan) {
        Ok(_) => panic!("executed register XOR must not reach tracegen before its replay port"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("does not support executed opcode"),
        "unexpected coverage error: {error}"
    );
}
