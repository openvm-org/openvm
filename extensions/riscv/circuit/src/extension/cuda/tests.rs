use openvm_circuit::{
    arch::{
        rvr::{cuda::GpuRvrProgram, RvrPreflightLimits},
        GenerationError, VirtualMachine, VmExecutor,
    },
    utils::{test_gpu_engine, test_system_config},
};
use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::Program,
    riscv::{RV64_IMM_AS, RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, SystemOpcode,
};
use openvm_riscv_transpiler::{
    BaseAluImmOpcode, BaseAluOpcode, BaseAluWImmOpcode, BaseAluWOpcode, BranchEqualOpcode,
    BranchLessThanOpcode, LessThanImmOpcode, LessThanOpcode, Rv64AuipcOpcode, Rv64JalLuiOpcode,
    Rv64JalrOpcode, Rv64LoadStoreOpcode, ShiftImmOpcode, ShiftOpcode, ShiftWImmOpcode,
    ShiftWOpcode,
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
fn rvr_gpu_tracegen_proves_system_and_rv64i_airs_without_record_arenas() {
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
        instruction(BaseAluOpcode::XOR, register_operands(22, 3, 4)),
        instruction(BaseAluOpcode::OR, register_operands(23, 3, 4)),
        instruction(BaseAluOpcode::AND, register_operands(24, 3, 4)),
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
        instruction(ShiftOpcode::SLL, register_operands(25, 1, 2)),
        instruction(ShiftOpcode::SRL, register_operands(26, 1, 2)),
        instruction(ShiftOpcode::SRA, register_operands(27, 1, 2)),
        instruction(ShiftWOpcode::SLLW, register_operands(28, 1, 2)),
        instruction(ShiftWOpcode::SRLW, register_operands(29, 1, 2)),
        instruction(ShiftWOpcode::SRAW, register_operands(30, 1, 2)),
        instruction(ShiftWImmOpcode::SLLIW, immediate_operands(17, 8, 1)),
        instruction(ShiftWImmOpcode::SRLIW, immediate_operands(18, 17, 1)),
        instruction(ShiftWImmOpcode::SRAIW, immediate_operands(19, 17, 1)),
        instruction(BaseAluImmOpcode::ORI, immediate_operands(20, 4, 2)),
        instruction(BaseAluImmOpcode::ANDI, immediate_operands(21, 20, 7)),
        Instruction::<F>::from_isize(
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
        Instruction::from_isize(
            BranchLessThanOpcode::BLT.global_opcode(),
            reg(1) as isize,
            reg(2) as isize,
            4,
            RV64_REGISTER_AS as isize,
            RV64_REGISTER_AS as isize,
        ),
        Instruction::from_isize(
            BranchLessThanOpcode::BLTU.global_opcode(),
            reg(2) as isize,
            reg(1) as isize,
            4,
            RV64_REGISTER_AS as isize,
            RV64_REGISTER_AS as isize,
        ),
        Instruction::from_isize(
            BranchLessThanOpcode::BGE.global_opcode(),
            reg(1) as isize,
            reg(1) as isize,
            4,
            RV64_REGISTER_AS as isize,
            RV64_REGISTER_AS as isize,
        ),
        Instruction::from_isize(
            BranchLessThanOpcode::BGEU.global_opcode(),
            reg(0) as isize,
            reg(0) as isize,
            4,
            RV64_REGISTER_AS as isize,
            RV64_REGISTER_AS as isize,
        ),
        Instruction::from_usize(
            Rv64JalLuiOpcode::LUI.global_opcode(),
            [reg(31), 0, 0x80000, RV64_REGISTER_AS as usize, 0, 1],
        ),
        Instruction::from_usize(
            Rv64JalLuiOpcode::JAL.global_opcode(),
            [reg(31), 0, 4, RV64_REGISTER_AS as usize, 0, 1],
        ),
        Instruction::from_usize(
            Rv64JalLuiOpcode::JAL.global_opcode(),
            [0, 0, 4, RV64_REGISTER_AS as usize, 0, 0],
        ),
        Instruction::from_usize(
            Rv64AuipcOpcode::AUIPC.global_opcode(),
            [reg(29), 0, 1, RV64_REGISTER_AS as usize, 0],
        ),
        Instruction::from_usize(
            Rv64LoadStoreOpcode::LOADB.global_opcode(),
            [
                reg(28),
                reg(1),
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            Rv64LoadStoreOpcode::LOADBU.global_opcode(),
            [
                reg(29),
                reg(1),
                1,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            Rv64JalrOpcode::JALR.global_opcode(),
            [reg(30), 0, 164, RV64_REGISTER_AS as usize, 0, 1, 0],
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let mut init_memory = [(1usize, 3u64), (2, 4u64)]
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
        .collect::<openvm_instructions::exe::SparseMemoryImage>();
    init_memory.insert((RV64_MEMORY_AS, 3), 0x80);
    init_memory.insert((RV64_MEMORY_AS, 4), 0xfe);
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };

    let executor = VmExecutor::new(config.clone()).unwrap();
    let rvr = executor.rvr_preflight_instance(&exe, None).unwrap();
    let state = rvr.create_initial_vm_state(Vec::<Vec<u8>>::new());

    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64IGpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    // The system memory trace starts from the segment's pre-mutation image.
    // Upload it before RVR consumes the host state and produces the final image.
    vm.transport_init_memory_to_device(&state.memory);
    let rvr_execution = rvr
        .execute_from_state(state, RvrPreflightLimits::new(64, 192))
        .unwrap();

    let device_ctx = &vm.engine.device().device_ctx;
    let gpu_program =
        GpuRvrProgram::upload(&program, &config.system.memory_config, device_ctx).unwrap();
    let (gpu_transcript, replay_plan) = gpu_program
        .upload_transcript(&rvr_execution.transcript, rvr_execution.endpoint)
        .unwrap();
    let mut tracegen =
        Rv64IRvrGpuTracegen::new(&gpu_program, &gpu_transcript, &replay_plan).unwrap();
    let proving_ctx = vm
        .generate_proving_ctx_from_rvr(
            &gpu_program,
            &gpu_transcript,
            &replay_plan,
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
        Instruction::<F>::from_usize(
            Rv64LoadStoreOpcode::LOADH.global_opcode(),
            [
                reg(1),
                0,
                0,
                RV64_REGISTER_AS as usize,
                openvm_instructions::riscv::RV64_MEMORY_AS as usize,
                1,
                0,
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
        Ok(_) => panic!("executed LOADH must not reach tracegen before its replay port"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("does not support executed opcode"),
        "unexpected coverage error: {error}"
    );
}
