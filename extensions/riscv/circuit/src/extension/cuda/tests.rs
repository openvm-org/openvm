use std::sync::Arc;

use openvm_circuit::{
    arch::{
        rvr::{
            cuda::GpuRvrProgram, RvrCheckpointPreflightLimits, RvrPreflightEndpoint,
            RvrPreflightLimits, RvrPreflightTranscript,
        },
        VirtualMachine, VmExecutor,
    },
    utils::{test_gpu_engine, test_system_config},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
    var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_common::copy::MemCopyD2H;
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    instruction::Instruction,
    program::Program,
    riscv::{RV64_IMM_AS, RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, PhantomDiscriminant, SysPhantom, SystemOpcode,
};
use openvm_riscv_transpiler::{
    BaseAluImmOpcode, BaseAluOpcode, BaseAluWImmOpcode, BaseAluWOpcode, BranchEqualOpcode,
    BranchLessThanOpcode, DivRemOpcode, DivRemWOpcode, LessThanImmOpcode, LessThanOpcode,
    MulHOpcode, MulOpcode, MulWOpcode, Rv64AuipcOpcode, Rv64HintStoreOpcode, Rv64JalLuiOpcode,
    Rv64JalrOpcode, Rv64LoadStoreOpcode, ShiftImmOpcode, ShiftOpcode, ShiftWImmOpcode,
    ShiftWOpcode,
};
use openvm_stark_backend::{
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    StarkEngine,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::Rv64ImRvrGpuTracegen;
use crate::{
    adapters::RV64_REGISTER_NUM_LIMBS, Rv64IConfig, Rv64IGpuBuilder, Rv64ImConfig,
    Rv64ImGpuBuilder, Rv64MultiplicationChipGpu,
};

type F = BabyBear;

fn reg(index: usize) -> usize {
    index * RV64_REGISTER_NUM_LIMBS
}

fn instruction(opcode: impl LocalOpcode, operands: [usize; 5]) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), operands)
}

fn checkpoint_ri(
    opcode: impl LocalOpcode,
    rd: usize,
    rs1: usize,
    immediate: usize,
) -> Instruction<F> {
    instruction(
        opcode,
        [
            reg(rd),
            reg(rs1),
            immediate,
            RV64_REGISTER_AS as usize,
            RV64_IMM_AS as usize,
        ],
    )
}

fn checkpoint_rr(opcode: impl LocalOpcode, rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instruction(
        opcode,
        [
            reg(rd),
            reg(rs1),
            reg(rs2),
            RV64_REGISTER_AS as usize,
            RV64_REGISTER_AS as usize,
        ],
    )
}

fn checkpoint_m(opcode: impl LocalOpcode, rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instruction(
        opcode,
        [
            reg(rd),
            reg(rs1),
            reg(rs2),
            RV64_REGISTER_AS as usize,
            RV64_IMM_AS as usize,
        ],
    )
}

fn checkpoint_branch(opcode: impl LocalOpcode, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(
        opcode.global_opcode(),
        [
            reg(rs1),
            reg(rs2),
            4,
            RV64_REGISTER_AS as usize,
            RV64_REGISTER_AS as usize,
        ],
    )
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
            Rv64LoadStoreOpcode::LOADH.global_opcode(),
            [
                reg(20),
                reg(1),
                4,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            Rv64LoadStoreOpcode::LOADHU.global_opcode(),
            [
                reg(21),
                reg(1),
                3,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            Rv64LoadStoreOpcode::LOADW.global_opcode(),
            [
                reg(22),
                reg(1),
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            Rv64LoadStoreOpcode::LOADWU.global_opcode(),
            [
                reg(23),
                reg(1),
                1,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            Rv64LoadStoreOpcode::LOADD.global_opcode(),
            [
                reg(24),
                reg(1),
                2,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            Rv64LoadStoreOpcode::STOREB.global_opcode(),
            [
                reg(2),
                reg(1),
                5,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            Rv64LoadStoreOpcode::STOREH.global_opcode(),
            [
                reg(2),
                reg(1),
                4,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            Rv64LoadStoreOpcode::STOREW.global_opcode(),
            [
                reg(2),
                reg(1),
                5,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            Rv64LoadStoreOpcode::STORED.global_opcode(),
            [
                reg(2),
                reg(1),
                6,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            Rv64JalrOpcode::JALR.global_opcode(),
            [reg(30), 0, 200, RV64_REGISTER_AS as usize, 0, 1, 0],
        ),
        Instruction::phantom(
            PhantomDiscriminant(SysPhantom::Nop as u16),
            F::from_u32(0x1234),
            F::from_u32(0x5678),
            0x1234,
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
    init_memory.insert((RV64_MEMORY_AS, 7), 0x7f);
    init_memory.insert((RV64_MEMORY_AS, 8), 0x80);
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
    let tracegen = Rv64ImRvrGpuTracegen::new(&gpu_program, &gpu_transcript, &replay_plan).unwrap();
    let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();

    // The synchronized replay error read above makes every trace independent
    // of the segment inputs. Release them before the prover reaches its GPU
    // memory peak; only the immutable program stays resident across segments.
    drop(replay_plan);
    drop(gpu_transcript);

    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();
}

#[test]
fn rvr_checkpoint_gpu_replay_proves_a_suspended_segment() {
    let instructions = [
        checkpoint_ri(BaseAluImmOpcode::ADDI, 1, 0, 7),
        Instruction::from_usize(
            Rv64JalLuiOpcode::JAL.global_opcode(),
            [0, 0, 4, RV64_REGISTER_AS as usize, 0, 0, 0],
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let exe = VmExe::new(program.clone());
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let checkpoint = executor
        .rvr_experimental_checkpoint_preflight_instance(&exe, None)
        .unwrap();
    let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64IGpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&state.memory);
    let mut execution = checkpoint
        .execute_from_state_for(state, RvrCheckpointPreflightLimits::new(2, 0, 2))
        .unwrap();
    assert_eq!(
        execution.endpoint,
        RvrPreflightEndpoint::Suspended {
            resume_pc: 8,
            final_timestamp: 4,
        }
    );
    assert_eq!(execution.retired, 2);

    let gpu_program = GpuRvrProgram::upload(
        &program,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .unwrap();
    let endpoint = execution.endpoint;
    execution.endpoint = RvrPreflightEndpoint::Suspended {
        resume_pc: execution.to_state.pc + 4,
        final_timestamp: execution.to_state.timestamp,
    };
    let error = Rv64ImRvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .err()
    .expect("mismatched suspended endpoint should be rejected");
    assert!(error
        .to_string()
        .contains("suspended endpoint does not match"));
    execution.endpoint = endpoint;

    let error = Rv64ImRvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired + 1,
    )
    .err()
    .expect("a metered-boundary retirement mismatch must be rejected before replay");
    assert!(error
        .to_string()
        .contains("retired 2 instructions, expected 3"));

    let (transcript, replay_plan) = Rv64ImRvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .unwrap();
    let tracegen = Rv64ImRvrGpuTracegen::new(&gpu_program, &transcript, &replay_plan).unwrap();
    let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
    drop(replay_plan);
    drop(transcript);
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();
}

#[test]
fn rvr_checkpoint_gpu_replay_carries_a_register_across_segments() {
    let instructions = [
        checkpoint_ri(BaseAluImmOpcode::ADDI, 1, 0, 7),
        Instruction::from_usize(
            Rv64JalLuiOpcode::JAL.global_opcode(),
            [0, 0, 4, RV64_REGISTER_AS as usize, 0, 0, 0],
        ),
        checkpoint_ri(BaseAluImmOpcode::ADDI, 2, 1, 5),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let exe = VmExe::new(program.clone());
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let checkpoint = executor
        .rvr_experimental_checkpoint_preflight_instance(&exe, None)
        .unwrap();
    let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64IGpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&state.memory);
    let execution = checkpoint
        .execute_from_state_for_exact(state, RvrCheckpointPreflightLimits::new(2, 0, 2))
        .unwrap();
    assert_eq!(
        execution.endpoint,
        RvrPreflightEndpoint::Suspended {
            resume_pc: 8,
            final_timestamp: 4,
        }
    );
    assert_eq!(execution.retired, 2);

    let gpu_program = GpuRvrProgram::upload(
        &program,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .unwrap();
    let (transcript, replay_plan) = Rv64ImRvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .unwrap();
    let tracegen = Rv64ImRvrGpuTracegen::new(&gpu_program, &transcript, &replay_plan).unwrap();
    let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
    drop(replay_plan);
    drop(transcript);
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();

    vm.transport_init_memory_to_device(&execution.state.memory);
    let execution = checkpoint
        .execute_from_state_for_exact(execution.state, RvrCheckpointPreflightLimits::new(2, 0, 1))
        .unwrap();
    assert_eq!(execution.endpoint, RvrPreflightEndpoint::Terminated);
    let x2: [u16; 4] = unsafe {
        execution
            .state
            .memory
            .read(RV64_REGISTER_AS, (reg(2) / 2) as u32)
    };
    assert_eq!(x2, [12, 0, 0, 0]);

    let (transcript, replay_plan) = Rv64ImRvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .unwrap();
    let tracegen = Rv64ImRvrGpuTracegen::new(&gpu_program, &transcript, &replay_plan).unwrap();
    let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
    drop(replay_plan);
    drop(transcript);
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();
}

#[test]
fn rvr_checkpoint_gpu_replay_proves_an_empty_suspended_segment() {
    let instructions = [Instruction::from_usize(
        SystemOpcode::TERMINATE.global_opcode(),
        [0, 0, 0, 0, 0],
    )];
    let program = Program::from_instructions(&instructions);
    let exe = VmExe::new(program.clone());
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let checkpoint = executor
        .rvr_experimental_checkpoint_preflight_instance(&exe, None)
        .unwrap();
    let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64IGpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&state.memory);
    let execution = checkpoint
        .execute_from_state_for(state, RvrCheckpointPreflightLimits::new(0, 0, 1))
        .unwrap();
    assert_eq!(
        execution.endpoint,
        RvrPreflightEndpoint::Suspended {
            resume_pc: 0,
            final_timestamp: 1,
        }
    );
    assert_eq!(execution.retired, 0);
    assert!(execution.transcript.checkpoints.is_empty());

    let gpu_program = GpuRvrProgram::upload(
        &program,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .unwrap();
    let (transcript, replay_plan) = Rv64ImRvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .unwrap();
    let tracegen = Rv64ImRvrGpuTracegen::new(&gpu_program, &transcript, &replay_plan).unwrap();
    let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
    drop(replay_plan);
    drop(transcript);
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();
}

#[test]
fn rvr_checkpoint_gpu_replay_rejects_terminate_in_a_suspended_segment() {
    let instructions: [Instruction<F>; 1] = [Instruction::from_usize(
        SystemOpcode::TERMINATE.global_opcode(),
        [0, 0, 0, 0, 0],
    )];
    let program = Program::from_instructions(&instructions);
    let exe = VmExe::new(program.clone());
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let checkpoint = executor
        .rvr_experimental_checkpoint_preflight_instance(&exe, None)
        .unwrap();
    let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut vm, _) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64IGpuBuilder, config.clone())
            .unwrap();
    vm.transport_init_memory_to_device(&state.memory);
    let mut execution = checkpoint
        .execute_from_state(state, RvrCheckpointPreflightLimits::new(1, 0, 1))
        .unwrap();
    execution.endpoint = RvrPreflightEndpoint::Suspended {
        resume_pc: execution.to_state.pc,
        final_timestamp: execution.to_state.timestamp,
    };
    let gpu_program = GpuRvrProgram::upload(
        &program,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .unwrap();
    let error = Rv64ImRvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .err()
    .expect("TERMINATE should be rejected for a suspended endpoint");
    assert!(error.to_string().contains("code 308"));
}

#[test]
fn rvr_checkpoint_gpu_replay_proves_bounded_rv64i_slice_differentially() {
    let jal = |rd| {
        Instruction::from_usize(
            Rv64JalLuiOpcode::JAL.global_opcode(),
            [
                reg(rd),
                0,
                4,
                RV64_REGISTER_AS as usize,
                0,
                usize::from(rd != 0),
                0,
            ],
        )
    };
    let jalr = |rd, rs1| {
        Instruction::from_usize(
            Rv64JalrOpcode::JALR.global_opcode(),
            [
                reg(rd),
                reg(rs1),
                0,
                RV64_REGISTER_AS as usize,
                0,
                usize::from(rd != 0),
                0,
            ],
        )
    };

    let mut instructions = vec![
        checkpoint_ri(BaseAluImmOpcode::ADDI, 1, 0, 9),
        checkpoint_ri(BaseAluImmOpcode::ADDI, 2, 0, 5),
        checkpoint_ri(BaseAluImmOpcode::ADDI, 29, 0, 0xff_ffff),
        checkpoint_ri(BaseAluImmOpcode::ADDI, 28, 0, 1),
        checkpoint_ri(ShiftImmOpcode::SLLI, 28, 28, 63),
        checkpoint_rr(BaseAluOpcode::ADD, 3, 0, 2),
        checkpoint_rr(BaseAluOpcode::SUB, 4, 1, 2),
        checkpoint_rr(BaseAluOpcode::XOR, 5, 1, 2),
        checkpoint_rr(BaseAluOpcode::OR, 5, 1, 2),
        checkpoint_rr(BaseAluOpcode::AND, 5, 1, 2),
        checkpoint_rr(LessThanOpcode::SLT, 6, 29, 1),
        checkpoint_rr(LessThanOpcode::SLTU, 6, 29, 1),
        checkpoint_rr(ShiftOpcode::SLL, 7, 1, 2),
        checkpoint_rr(ShiftOpcode::SRL, 7, 28, 2),
        checkpoint_rr(ShiftOpcode::SRA, 8, 28, 2),
        checkpoint_rr(BaseAluWOpcode::ADDW, 9, 28, 2),
        checkpoint_rr(BaseAluWOpcode::SUBW, 9, 28, 2),
        checkpoint_rr(ShiftWOpcode::SLLW, 10, 1, 2),
        checkpoint_rr(ShiftWOpcode::SRLW, 10, 28, 2),
        checkpoint_rr(ShiftWOpcode::SRAW, 11, 28, 2),
        checkpoint_ri(BaseAluImmOpcode::XORI, 13, 29, 0x123),
        checkpoint_ri(BaseAluImmOpcode::ORI, 13, 29, 0x123),
        checkpoint_ri(BaseAluImmOpcode::ANDI, 13, 29, 0x123),
        checkpoint_ri(LessThanImmOpcode::SLTI, 14, 29, 0),
        checkpoint_ri(LessThanImmOpcode::SLTIU, 14, 29, 0),
        checkpoint_ri(ShiftImmOpcode::SRLI, 15, 28, 1),
        checkpoint_ri(ShiftImmOpcode::SRAI, 16, 28, 1),
        checkpoint_ri(BaseAluWImmOpcode::ADDIW, 17, 28, 0xff_ffff),
        checkpoint_ri(ShiftWImmOpcode::SLLIW, 18, 1, 3),
        checkpoint_ri(ShiftWImmOpcode::SRLIW, 18, 28, 3),
        checkpoint_ri(ShiftWImmOpcode::SRAIW, 19, 28, 3),
        checkpoint_branch(BranchEqualOpcode::BEQ, 1, 1),
        checkpoint_branch(BranchEqualOpcode::BNE, 1, 2),
        checkpoint_branch(BranchLessThanOpcode::BLT, 29, 1),
        checkpoint_branch(BranchLessThanOpcode::BLTU, 1, 29),
        checkpoint_branch(BranchLessThanOpcode::BGE, 1, 29),
        checkpoint_branch(BranchLessThanOpcode::BGEU, 29, 1),
        Instruction::from_usize(
            Rv64JalLuiOpcode::LUI.global_opcode(),
            [reg(20), 0, 0x8_0000, RV64_REGISTER_AS as usize, 0, 1, 0],
        ),
        Instruction::from_usize(
            Rv64AuipcOpcode::AUIPC.global_opcode(),
            [reg(21), 0, 0, RV64_REGISTER_AS as usize, 0, 0, 0],
        ),
        jal(0),
        jal(22),
    ];

    // The source is deliberately odd: JALR must read the old x31 value,
    // clear target bit zero, and only then overwrite the aliased destination.
    let jalr_alias_target = (instructions.len() + 2) * 4 + 1;
    instructions.push(checkpoint_ri(
        BaseAluImmOpcode::ADDI,
        31,
        0,
        jalr_alias_target,
    ));
    instructions.push(jalr(31, 31));
    let jalr_gap_target = (instructions.len() + 2) * 4;
    instructions.push(checkpoint_ri(
        BaseAluImmOpcode::ADDI,
        30,
        0,
        jalr_gap_target,
    ));
    instructions.push(jalr(0, 30));
    instructions.extend([
        checkpoint_ri(BaseAluImmOpcode::ADDI, 27, 0, 8),
        Instruction::from_usize(
            Rv64LoadStoreOpcode::LOADD.global_opcode(),
            [
                reg(26),
                reg(27),
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        checkpoint_m(MulOpcode::MUL, 23, 1, 2),
        checkpoint_m(MulHOpcode::MULH, 24, 29, 2),
        checkpoint_m(MulHOpcode::MULHSU, 25, 29, 2),
        checkpoint_m(MulHOpcode::MULHU, 26, 29, 2),
        checkpoint_m(MulWOpcode::MULW, 27, 29, 2),
        checkpoint_m(DivRemOpcode::DIV, 3, 28, 29),
        checkpoint_m(DivRemOpcode::REM, 4, 28, 29),
        checkpoint_m(DivRemOpcode::DIV, 5, 1, 0),
        checkpoint_m(DivRemOpcode::REM, 6, 1, 0),
        checkpoint_m(DivRemOpcode::DIVU, 7, 1, 0),
        checkpoint_m(DivRemOpcode::REMU, 8, 1, 0),
        checkpoint_m(DivRemOpcode::DIVU, 7, 1, 2),
        checkpoint_m(DivRemOpcode::REMU, 8, 1, 2),
        checkpoint_ri(BaseAluImmOpcode::ADDI, 30, 0, 1),
        checkpoint_ri(ShiftImmOpcode::SLLI, 30, 30, 31),
        checkpoint_m(DivRemWOpcode::DIVW, 9, 30, 29),
        checkpoint_m(DivRemWOpcode::REMW, 10, 30, 29),
        checkpoint_m(DivRemWOpcode::DIVUW, 11, 1, 0),
        checkpoint_m(DivRemWOpcode::REMUW, 12, 1, 0),
        checkpoint_m(DivRemWOpcode::DIVUW, 11, 1, 2),
        checkpoint_m(DivRemWOpcode::REMUW, 12, 1, 2),
        Instruction::phantom(
            PhantomDiscriminant(SysPhantom::Nop as u16),
            F::ZERO,
            F::ZERO,
            0,
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ]);
    let program = Program::from_instructions(&instructions);
    let loaded = 0x8877_6655_4433_2211u64;
    let init_memory = loaded
        .to_le_bytes()
        .into_iter()
        .enumerate()
        .map(|(offset, byte)| ((RV64_MEMORY_AS, 8 + offset as u32), byte))
        .collect();
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
    let config = Rv64ImConfig {
        rv64i: Rv64IConfig {
            system: test_system_config(),
            ..Default::default()
        },
        mul: Default::default(),
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let full = executor.rvr_preflight_instance(&exe, None).unwrap();
    let checkpoint = executor
        .rvr_experimental_checkpoint_preflight_instance(&exe, None)
        .unwrap();
    let max_instructions = instructions.len();
    let max_memory_events = max_instructions * 3;

    let full_execution = full
        .execute(
            Vec::<Vec<u8>>::new(),
            RvrPreflightLimits::new(max_instructions, max_memory_events),
        )
        .unwrap();
    let checkpoint_state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());

    let (mut checkpoint_vm, checkpoint_pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64ImGpuBuilder, config.clone())
            .unwrap();
    let cached_program = checkpoint_vm.commit_program_on_device(&program);
    checkpoint_vm.load_program(cached_program);
    checkpoint_vm.transport_init_memory_to_device(&checkpoint_state.memory);
    let checkpoint_execution = checkpoint
        .execute_from_state(
            checkpoint_state,
            RvrCheckpointPreflightLimits::new(max_instructions, 1, 8),
        )
        .unwrap();
    assert!(!checkpoint_execution.transcript.checkpoints.is_empty());
    assert_eq!(checkpoint_execution.transcript.residuals, vec![loaded]);
    assert_eq!(checkpoint_execution.to_state.pc, full_execution.state.pc());
    assert_eq!(
        checkpoint_execution.to_state.timestamp,
        full_execution
            .transcript
            .program_log
            .last()
            .unwrap()
            .timestamp
    );
    assert_eq!(checkpoint_execution.endpoint, full_execution.endpoint);
    assert_eq!(checkpoint_execution.retired as usize, max_instructions);
    for register in 1..32 {
        let pointer = (reg(register) / 2) as u32;
        let checkpoint_value: [u16; 4] = unsafe {
            checkpoint_execution
                .state
                .memory
                .read(RV64_REGISTER_AS, pointer)
        };
        let full_value: [u16; 4] =
            unsafe { full_execution.state.memory.read(RV64_REGISTER_AS, pointer) };
        assert_eq!(checkpoint_value, full_value);
    }

    let checkpoint_program = GpuRvrProgram::upload(
        &program,
        &config.rv64i.system.memory_config,
        &checkpoint_vm.engine.device().device_ctx,
    )
    .unwrap();
    let (checkpoint_transcript, checkpoint_plan) = Rv64ImRvrGpuTracegen::expand_checkpoint_replay(
        &checkpoint_vm,
        &checkpoint_program,
        &checkpoint_execution,
        checkpoint_execution.retired,
    )
    .unwrap();
    let checkpoint_tracegen = Rv64ImRvrGpuTracegen::new(
        &checkpoint_program,
        &checkpoint_transcript,
        &checkpoint_plan,
    )
    .unwrap();
    let checkpoint_ctx = checkpoint_tracegen
        .generate_proving_ctx(&mut checkpoint_vm)
        .unwrap();
    drop(checkpoint_plan);
    drop(checkpoint_transcript);
    let checkpoint_proof = checkpoint_vm
        .engine
        .prove(checkpoint_vm.pk(), checkpoint_ctx)
        .unwrap();
    checkpoint_vm
        .engine
        .verify(&checkpoint_pk.get_vk(), &checkpoint_proof)
        .unwrap();

    let legacy_state = full.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut legacy_vm, legacy_pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64ImGpuBuilder, config.clone())
            .unwrap();
    let cached_program = legacy_vm.commit_program_on_device(&program);
    legacy_vm.load_program(cached_program);
    legacy_vm.transport_init_memory_to_device(&legacy_state.memory);
    let legacy_execution = full
        .execute_from_state(
            legacy_state,
            RvrPreflightLimits::new(max_instructions, max_memory_events),
        )
        .unwrap();
    let legacy_program = GpuRvrProgram::upload(
        &program,
        &config.rv64i.system.memory_config,
        &legacy_vm.engine.device().device_ctx,
    )
    .unwrap();
    let (legacy_transcript, legacy_plan) = legacy_program
        .upload_transcript(&legacy_execution.transcript, legacy_execution.endpoint)
        .unwrap();
    let legacy_tracegen =
        Rv64ImRvrGpuTracegen::new(&legacy_program, &legacy_transcript, &legacy_plan).unwrap();
    let legacy_ctx = legacy_tracegen
        .generate_proving_ctx(&mut legacy_vm)
        .unwrap();
    drop(legacy_plan);
    drop(legacy_transcript);
    let legacy_proof = legacy_vm.engine.prove(legacy_vm.pk(), legacy_ctx).unwrap();
    legacy_vm
        .engine
        .verify(&legacy_pk.get_vk(), &legacy_proof)
        .unwrap();
}

#[test]
fn rvr_checkpoint_gpu_replay_proves_all_memory_intent_shapes() {
    let memory_instruction = |opcode: Rv64LoadStoreOpcode,
                              reg_operand: usize,
                              offset: u16,
                              offset_is_negative: bool,
                              is_load: bool| {
        Instruction::from_usize(
            opcode.global_opcode(),
            [
                reg(reg_operand),
                reg(1),
                usize::from(offset),
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                usize::from(!is_load || reg_operand != 0),
                usize::from(offset_is_negative),
            ],
        )
    };
    let block_boundary = || {
        Instruction::from_usize(
            Rv64JalLuiOpcode::JAL.global_opcode(),
            [0, 0, 4, RV64_REGISTER_AS as usize, 0, 0, 0],
        )
    };

    let mut instructions = Vec::new();
    let mut append_store_load = |store, load, rd, offset: u16, offset_is_negative: bool| {
        instructions.push(memory_instruction(
            store,
            2,
            offset,
            offset_is_negative,
            false,
        ));
        // Force a basic-block/checkpoint boundary between the store and
        // dependent load, so device chunks cannot rely on mutable memory.
        instructions.push(block_boundary());
        instructions.push(memory_instruction(
            load,
            rd,
            offset,
            offset_is_negative,
            true,
        ));
        instructions.push(block_boundary());
    };
    append_store_load(
        Rv64LoadStoreOpcode::STOREB,
        Rv64LoadStoreOpcode::LOADBU,
        3,
        0,
        false,
    );
    append_store_load(
        Rv64LoadStoreOpcode::STOREH,
        Rv64LoadStoreOpcode::LOADH,
        4,
        7,
        false,
    );
    append_store_load(
        Rv64LoadStoreOpcode::STOREW,
        Rv64LoadStoreOpcode::LOADW,
        5,
        6,
        false,
    );
    append_store_load(
        Rv64LoadStoreOpcode::STORED,
        Rv64LoadStoreOpcode::LOADD,
        6,
        u16::MAX,
        true,
    );
    instructions.extend([
        memory_instruction(Rv64LoadStoreOpcode::LOADB, 7, 6, false, true),
        memory_instruction(Rv64LoadStoreOpcode::LOADH, 10, 5, false, true),
        memory_instruction(Rv64LoadStoreOpcode::LOADW, 11, 3, false, true),
        memory_instruction(Rv64LoadStoreOpcode::LOADHU, 8, 7, false, true),
        memory_instruction(Rv64LoadStoreOpcode::LOADWU, 9, 6, false, true),
        // A disabled destination still reserves its AIR timestamp slot but
        // appends no residual and no register-write event.
        memory_instruction(Rv64LoadStoreOpcode::LOADD, 0, u16::MAX, true, true),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ]);

    let program = Program::from_instructions(&instructions);
    let initial_registers = [(1usize, 40u64), (2, 0x8877_6655_4433_2211u64)]
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
    let exe = VmExe::new(program.clone()).with_init_memory(initial_registers);
    let config = Rv64ImConfig {
        rv64i: Rv64IConfig {
            system: test_system_config(),
            ..Default::default()
        },
        mul: Default::default(),
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let checkpoint = executor
        .rvr_experimental_checkpoint_preflight_instance(&exe, None)
        .unwrap();
    let checkpoint_state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut checkpoint_vm, checkpoint_pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64ImGpuBuilder, config.clone())
            .unwrap();
    let cached_program = checkpoint_vm.commit_program_on_device(&program);
    checkpoint_vm.load_program(cached_program);
    checkpoint_vm.transport_init_memory_to_device(&checkpoint_state.memory);
    let checkpoint_execution = checkpoint
        .execute_from_state(
            checkpoint_state,
            RvrCheckpointPreflightLimits::new(instructions.len(), 16, 1),
        )
        .unwrap();

    assert!(checkpoint_execution.transcript.checkpoints.len() >= 4);
    assert_eq!(
        checkpoint_execution.transcript.residuals,
        [
            0x11,
            0x2211,
            0x4433_2211,
            0x8877_6655_4433_2211,
            0xffff_ffff_ffff_ff88,
            0xffff_ffff_ffff_8877,
            0xffff_ffff_8877_6655,
            0x3322,
            0x4433_2288,
        ]
    );
    let checkpoint_program = GpuRvrProgram::upload(
        &program,
        &config.rv64i.system.memory_config,
        &checkpoint_vm.engine.device().device_ctx,
    )
    .unwrap();
    let (checkpoint_transcript, checkpoint_plan) = Rv64ImRvrGpuTracegen::expand_checkpoint_replay(
        &checkpoint_vm,
        &checkpoint_program,
        &checkpoint_execution,
        checkpoint_execution.retired,
    )
    .unwrap();
    let checkpoint_tracegen = Rv64ImRvrGpuTracegen::new(
        &checkpoint_program,
        &checkpoint_transcript,
        &checkpoint_plan,
    )
    .unwrap();
    let checkpoint_ctx = checkpoint_tracegen
        .generate_proving_ctx(&mut checkpoint_vm)
        .unwrap();
    drop(checkpoint_plan);
    drop(checkpoint_transcript);
    let proof = checkpoint_vm
        .engine
        .prove(checkpoint_vm.pk(), checkpoint_ctx)
        .unwrap();
    checkpoint_vm
        .engine
        .verify(&checkpoint_pk.get_vk(), &proof)
        .unwrap();
}

#[test]
fn rvr_gpu_tracegen_proves_rv64m_airs_without_record_arenas() {
    let m_operands = |rd, rs1, rs2| {
        [
            reg(rd),
            reg(rs1),
            reg(rs2),
            RV64_REGISTER_AS as usize,
            RV64_IMM_AS as usize,
        ]
    };
    let instructions = [
        instruction(MulOpcode::MUL, m_operands(3, 1, 2)),
        instruction(MulWOpcode::MULW, m_operands(4, 1, 2)),
        instruction(MulHOpcode::MULH, m_operands(5, 1, 2)),
        instruction(MulHOpcode::MULHSU, m_operands(6, 1, 2)),
        instruction(MulHOpcode::MULHU, m_operands(7, 1, 2)),
        instruction(DivRemOpcode::DIV, m_operands(8, 1, 2)),
        instruction(DivRemOpcode::DIVU, m_operands(9, 1, 2)),
        instruction(DivRemOpcode::REM, m_operands(10, 1, 2)),
        instruction(DivRemOpcode::REMU, m_operands(11, 1, 2)),
        instruction(DivRemWOpcode::DIVW, m_operands(12, 1, 2)),
        instruction(DivRemWOpcode::DIVUW, m_operands(13, 1, 2)),
        instruction(DivRemWOpcode::REMW, m_operands(14, 1, 2)),
        instruction(DivRemWOpcode::REMUW, m_operands(15, 1, 2)),
        // Source x0 reads, destination aliases, divide by zero, and signed
        // overflow all use the same fixed read/read/write replay schedule.
        instruction(MulOpcode::MUL, m_operands(1, 1, 0)),
        instruction(MulWOpcode::MULW, m_operands(2, 0, 2)),
        instruction(DivRemOpcode::DIV, m_operands(18, 16, 17)),
        instruction(DivRemOpcode::REM, m_operands(19, 16, 17)),
        instruction(DivRemOpcode::DIVU, m_operands(20, 1, 0)),
        instruction(DivRemOpcode::REMU, m_operands(21, 1, 0)),
        instruction(DivRemWOpcode::DIVW, m_operands(22, 16, 17)),
        instruction(DivRemWOpcode::REMW, m_operands(23, 16, 17)),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let init_memory = [
        (1usize, (-5i64) as u64),
        (2, 3u64),
        (16, i64::MIN as u64),
        (17, u64::MAX),
    ]
    .into_iter()
    .flat_map(|(register, value)| {
        value
            .to_le_bytes()
            .into_iter()
            .enumerate()
            .map(move |(offset, byte)| ((RV64_REGISTER_AS, (reg(register) + offset) as u32), byte))
    })
    .collect();
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
    let config = Rv64ImConfig {
        rv64i: Rv64IConfig {
            system: test_system_config(),
            ..Default::default()
        },
        mul: Default::default(),
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let rvr = executor.rvr_preflight_instance(&exe, None).unwrap();
    let state = rvr.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64ImGpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&state.memory);
    let execution = rvr
        .execute_from_state(state, RvrPreflightLimits::new(32, 96))
        .unwrap();
    assert_eq!(execution.transcript.memory_log.len(), 21 * 3);

    let device_ctx = &vm.engine.device().device_ctx;
    let gpu_program =
        GpuRvrProgram::upload(&program, &config.rv64i.system.memory_config, device_ctx).unwrap();
    let (gpu_transcript, replay_plan) = gpu_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    let tracegen = Rv64ImRvrGpuTracegen::new(&gpu_program, &gpu_transcript, &replay_plan).unwrap();
    let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
    drop(replay_plan);
    drop(gpu_transcript);
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();
}

#[test]
fn rvr_mul_replay_rejects_corrupt_results_and_predecessors_before_lookups() {
    let instructions = [
        instruction(
            MulOpcode::MUL,
            [
                reg(3),
                reg(1),
                reg(1),
                RV64_REGISTER_AS as usize,
                RV64_IMM_AS as usize,
            ],
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let init_memory = 2u64
        .to_le_bytes()
        .into_iter()
        .enumerate()
        .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(1) + offset) as u32), byte))
        .collect();
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
    let config = Rv64ImConfig {
        rv64i: Rv64IConfig {
            system: test_system_config(),
            ..Default::default()
        },
        mul: Default::default(),
    };
    let execution = VmExecutor::new(config.clone())
        .unwrap()
        .rvr_preflight_instance(&exe, None)
        .unwrap()
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(4, 4))
        .unwrap();
    assert_eq!(execution.transcript.memory_log.len(), 3);
    let engine = test_gpu_engine();
    let device_ctx = &engine.device().device_ctx;
    let gpu_program =
        GpuRvrProgram::upload(&program, &config.rv64i.system.memory_config, device_ctx).unwrap();

    let reject = |transcript: &RvrPreflightTranscript, expected_code| {
        let (gpu_transcript, replay_plan) = gpu_program
            .upload_transcript(transcript, RvrPreflightEndpoint::Terminated)
            .unwrap();
        let range_checker = Arc::new(VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        ));
        let bitwise_lookup = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
        let range_tuple = Arc::new(RangeTupleCheckerChipGPU::new(
            config.mul.range_tuple_checker_sizes,
            device_ctx.clone(),
        ));
        let chip = Rv64MultiplicationChipGpu::new(
            range_checker.clone(),
            bitwise_lookup.clone(),
            range_tuple.clone(),
            config.rv64i.system.memory_config.timestamp_max_bits,
        );
        chip.generate_proving_ctx_from_rvr(&gpu_program, &gpu_transcript, &replay_plan)
            .unwrap();
        assert_eq!(gpu_transcript.error_code().unwrap(), expected_code);
        for count in [
            range_checker.count.to_host_on(device_ctx).unwrap(),
            bitwise_lookup.count.to_host_on(device_ctx).unwrap(),
            range_tuple.count.to_host_on(device_ctx).unwrap(),
        ] {
            assert!(count.iter().all(|value| value.as_canonical_u32() == 0));
        }
    };

    let mut corrupt_result = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    corrupt_result.memory_log[2].value[0] = 5;
    reject(&corrupt_result, 609);

    let mut corrupt_predecessor = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    corrupt_predecessor.memory_log[1].value[0] = 3;
    corrupt_predecessor.memory_log[2].value[0] = 6;
    reject(&corrupt_predecessor, 608);
}

#[test]
fn rvr_mul_replay_rejects_raw_x0_destination_without_lookups() {
    let instructions = [
        instruction(
            MulOpcode::MUL,
            [
                0,
                reg(1),
                reg(2),
                RV64_REGISTER_AS as usize,
                RV64_IMM_AS as usize,
            ],
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let init_memory = [(1usize, 2u64), (2, 3u64)]
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
    let config = Rv64ImConfig {
        rv64i: Rv64IConfig {
            system: test_system_config(),
            ..Default::default()
        },
        mul: Default::default(),
    };
    let execution = VmExecutor::new(config.clone())
        .unwrap()
        .rvr_preflight_instance(&exe, None)
        .unwrap()
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(4, 3))
        .unwrap();
    assert_eq!(execution.transcript.memory_log.len(), 2);

    let engine = test_gpu_engine();
    let device_ctx = &engine.device().device_ctx;
    let gpu_program =
        GpuRvrProgram::upload(&program, &config.rv64i.system.memory_config, device_ctx).unwrap();
    let (gpu_transcript, replay_plan) = gpu_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    let range_checker = Arc::new(VariableRangeCheckerChipGPU::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
        device_ctx.clone(),
    ));
    let bitwise_lookup = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
    let range_tuple = Arc::new(RangeTupleCheckerChipGPU::new(
        config.mul.range_tuple_checker_sizes,
        device_ctx.clone(),
    ));
    let chip = Rv64MultiplicationChipGpu::new(
        range_checker.clone(),
        bitwise_lookup.clone(),
        range_tuple.clone(),
        config.rv64i.system.memory_config.timestamp_max_bits,
    );
    chip.generate_proving_ctx_from_rvr(&gpu_program, &gpu_transcript, &replay_plan)
        .unwrap();
    assert_eq!(gpu_transcript.error_code().unwrap(), 604);
    for count in [
        range_checker.count.to_host_on(device_ctx).unwrap(),
        bitwise_lookup.count.to_host_on(device_ctx).unwrap(),
        range_tuple.count.to_host_on(device_ctx).unwrap(),
    ] {
        assert!(count.iter().all(|value| value.as_canonical_u32() == 0));
    }
}

#[test]
fn rvr_checkpoint_gpu_replay_proves_hint_store_without_record_arenas() {
    let instructions = [
        Instruction::<F>::from_usize(
            Rv64HintStoreOpcode::HINT_STORED.global_opcode(),
            [
                0,
                reg(1),
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::<F>::from_usize(
            Rv64HintStoreOpcode::HINT_BUFFER.global_opcode(),
            [
                reg(2),
                reg(3),
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let mut init_memory: SparseMemoryImage = [(1usize, 32u64), (2, 3u64), (3, 64u64)]
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
    // Both hint instructions overwrite nonzero initial words. This exercises the first-write
    // seed path and makes an incorrectly zeroed write predecessor fail the memory argument.
    init_memory.extend(
        [(32u32, 0x55u8), (39, 0xaa), (64, 0x12), (87, 0xfe)]
            .into_iter()
            .map(|(byte_ptr, byte)| ((RV64_MEMORY_AS, byte_ptr), byte)),
    );
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let checkpoint = executor
        .rvr_experimental_checkpoint_preflight_instance(&exe, None)
        .unwrap();
    let mut state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let hint_words = [
        0x0123_4567_89ab_cdefu64,
        0x1111_2222_3333_4444,
        0xaaaa_bbbb_cccc_dddd,
        0xfedc_ba98_7654_3210,
    ];
    state.streams.hint_stream.set_hint(
        hint_words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect(),
    );
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64IGpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&state.memory);
    let mut execution = checkpoint
        .execute_from_state(
            state,
            RvrCheckpointPreflightLimits::new(instructions.len(), hint_words.len(), 1),
        )
        .unwrap();
    assert_eq!(execution.transcript.residuals, hint_words);
    assert_eq!(execution.to_state.timestamp, 13);

    let gpu_program = GpuRvrProgram::upload(
        &program,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .unwrap();
    let missing = execution.transcript.residuals.pop().unwrap();
    let error = Rv64ImRvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .err()
    .expect("missing hint residual must fail checkpoint replay");
    assert!(error.to_string().contains("code 306"), "{error}");
    execution.transcript.residuals.push(missing);

    let (transcript, replay_plan) = Rv64ImRvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .unwrap();
    assert_eq!(
        replay_plan
            .opcode_range(Rv64HintStoreOpcode::HINT_STORED.global_opcode())
            .len(),
        1
    );
    assert_eq!(
        replay_plan
            .opcode_range(Rv64HintStoreOpcode::HINT_BUFFER.global_opcode())
            .len(),
        1
    );
    let tracegen = Rv64ImRvrGpuTracegen::new(&gpu_program, &transcript, &replay_plan).unwrap();
    let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
    assert_eq!(transcript.error_code().unwrap(), 0);
    drop(replay_plan);
    drop(transcript);
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();
}
