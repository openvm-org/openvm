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
    exe::VmExe,
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
fn rvr_checkpoint_gpu_replay_proves_bounded_rv64i_slice_differentially() {
    let instructions = [
        instruction(
            BaseAluImmOpcode::ADDI,
            [
                reg(31),
                reg(0),
                8,
                RV64_REGISTER_AS as usize,
                RV64_IMM_AS as usize,
            ],
        ),
        Instruction::from_usize(
            Rv64LoadStoreOpcode::LOADD.global_opcode(),
            [
                reg(2),
                reg(31),
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        instruction(
            BaseAluImmOpcode::ADDI,
            [
                reg(3),
                reg(2),
                1,
                RV64_REGISTER_AS as usize,
                RV64_IMM_AS as usize,
            ],
        ),
        Instruction::from_usize(
            BranchEqualOpcode::BNE.global_opcode(),
            [
                reg(3),
                reg(2),
                8,
                RV64_REGISTER_AS as usize,
                RV64_REGISTER_AS as usize,
            ],
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 1, 0, 0]),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let loaded = 0x8877_6655_4433_2211u64;
    let init_memory = loaded
        .to_le_bytes()
        .into_iter()
        .enumerate()
        .map(|(offset, byte)| ((RV64_MEMORY_AS, 8 + offset as u32), byte))
        .collect();
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let full = executor.rvr_preflight_instance(&exe, None).unwrap();
    let checkpoint = executor
        .rvr_experimental_checkpoint_preflight_instance(&exe, None)
        .unwrap();

    let full_execution = full
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(5, 16))
        .unwrap();
    let checkpoint_state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());

    let (mut checkpoint_vm, checkpoint_pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64IGpuBuilder, config.clone())
            .unwrap();
    let cached_program = checkpoint_vm.commit_program_on_device(&program);
    checkpoint_vm.load_program(cached_program);
    checkpoint_vm.transport_init_memory_to_device(&checkpoint_state.memory);
    let checkpoint_execution = checkpoint
        .execute_from_state(checkpoint_state, RvrCheckpointPreflightLimits::new(5, 1, 2))
        .unwrap();
    assert_eq!(checkpoint_execution.transcript.checkpoints.len(), 1);
    assert_eq!(checkpoint_execution.transcript.residuals, vec![loaded]);
    let checkpoint_anchor = checkpoint_execution.transcript.checkpoints[0];
    assert_eq!(
        (
            checkpoint_anchor.pc,
            checkpoint_anchor.timestamp,
            checkpoint_anchor.retired,
            checkpoint_anchor.residual_cursor,
        ),
        (20, 11, 4, 1),
    );
    assert_eq!(checkpoint_anchor.regs[30], 8);
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
    assert_eq!(checkpoint_execution.retired, 5);
    for register in [2, 3, 31] {
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
        &config.system.memory_config,
        &checkpoint_vm.engine.device().device_ctx,
    )
    .unwrap();
    let (checkpoint_transcript, checkpoint_plan) = Rv64ImRvrGpuTracegen::expand_checkpoint_replay(
        &checkpoint_vm,
        &checkpoint_program,
        &checkpoint_execution,
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
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64IGpuBuilder, config.clone())
            .unwrap();
    let cached_program = legacy_vm.commit_program_on_device(&program);
    legacy_vm.load_program(cached_program);
    legacy_vm.transport_init_memory_to_device(&legacy_state.memory);
    let legacy_execution = full
        .execute_from_state(legacy_state, RvrPreflightLimits::new(5, 16))
        .unwrap();
    let legacy_program = GpuRvrProgram::upload(
        &program,
        &config.system.memory_config,
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
fn rvr_gpu_tracegen_rejects_an_executed_unported_opcode_before_tracegen() {
    let instructions = [
        Instruction::<F>::from_usize(
            Rv64HintStoreOpcode::HINT_BUFFER.global_opcode(),
            [
                reg(1),
                reg(2),
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let init_memory = [(1usize, 1u64), (2, 16u64)]
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
    let mut state = rvr.create_initial_vm_state(Vec::<Vec<u8>>::new());
    state.streams.hint_stream.set_hint(vec![0x42; 8]);
    let execution = rvr
        .execute_from_state(state, RvrPreflightLimits::new(8, 16))
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

    let error = match Rv64ImRvrGpuTracegen::new(&gpu_program, &gpu_transcript, &replay_plan) {
        Ok(_) => panic!("executed HINT_BUFFER must not reach tracegen before its replay port"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("does not support executed opcode"),
        "unexpected coverage error: {error}"
    );
}
