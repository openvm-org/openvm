use openvm_bigint_transpiler::{
    BaseAlu256Opcode, BranchEqual256Opcode, BranchLessThan256Opcode, LessThan256Opcode,
    Mul256Opcode, Shift256Opcode,
};
use openvm_circuit::{
    arch::{
        rvr::{PreflightEndpoint, PreflightLimits},
        PreflightHistory, PreflightMemoryLog, VirtualMachine, VmExecutor,
    },
    utils::{test_gpu_engine, test_system_config},
};
use openvm_cuda_backend::prelude::F;
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    instruction::Instruction,
    program::Program,
    riscv::{IMM_AS, MEMORY_AS, REGISTER_AS, REGISTER_BYTES},
    LocalOpcode, SystemOpcode, VmOpcode,
};
use openvm_riscv_transpiler::{
    BaseAluImmOpcode, BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode,
    MulOpcode, ShiftOpcode,
};
use openvm_stark_backend::StarkEngine;

use super::{Int256PreflightGpuTracegen, Int256Rv64GpuBuilder};
use crate::Int256Rv64Config;

const DST_PTR: u32 = 0x100;
const LHS_PTR: u32 = 0x200;
const RHS_PTR: u32 = 0x300;

fn reg(index: usize) -> usize {
    index * REGISTER_BYTES as usize
}

fn fixture(equal: bool) -> (Program, VmExe) {
    let instructions = [
        Instruction::from_usize(
            BaseAluImmOpcode::ADDI.global_opcode(),
            [reg(4), reg(0), 7, REGISTER_AS as usize, IMM_AS as usize],
        ),
        Instruction::from_usize(
            BaseAlu256Opcode(BaseAluOpcode::ADD).global_opcode(),
            [
                reg(1),
                reg(2),
                reg(3),
                REGISTER_AS as usize,
                MEMORY_AS as usize,
            ],
        ),
        Instruction::from_usize(
            BranchEqual256Opcode(BranchEqualOpcode::BEQ).global_opcode(),
            [reg(2), reg(3), 8, REGISTER_AS as usize, MEMORY_AS as usize],
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
    ];
    let program = Program::from_instructions(&instructions);

    let lhs =
        std::array::from_fn::<_, 32, _>(|index| (index as u8).wrapping_mul(17).wrapping_add(3));
    let rhs = if equal {
        lhs
    } else {
        let mut rhs = lhs;
        rhs[31] ^= 1;
        rhs
    };
    let mut init_memory = SparseMemoryImage::default();
    for (register, pointer) in [(1, DST_PTR), (2, LHS_PTR), (3, RHS_PTR)] {
        init_memory.extend(
            u64::from(pointer)
                .to_le_bytes()
                .into_iter()
                .enumerate()
                .map(|(offset, byte)| ((REGISTER_AS, (reg(register) + offset) as u32), byte)),
        );
    }
    init_memory.extend(
        lhs.into_iter()
            .enumerate()
            .map(|(offset, byte)| ((MEMORY_AS, LHS_PTR + offset as u32), byte)),
    );
    init_memory.extend(
        rhs.into_iter()
            .enumerate()
            .map(|(offset, byte)| ((MEMORY_AS, RHS_PTR + offset as u32), byte)),
    );
    (
        program.clone(),
        VmExe::new(program).with_init_memory(init_memory),
    )
}

#[derive(Clone, Copy)]
struct OpcodeCase {
    opcode: VmOpcode,
    expected_branch: Option<bool>,
}

fn all_opcode_fixture() -> (Vec<OpcodeCase>, Program, VmExe) {
    let cases = vec![
        OpcodeCase {
            opcode: BaseAlu256Opcode(BaseAluOpcode::ADD).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: BaseAlu256Opcode(BaseAluOpcode::SUB).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: BaseAlu256Opcode(BaseAluOpcode::XOR).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: BaseAlu256Opcode(BaseAluOpcode::OR).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: BaseAlu256Opcode(BaseAluOpcode::AND).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Shift256Opcode(ShiftOpcode::SLL).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Shift256Opcode(ShiftOpcode::SRL).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Shift256Opcode(ShiftOpcode::SRA).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: LessThan256Opcode(LessThanOpcode::SLT).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: LessThan256Opcode(LessThanOpcode::SLTU).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Mul256Opcode(MulOpcode::MUL).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: BranchEqual256Opcode(BranchEqualOpcode::BEQ).global_opcode(),
            expected_branch: Some(false),
        },
        OpcodeCase {
            opcode: BranchEqual256Opcode(BranchEqualOpcode::BNE).global_opcode(),
            expected_branch: Some(true),
        },
        OpcodeCase {
            opcode: BranchLessThan256Opcode(BranchLessThanOpcode::BLT).global_opcode(),
            expected_branch: Some(true),
        },
        OpcodeCase {
            opcode: BranchLessThan256Opcode(BranchLessThanOpcode::BLTU).global_opcode(),
            expected_branch: Some(false),
        },
        OpcodeCase {
            opcode: BranchLessThan256Opcode(BranchLessThanOpcode::BGE).global_opcode(),
            expected_branch: Some(false),
        },
        OpcodeCase {
            opcode: BranchLessThan256Opcode(BranchLessThanOpcode::BGEU).global_opcode(),
            expected_branch: Some(true),
        },
    ];
    assert_eq!(cases.len(), 17);

    // lhs is signed -2^255 but unsigned 2^255. This makes the signed and unsigned
    // branch/comparison pairs take opposite boundary paths. A nonzero rhs also
    // exercises multiplication, cross-limb subtraction, and nontrivial shifts.
    let mut lhs = [0u8; 32];
    lhs[31] = 0x80;
    let mut rhs = [0u8; 32];
    rhs[0] = 65;
    let mut instructions = cases
        .iter()
        .map(|case| {
            if let Some(expected_branch) = case.expected_branch {
                Instruction::from_isize(
                    case.opcode,
                    reg(2) as isize,
                    reg(3) as isize,
                    if expected_branch { 4 } else { -4 },
                    REGISTER_AS as isize,
                    MEMORY_AS as isize,
                )
            } else {
                Instruction::from_usize(
                    case.opcode,
                    [
                        reg(1),
                        reg(2),
                        reg(3),
                        REGISTER_AS as usize,
                        MEMORY_AS as usize,
                    ],
                )
            }
        })
        .collect::<Vec<_>>();
    instructions.push(Instruction::from_usize(
        SystemOpcode::TERMINATE.global_opcode(),
        [0; 5],
    ));
    let program = Program::from_instructions(&instructions);

    let mut init_memory = SparseMemoryImage::default();
    for (register, pointer) in [(1, DST_PTR), (2, LHS_PTR), (3, RHS_PTR)] {
        init_memory.extend(
            u64::from(pointer)
                .to_le_bytes()
                .into_iter()
                .enumerate()
                .map(|(offset, byte)| ((REGISTER_AS, (reg(register) + offset) as u32), byte)),
        );
    }
    init_memory.extend(
        lhs.into_iter()
            .enumerate()
            .map(|(offset, byte)| ((MEMORY_AS, LHS_PTR + offset as u32), byte)),
    );
    init_memory.extend(
        rhs.into_iter()
            .enumerate()
            .map(|(offset, byte)| ((MEMORY_AS, RHS_PTR + offset as u32), byte)),
    );
    (
        cases,
        program.clone(),
        VmExe::new(program).with_init_memory(init_memory),
    )
}

#[test]
fn all_int256_opcodes_checkpoint_expand_and_prove() {
    let (cases, program, exe) = all_opcode_fixture();
    let config = Int256Rv64Config {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::<F, _>::new(config.clone()).unwrap();
    let checkpoint = executor.preflight_instance(&exe).unwrap();
    let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Int256Rv64GpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&state.memory);
    let execution = checkpoint
        .execute_from_state(state, PreflightLimits::new(cases.len() + 1, 50, 1))
        .unwrap();

    assert_eq!(execution.to_state.pc, (cases.len() * 4) as u32);
    assert_eq!(execution.to_state.timestamp, 226);
    assert_eq!(execution.transcript.replay_values.len(), 50);
    assert_eq!(
        &execution.transcript.replay_values[44..],
        &[0, 1, 1, 0, 0, 1]
    );

    let gpu_program = Int256PreflightGpuTracegen::upload_postflight_program(
        &program,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .unwrap();
    let (transcript, replay_plan) =
        Int256PreflightGpuTracegen::postflight(&vm, &gpu_program, &execution, execution.retired)
            .unwrap();
    assert_eq!(transcript.error_code().unwrap(), 0);
    assert_eq!(transcript.memory_log_host().unwrap().len(), 225);
    for case in &cases {
        assert_eq!(replay_plan.opcode_range(case.opcode).len(), 1);
    }
    let host_history = PreflightHistory {
        program: transcript.program_log_host().unwrap(),
        memory: PreflightMemoryLog {
            accesses: transcript.memory_log_host().unwrap(),
            initial_writes: transcript.initial_write_log_host().unwrap(),
            ..Default::default()
        },
    };

    let tracegen =
        Int256PreflightGpuTracegen::new(gpu_program.program(), &transcript, &replay_plan);
    let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
    drop(replay_plan);
    drop(transcript);
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();

    let invalid_state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut invalid_vm, _) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Int256Rv64GpuBuilder, config.clone())
            .unwrap();
    let cached_program = invalid_vm.commit_program_on_device(&program);
    invalid_vm.load_program(cached_program);
    invalid_vm.transport_init_memory_to_device(&invalid_state.memory);
    let invalid_gpu_program = Int256PreflightGpuTracegen::upload_postflight_program(
        &program,
        &config.system.memory_config,
        &invalid_vm.engine.device().device_ctx,
    )
    .unwrap();
    let mut invalid_history = host_history;
    let pointer_block = reg(1) as u32 / 2;
    let mut mutated_reads = 0;
    for pointer_event in invalid_history.memory.accesses.iter_mut().filter(|event| {
        event.address_space() == REGISTER_AS && !event.is_write() && event.pointer == pointer_block
    }) {
        assert_eq!(pointer_event.value, [DST_PTR as u16, 0, 0, 0]);
        pointer_event.value[0] += 2;
        mutated_reads += 1;
    }
    assert!(
        mutated_reads > 1,
        "fixture must reuse the destination pointer"
    );
    let (invalid_transcript, invalid_plan) = invalid_gpu_program
        .program()
        .upload_history_for_test(&program, &invalid_history, Some(0))
        .unwrap();
    let error = Int256PreflightGpuTracegen::new(
        invalid_gpu_program.program(),
        &invalid_transcript,
        &invalid_plan,
    )
    .generate_proving_ctx(&mut invalid_vm)
    .err()
    .expect("Int256 GPU replay must reject a two-byte-aligned heap pointer");
    assert!(error.to_string().contains("code 403"), "{error}");
}

#[test]
fn int256_checkpoint_replay_rejects_wrapping_transitions() {
    let instructions = [
        Instruction::from_usize(
            BranchEqual256Opcode(BranchEqualOpcode::BEQ).global_opcode(),
            [reg(2), reg(3), 8, REGISTER_AS as usize, MEMORY_AS as usize],
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
    ];
    let program = Program::from_instructions(&instructions);
    let mut init_memory = SparseMemoryImage::default();
    for (register, pointer) in [(2, LHS_PTR), (3, RHS_PTR)] {
        init_memory.extend(
            u64::from(pointer)
                .to_le_bytes()
                .into_iter()
                .enumerate()
                .map(|(offset, byte)| ((REGISTER_AS, (reg(register) + offset) as u32), byte)),
        );
    }
    init_memory.extend(
        [0u8; 32]
            .into_iter()
            .enumerate()
            .map(|(offset, byte)| ((MEMORY_AS, LHS_PTR + offset as u32), byte)),
    );
    init_memory.extend(
        std::iter::once(1u8)
            .chain([0u8; 31])
            .enumerate()
            .map(|(offset, byte)| ((MEMORY_AS, RHS_PTR + offset as u32), byte)),
    );
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
    let config = Int256Rv64Config {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::<F, _>::new(config.clone()).unwrap();
    let checkpoint = executor.preflight_instance(&exe).unwrap();
    let initial_state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut source_vm, _) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Int256Rv64GpuBuilder, config.clone())
            .unwrap();
    let cached_program = source_vm.commit_program_on_device(&program);
    source_vm.load_program(cached_program);
    source_vm.transport_init_memory_to_device(&initial_state.memory);
    let execution = checkpoint
        .execute_from_state_for(initial_state, PreflightLimits::new(1, 1, 1))
        .unwrap();
    assert_eq!(execution.transcript.replay_values.len(), 1);
    assert_eq!(execution.endpoint, PreflightEndpoint::Suspended);
    let gpu_program = Int256PreflightGpuTracegen::upload_postflight_program(
        &program,
        &config.system.memory_config,
        &source_vm.engine.device().device_ctx,
    )
    .unwrap();
    let (transcript, replay_plan) = Int256PreflightGpuTracegen::postflight(
        &source_vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .unwrap();
    let host_history = PreflightHistory {
        program: transcript.program_log_host().unwrap(),
        memory: PreflightMemoryLog {
            accesses: transcript.memory_log_host().unwrap(),
            initial_writes: transcript.initial_write_log_host().unwrap(),
            ..Default::default()
        },
    };
    drop(replay_plan);
    drop(transcript);

    for corrupt_timestamp in [true, false] {
        let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
        let (mut vm, _) = VirtualMachine::new_with_keygen(
            test_gpu_engine(),
            Int256Rv64GpuBuilder,
            config.clone(),
        )
        .unwrap();
        let cached_program = vm.commit_program_on_device(&program);
        vm.load_program(cached_program);
        vm.transport_init_memory_to_device(&state.memory);
        let gpu_program = Int256PreflightGpuTracegen::upload_postflight_program(
            &program,
            &config.system.memory_config,
            &vm.engine.device().device_ctx,
        )
        .unwrap();
        let (mut transcript, replay_plan) = gpu_program
            .program()
            .upload_history_for_test(&program, &host_history, None)
            .unwrap();
        let mut program_log = host_history.program.clone();
        if corrupt_timestamp {
            // Int256 branch rows consume 10 timed events. This pair would pass a
            // wrapping addition check while violating the u32 timestamp domain.
            program_log[0].timestamp = u32::MAX - 9;
            program_log[1].timestamp = 0;
        } else {
            // A sequential +4 from this PC wraps to zero.
            program_log[0].pc = u32::MAX - 3;
            program_log[1].pc = 0;
        }
        transcript
            .replace_program_log_for_test(&program_log)
            .unwrap();
        let error =
            Int256PreflightGpuTracegen::new(gpu_program.program(), &transcript, &replay_plan)
                .generate_proving_ctx(&mut vm)
                .err()
                .expect("Int256 GPU replay must reject a wrapping transition");
        let expected_code = if corrupt_timestamp { 401 } else { 402 };
        assert!(
            error.to_string().contains(&format!("code {expected_code}")),
            "{error}"
        );
    }
}

#[test]
fn mixed_rv64_int256_checkpoint_expansion_proves_both_branch_outcomes() {
    for (equal, expected_pc, expected_branch_replay_value) in [(false, 12, 0u64), (true, 16, 1u64)]
    {
        let (program, exe) = fixture(equal);
        let config = Int256Rv64Config {
            system: test_system_config(),
            ..Default::default()
        };
        let executor = VmExecutor::<F, _>::new(config.clone()).unwrap();
        let checkpoint = executor.preflight_instance(&exe).unwrap();
        let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
        let (mut vm, pk) = VirtualMachine::new_with_keygen(
            test_gpu_engine(),
            Int256Rv64GpuBuilder,
            config.clone(),
        )
        .unwrap();
        let cached_program = vm.commit_program_on_device(&program);
        vm.load_program(cached_program);
        vm.transport_init_memory_to_device(&state.memory);
        let mut execution = checkpoint
            .execute_from_state(state, PreflightLimits::new(4, 5, 1))
            .unwrap();

        assert_eq!(execution.to_state.pc, expected_pc);
        assert_eq!(execution.to_state.timestamp, 28);
        assert_eq!(execution.transcript.replay_values.len(), 5);
        assert_eq!(
            execution.transcript.replay_values[4],
            expected_branch_replay_value
        );

        let gpu_program = Int256PreflightGpuTracegen::upload_postflight_program(
            &program,
            &config.system.memory_config,
            &vm.engine.device().device_ctx,
        )
        .unwrap();

        execution.transcript.replay_values[4] = 2;
        let error = Int256PreflightGpuTracegen::postflight(
            &vm,
            &gpu_program,
            &execution,
            execution.retired,
        )
        .err()
        .expect("a non-boolean branch replay value must fail before replay mutation");
        assert!(error.to_string().contains("code 306"), "{error}");

        execution.transcript.replay_values[4] = expected_branch_replay_value ^ 1;
        let error = Int256PreflightGpuTracegen::postflight(
            &vm,
            &gpu_program,
            &execution,
            execution.retired,
        )
        .err()
        .expect("a corrupt branch replay value must disagree with the checkpoint anchor");
        assert!(error.to_string().contains("code 307"), "{error}");

        execution.transcript.replay_values[4] = expected_branch_replay_value;
        let (transcript, replay_plan) = Int256PreflightGpuTracegen::postflight(
            &vm,
            &gpu_program,
            &execution,
            execution.retired,
        )
        .unwrap();
        assert_eq!(transcript.error_code().unwrap(), 0);
        assert_eq!(transcript.memory_log_host().unwrap().len(), 27);
        assert_eq!(
            transcript
                .program_log_host()
                .unwrap()
                .iter()
                .map(|event| (event.pc, event.timestamp))
                .collect::<Vec<_>>(),
            [
                (0, 1),
                (4, 3),
                (8, 18),
                (expected_pc, 28),
                (expected_pc, 28)
            ]
        );

        let tracegen =
            Int256PreflightGpuTracegen::new(gpu_program.program(), &transcript, &replay_plan);
        let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
        drop(replay_plan);
        drop(transcript);
        let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
        vm.engine.verify(&pk.get_vk(), &proof).unwrap();
    }
}
