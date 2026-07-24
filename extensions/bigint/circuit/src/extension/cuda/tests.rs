use openvm_bigint_transpiler::{
    Rv64BaseAlu256Opcode, Rv64BranchEqual256Opcode, Rv64BranchLessThan256Opcode,
    Rv64LessThan256Opcode, Rv64Mul256Opcode, Rv64Shift256Opcode,
};
use openvm_circuit::{
    arch::{
        rvr::{
            cuda::{RvrCheckpointAccessRegistry, RvrCheckpointAccessSpan},
            RvrCheckpointPreflightLimits,
        },
        VirtualMachine, VmExecutor,
    },
    utils::{test_gpu_engine, test_system_config},
};
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    instruction::Instruction,
    program::Program,
    riscv::{RV64_IMM_AS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_BYTES},
    LocalOpcode, SystemOpcode, VmOpcode,
};
use openvm_riscv_circuit::Rv64ImRvrGpuTracegen;
use openvm_riscv_transpiler::{
    BaseAluImmOpcode, BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode,
    MulOpcode, ShiftOpcode,
};
use openvm_stark_backend::{p3_field::PrimeField32, StarkEngine};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::{Int256Rv64GpuBuilder, Int256RvrGpuTracegen};
use crate::Int256Rv64Config;

type F = BabyBear;

const DST_PTR: u32 = 0x100;
const LHS_PTR: u32 = 0x200;
const RHS_PTR: u32 = 0x300;

fn reg(index: usize) -> usize {
    index * RV64_REGISTER_BYTES as usize
}

fn fixture(equal: bool) -> (Program<F>, VmExe<F>) {
    let instructions = [
        Instruction::<F>::from_usize(
            BaseAluImmOpcode::ADDI.global_opcode(),
            [
                reg(4),
                reg(0),
                7,
                RV64_REGISTER_AS as usize,
                RV64_IMM_AS as usize,
            ],
        ),
        Instruction::<F>::from_usize(
            Rv64BaseAlu256Opcode(BaseAluOpcode::ADD).global_opcode(),
            [
                reg(1),
                reg(2),
                reg(3),
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::<F>::from_usize(
            Rv64BranchEqual256Opcode(BranchEqualOpcode::BEQ).global_opcode(),
            [
                reg(2),
                reg(3),
                8,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::<F>::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
        Instruction::<F>::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
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
                .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(register) + offset) as u32), byte)),
        );
    }
    init_memory.extend(
        lhs.into_iter()
            .enumerate()
            .map(|(offset, byte)| ((RV64_MEMORY_AS, LHS_PTR + offset as u32), byte)),
    );
    init_memory.extend(
        rhs.into_iter()
            .enumerate()
            .map(|(offset, byte)| ((RV64_MEMORY_AS, RHS_PTR + offset as u32), byte)),
    );
    (
        program.clone(),
        VmExe::new(program).with_init_memory(init_memory),
    )
}

#[test]
fn int256_checkpoint_registry_rejects_native_opcode_collision() {
    let native_opcode = BaseAluImmOpcode::ADDI.global_opcode().as_usize() as u32;
    let span = RvrCheckpointAccessSpan::read_fixed(RV64_MEMORY_AS, 0, 1);
    let mut registry = RvrCheckpointAccessRegistry::default();
    registry
        .register(native_opcode, &[1], 0, 4, 5, &[span])
        .unwrap();
    let error = registry
        .validate_no_native_collisions(Rv64ImRvrGpuTracegen::checkpoint_opcode_bases())
        .unwrap_err();
    assert!(error.to_string().contains("both native"), "{error}");
}

#[derive(Clone, Copy)]
struct OpcodeCase {
    opcode: VmOpcode,
    expected_branch: Option<bool>,
}

fn all_opcode_fixture() -> (Vec<OpcodeCase>, Program<F>, VmExe<F>) {
    let cases = vec![
        OpcodeCase {
            opcode: Rv64BaseAlu256Opcode(BaseAluOpcode::ADD).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Rv64BaseAlu256Opcode(BaseAluOpcode::SUB).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Rv64BaseAlu256Opcode(BaseAluOpcode::XOR).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Rv64BaseAlu256Opcode(BaseAluOpcode::OR).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Rv64BaseAlu256Opcode(BaseAluOpcode::AND).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Rv64Shift256Opcode(ShiftOpcode::SLL).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Rv64Shift256Opcode(ShiftOpcode::SRL).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Rv64Shift256Opcode(ShiftOpcode::SRA).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Rv64LessThan256Opcode(LessThanOpcode::SLT).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Rv64LessThan256Opcode(LessThanOpcode::SLTU).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Rv64Mul256Opcode(MulOpcode::MUL).global_opcode(),
            expected_branch: None,
        },
        OpcodeCase {
            opcode: Rv64BranchEqual256Opcode(BranchEqualOpcode::BEQ).global_opcode(),
            expected_branch: Some(false),
        },
        OpcodeCase {
            opcode: Rv64BranchEqual256Opcode(BranchEqualOpcode::BNE).global_opcode(),
            expected_branch: Some(true),
        },
        OpcodeCase {
            opcode: Rv64BranchLessThan256Opcode(BranchLessThanOpcode::BLT).global_opcode(),
            expected_branch: Some(true),
        },
        OpcodeCase {
            opcode: Rv64BranchLessThan256Opcode(BranchLessThanOpcode::BLTU).global_opcode(),
            expected_branch: Some(false),
        },
        OpcodeCase {
            opcode: Rv64BranchLessThan256Opcode(BranchLessThanOpcode::BGE).global_opcode(),
            expected_branch: Some(false),
        },
        OpcodeCase {
            opcode: Rv64BranchLessThan256Opcode(BranchLessThanOpcode::BGEU).global_opcode(),
            expected_branch: Some(true),
        },
    ];
    assert_eq!(cases.len(), 17);

    // lhs is signed -2^255 but unsigned 2^255; rhs is zero. This makes the
    // signed and unsigned branch/comparison pairs take opposite boundary paths.
    let mut lhs = [0u8; 32];
    lhs[31] = 0x80;
    let rhs = [0u8; 32];
    let negative_offset = (F::ORDER_U32 - 4) as usize;
    let mut instructions = cases
        .iter()
        .map(|case| {
            if let Some(expected_branch) = case.expected_branch {
                Instruction::<F>::from_usize(
                    case.opcode,
                    [
                        reg(2),
                        reg(3),
                        if expected_branch { 4 } else { negative_offset },
                        RV64_REGISTER_AS as usize,
                        RV64_MEMORY_AS as usize,
                    ],
                )
            } else {
                Instruction::<F>::from_usize(
                    case.opcode,
                    [
                        reg(1),
                        reg(2),
                        reg(3),
                        RV64_REGISTER_AS as usize,
                        RV64_MEMORY_AS as usize,
                    ],
                )
            }
        })
        .collect::<Vec<_>>();
    instructions.push(Instruction::<F>::from_usize(
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
                .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(register) + offset) as u32), byte)),
        );
    }
    init_memory.extend(
        lhs.into_iter()
            .enumerate()
            .map(|(offset, byte)| ((RV64_MEMORY_AS, LHS_PTR + offset as u32), byte)),
    );
    init_memory.extend(
        rhs.into_iter()
            .enumerate()
            .map(|(offset, byte)| ((RV64_MEMORY_AS, RHS_PTR + offset as u32), byte)),
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
    let executor = VmExecutor::new(config.clone()).unwrap();
    let checkpoint = executor
        .rvr_experimental_checkpoint_preflight_instance(&exe, None)
        .unwrap();
    let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Int256Rv64GpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&state.memory);
    let execution = checkpoint
        .execute_from_state(
            state,
            RvrCheckpointPreflightLimits::new(cases.len() + 1, 50, 1),
        )
        .unwrap();

    assert_eq!(execution.to_state.pc, (cases.len() * 4) as u32);
    assert_eq!(execution.to_state.timestamp, 226);
    assert_eq!(execution.transcript.residuals.len(), 50);
    assert_eq!(&execution.transcript.residuals[44..], &[0, 1, 1, 0, 0, 1]);

    let gpu_program = Int256RvrGpuTracegen::upload_checkpoint_program(
        &program,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .unwrap();
    let (transcript, replay_plan) = Int256RvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .unwrap();
    assert_eq!(transcript.error_code().unwrap(), 0);
    assert_eq!(transcript.memory_log_host().unwrap().len(), 225);
    for case in &cases {
        assert_eq!(replay_plan.opcode_range(case.opcode).len(), 1);
    }

    let tracegen = Int256RvrGpuTracegen::new(&gpu_program, &transcript, &replay_plan);
    let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
    drop(replay_plan);
    drop(transcript);
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();
}

#[test]
fn mixed_rv64_int256_checkpoint_expansion_proves_both_branch_outcomes() {
    for (equal, expected_pc, expected_branch_residual) in [(false, 12, 0u64), (true, 16, 1u64)] {
        let (program, exe) = fixture(equal);
        let config = Int256Rv64Config {
            system: test_system_config(),
            ..Default::default()
        };
        let executor = VmExecutor::new(config.clone()).unwrap();
        let checkpoint = executor
            .rvr_experimental_checkpoint_preflight_instance(&exe, None)
            .unwrap();
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
            .execute_from_state(state, RvrCheckpointPreflightLimits::new(4, 5, 1))
            .unwrap();

        assert_eq!(execution.to_state.pc, expected_pc);
        assert_eq!(execution.to_state.timestamp, 28);
        assert_eq!(execution.transcript.residuals.len(), 5);
        assert_eq!(execution.transcript.residuals[4], expected_branch_residual);

        let gpu_program = Int256RvrGpuTracegen::upload_checkpoint_program(
            &program,
            &config.system.memory_config,
            &vm.engine.device().device_ctx,
        )
        .unwrap();

        execution.transcript.residuals[4] = 2;
        let error = Int256RvrGpuTracegen::expand_checkpoint_replay(
            &vm,
            &gpu_program,
            &execution,
            execution.retired,
        )
        .err()
        .expect("a non-boolean branch residual must fail before replay mutation");
        assert!(error.to_string().contains("code 306"), "{error}");

        execution.transcript.residuals[4] = expected_branch_residual ^ 1;
        let error = Int256RvrGpuTracegen::expand_checkpoint_replay(
            &vm,
            &gpu_program,
            &execution,
            execution.retired,
        )
        .err()
        .expect("a corrupt branch residual must disagree with the checkpoint anchor");
        assert!(error.to_string().contains("code 307"), "{error}");

        execution.transcript.residuals[4] = expected_branch_residual;
        let (transcript, replay_plan) = Int256RvrGpuTracegen::expand_checkpoint_replay(
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

        let tracegen = Int256RvrGpuTracegen::new(&gpu_program, &transcript, &replay_plan);
        let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
        drop(replay_plan);
        drop(transcript);
        let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
        vm.engine.verify(&pk.get_vk(), &proof).unwrap();
    }
}
