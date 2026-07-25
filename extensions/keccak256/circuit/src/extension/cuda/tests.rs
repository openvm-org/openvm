use openvm_circuit::{
    arch::{
        rvr::{
            cuda::{GpuRvrProgram, RvrCheckpointAccessRegistry, RvrCheckpointAccessSpan},
            FullLogPreflightTranscript, PreflightEndpoint, PreflightLimits,
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
use openvm_keccak256_transpiler::{KeccakfOpcode, XorinOpcode};
use openvm_riscv_circuit::Rv64ImRvrGpuTracegen;
use openvm_riscv_transpiler::BaseAluImmOpcode;
use openvm_stark_backend::StarkEngine;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rvr_state::PreflightProgramEvent;

use super::{Keccak256Rv64GpuBuilder, Keccak256RvrGpuTracegen};
use crate::Keccak256Rv64Config;

type F = BabyBear;

fn reg(index: usize) -> usize {
    index * RV64_REGISTER_BYTES as usize
}

#[test]
fn checkpoint_access_registry_rejects_duplicate_and_invalid_schedules() {
    let opcode = KeccakfOpcode::KECCAKF.global_opcode().as_usize() as u32;
    let span = RvrCheckpointAccessSpan::write_fixed_from_residuals(RV64_MEMORY_AS, 0, 25);
    let mut registry = RvrCheckpointAccessRegistry::default();
    registry.register(opcode, &[1], 0, 4, 5, &[span]).unwrap();
    let duplicate = registry
        .register(opcode, &[1], 0, 4, 5, &[span])
        .unwrap_err();
    assert!(duplicate.to_string().contains("duplicate"), "{duplicate}");

    let invalid_span = RvrCheckpointAccessSpan::read_fixed(RV64_MEMORY_AS, 1, 1);
    let invalid = RvrCheckpointAccessRegistry::default()
        .register(opcode, &[1], 0, 4, 5, &[invalid_span])
        .unwrap_err();
    assert!(
        invalid.to_string().contains("invalid access span"),
        "{invalid}"
    );

    let invalid_mask = RvrCheckpointAccessRegistry::default()
        .register(opcode, &[1], 1, 4, 5, &[span])
        .unwrap_err();
    assert!(
        invalid_mask.to_string().contains("invalid operand layout"),
        "{invalid_mask}"
    );
    let oversized = RvrCheckpointAccessRegistry::default()
        .register(u32::MAX, &[1], 0, 4, 5, &[span])
        .unwrap_err();
    assert!(
        oversized.to_string().contains("dense checkpoint dispatch"),
        "{oversized}"
    );

    let native_opcode = BaseAluImmOpcode::ADDI.global_opcode().as_usize() as u32;
    let mut collision = RvrCheckpointAccessRegistry::default();
    collision
        .register(native_opcode, &[1], 0, 4, 5, &[span])
        .unwrap();
    let collision = collision
        .validate_no_native_collisions(Rv64ImRvrGpuTracegen::checkpoint_opcode_bases())
        .unwrap_err();
    assert!(collision.to_string().contains("both native"), "{collision}");
}

#[test]
fn checkpoint_replay_expands_keccak_schedules_and_rejects_missing_residuals() {
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
            XorinOpcode::XORIN.global_opcode(),
            [
                reg(1),
                reg(2),
                reg(3),
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::<F>::from_usize(
            KeccakfOpcode::KECCAKF.global_opcode(),
            [
                reg(1),
                0,
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::<F>::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let buffer_ptr = 64u64;
    let xorin_len = 136u64;
    // test_system_config has 2^22 u16 AS2 cells, hence 2^23 bytes.
    // Put the maximum-length input exactly at the byte-domain upper boundary.
    let input_ptr = (1u64 << 23) - xorin_len;
    let mut init_memory: SparseMemoryImage = [(1usize, buffer_ptr), (2, input_ptr), (3, xorin_len)]
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
    init_memory.extend((0..200u32).map(|offset| {
        (
            (RV64_MEMORY_AS, buffer_ptr as u32 + offset),
            offset.wrapping_mul(17) as u8,
        )
    }));
    init_memory.extend((0..xorin_len as u32).map(|offset| {
        (
            (RV64_MEMORY_AS, input_ptr as u32 + offset),
            offset.wrapping_mul(29).wrapping_add(3) as u8,
        )
    }));
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
    let config = Keccak256Rv64Config {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let checkpoint = executor.preflight_instance(&exe, None).unwrap();
    let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Keccak256Rv64GpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&state.memory);
    let mut execution = checkpoint
        .execute_from_state(state, PreflightLimits::new(instructions.len(), 42, 1))
        .unwrap();

    // ADDI: one register read and one write. XORIN: 3 register reads +
    // 2 * (read, read, write). KECCAKF: one register read + 25 writes.
    // TERMINATE consumes no clock slot.
    assert_eq!(execution.to_state.timestamp, 83);
    assert_eq!(execution.transcript.residuals.len(), 42);

    let malformed = Program::from_instructions(&[
        Instruction::<F>::from_usize(
            XorinOpcode::XORIN.global_opcode(),
            [
                reg(1) + 1,
                reg(2),
                reg(3),
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::<F>::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
    ]);
    let malformed = Keccak256RvrGpuTracegen::upload_checkpoint_program(
        &malformed,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .err()
    .expect("an instruction incompatible with the Keccak schedule must be rejected");
    assert!(
        malformed
            .to_string()
            .contains("incompatible with its access schedule"),
        "{malformed}"
    );

    let unclaimed_program = GpuRvrProgram::upload(
        &program,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .unwrap();
    let unclaimed = Keccak256RvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &unclaimed_program,
        &execution,
        execution.retired,
    )
    .err()
    .expect("expansion without Keccak access schedules must fail");
    assert!(unclaimed.to_string().contains("code 303"), "{unclaimed}");

    let gpu_program = Keccak256RvrGpuTracegen::upload_checkpoint_program(
        &program,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .unwrap();
    let missing = execution.transcript.residuals.pop().unwrap();
    let error = Keccak256RvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .err()
    .expect("missing Keccak residual must fail checkpoint replay");
    assert!(error.to_string().contains("code 306"), "{error}");
    execution.transcript.residuals.push(missing);

    let (transcript, replay_plan) = Keccak256RvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .unwrap();
    assert!(
        Rv64ImRvrGpuTracegen::new(&gpu_program, &transcript, &replay_plan).is_err(),
        "the plain RV64 coordinator must not silently claim Keccak opcodes"
    );
    assert_eq!(transcript.error_code().unwrap(), 0);
    assert_eq!(
        replay_plan
            .opcode_range(XorinOpcode::XORIN.global_opcode())
            .len(),
        1
    );
    assert_eq!(
        replay_plan
            .opcode_range(KeccakfOpcode::KECCAKF.global_opcode())
            .len(),
        1
    );

    let program_log = transcript.program_log_host().unwrap();
    assert_eq!(
        program_log
            .iter()
            .map(|event| event.timestamp)
            .collect::<Vec<_>>(),
        [1, 3, 57, 83, 83]
    );
    let memory_log = transcript.memory_log_host().unwrap();
    assert_eq!(memory_log.len(), 82);
    assert_eq!(
        memory_log
            .iter()
            .map(|event| event.timestamp)
            .collect::<Vec<_>>(),
        (1..83).collect::<Vec<_>>()
    );

    let tracegen = Keccak256RvrGpuTracegen::new(&gpu_program, &transcript, &replay_plan);
    let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
    drop(replay_plan);
    drop(transcript);
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();

    let mut invalid_state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    // The fixture register lies within the configured RV64 register space.
    unsafe {
        invalid_state
            .memory
            .write_bytes(RV64_REGISTER_AS, reg(3) as u32, 7u64.to_le_bytes());
    }
    let invalid = checkpoint.execute_from_state(
        invalid_state,
        PreflightLimits::new(instructions.len(), 26, 1),
    );
    assert!(
        invalid.is_err(),
        "an unprovable partial XORIN block must fail before mutating memory"
    );

    let zero_program = Program::from_instructions(&[
        Instruction::<F>::from_usize(
            XorinOpcode::XORIN.global_opcode(),
            [
                reg(1),
                reg(2),
                reg(3),
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::<F>::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
    ]);
    let zero_memory: SparseMemoryImage = [(1usize, 3u64), (2, 5), (3, 0)]
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
    let zero_exe = VmExe::new(zero_program.clone()).with_init_memory(zero_memory);
    let zero_checkpoint = executor.preflight_instance(&zero_exe, None).unwrap();
    let zero_state = zero_checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let zero_cached_program = vm.commit_program_on_device(&zero_program);
    vm.load_program(zero_cached_program);
    vm.transport_init_memory_to_device(&zero_state.memory);
    let zero_execution = zero_checkpoint
        .execute_from_state(zero_state, PreflightLimits::new(2, 0, 1))
        .unwrap();
    assert_eq!(zero_execution.to_state.timestamp, 4);
    assert!(zero_execution.transcript.residuals.is_empty());
    let zero_gpu_program = Keccak256RvrGpuTracegen::upload_checkpoint_program(
        &zero_program,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .unwrap();
    let (zero_transcript, zero_plan) = Keccak256RvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &zero_gpu_program,
        &zero_execution,
        zero_execution.retired,
    )
    .unwrap();
    assert_eq!(zero_transcript.memory_log_host().unwrap().len(), 3);
    let zero_ctx = Keccak256RvrGpuTracegen::new(&zero_gpu_program, &zero_transcript, &zero_plan)
        .generate_proving_ctx(&mut vm)
        .unwrap();
    drop(zero_plan);
    drop(zero_transcript);
    let zero_proof = vm.engine.prove(vm.pk(), zero_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &zero_proof).unwrap();
}

#[test]
fn combined_keccak_coordinator_rejects_an_unclaimed_opcode() {
    let unknown_opcode = 0x00ff_0000usize;
    let instructions = [
        Instruction::<F>::from_usize(VmOpcode::from_usize(unknown_opcode), [0; 5]),
        Instruction::<F>::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
    ];
    let program = Program::from_instructions(&instructions);
    let transcript = FullLogPreflightTranscript {
        program_log: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 2,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 2,
            },
        ],
        memory_log: vec![],
        initial_write_log: vec![],
    };
    let memory_config = openvm_circuit::arch::MemoryConfig::default();
    let engine = test_gpu_engine();
    let device_ctx = &engine.device().device_ctx;
    let gpu_program = GpuRvrProgram::upload(&program, &memory_config, device_ctx).unwrap();
    let (gpu_transcript, replay_plan) = gpu_program
        .upload_transcript(&transcript, PreflightEndpoint::Terminated)
        .unwrap();
    let claimed = [
        XorinOpcode::XORIN.global_opcode().as_usize() as u32,
        KeccakfOpcode::KECCAKF.global_opcode().as_usize() as u32,
    ];
    let error = Rv64ImRvrGpuTracegen::new_after_claiming_extension_opcodes(
        &gpu_program,
        &gpu_transcript,
        &replay_plan,
        &claimed,
    )
    .err()
    .expect("an opcode owned by neither RV64 nor Keccak must fail closed");
    assert!(error.to_string().contains("does not support"), "{error}");
}
