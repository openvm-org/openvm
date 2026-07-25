use openvm_circuit::{
    arch::{
        rvr::{
            cuda::{GpuRvrProgram, RvrCheckpointAccessRegistry, RvrCheckpointAccessSpan},
            RvrCheckpointPreflightLimits, RvrPreflightEndpoint, RvrPreflightTranscript,
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
    LocalOpcode, SystemOpcode,
};
use openvm_riscv_circuit::Rv64ImRvrGpuTracegen;
use openvm_riscv_transpiler::BaseAluImmOpcode;
use openvm_sha2_transpiler::Rv64Sha2Opcode;
use openvm_stark_backend::StarkEngine;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rvr_state::{
    PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent, PREFLIGHT_WRITE_BIT,
};
use sha2::{compress256, compress512, digest::generic_array::GenericArray};

use super::{Sha2Rv64GpuBuilder, Sha2RvrGpuTracegen};
use crate::Sha2Rv64Config;

type F = BabyBear;

const DST_PTR: u32 = 0x1000;
const STATE_PTR: u32 = 0x2000;
const INPUT_PTR: u32 = 0x3000;

fn reg(index: usize) -> u32 {
    (index * RV64_REGISTER_BYTES as usize) as u32
}

fn limbs(bytes: &[u8]) -> [u16; 4] {
    std::array::from_fn(|i| u16::from_le_bytes(bytes[2 * i..2 * i + 2].try_into().unwrap()))
}

fn event(
    timestamp: u32,
    address_space: u32,
    byte_pointer: u32,
    is_write: bool,
    bytes: &[u8],
) -> PreflightMemoryEvent {
    PreflightMemoryEvent {
        timestamp,
        address_space_and_kind: address_space | if is_write { PREFLIGHT_WRITE_BIT } else { 0 },
        pointer: byte_pointer / 2,
        value: limbs(bytes),
    }
}

fn seed(address_space: u32, byte_pointer: u32, bytes: &[u8]) -> PreflightInitialWrite {
    PreflightInitialWrite {
        address_space,
        pointer: byte_pointer / 2,
        initial_value: limbs(bytes),
    }
}

fn append_sha_events(
    memory: &mut Vec<PreflightMemoryEvent>,
    from_timestamp: u32,
    block_bytes: &[u8],
    state_bytes: &[u8],
    result_bytes: &[u8],
) {
    let mut timestamp = from_timestamp;
    for (register, pointer) in [(1, DST_PTR), (2, STATE_PTR), (3, INPUT_PTR)] {
        memory.push(event(
            timestamp,
            RV64_REGISTER_AS,
            reg(register),
            false,
            &u64::from(pointer).to_le_bytes(),
        ));
        timestamp += 1;
    }
    for (index, bytes) in block_bytes.chunks_exact(8).enumerate() {
        memory.push(event(
            timestamp,
            RV64_MEMORY_AS,
            INPUT_PTR + (index * 8) as u32,
            false,
            bytes,
        ));
        timestamp += 1;
    }
    for (index, bytes) in state_bytes.chunks_exact(8).enumerate() {
        memory.push(event(
            timestamp,
            RV64_MEMORY_AS,
            STATE_PTR + (index * 8) as u32,
            false,
            bytes,
        ));
        timestamp += 1;
    }
    for (index, bytes) in result_bytes.chunks_exact(8).enumerate() {
        memory.push(event(
            timestamp,
            RV64_MEMORY_AS,
            DST_PTR + (index * 8) as u32,
            true,
            bytes,
        ));
        timestamp += 1;
    }
}

fn sha_results(state: &[u8; 64], input: &[u8; 128]) -> ([u8; 32], [u8; 64]) {
    let mut state256 = std::array::from_fn::<_, 8, _>(|i| {
        u32::from_le_bytes(state[4 * i..4 * i + 4].try_into().unwrap())
    });
    let block256 = GenericArray::clone_from_slice(&input[..64]);
    compress256(&mut state256, &[block256]);
    let mut result256 = [0u8; 32];
    for (dst, word) in result256.chunks_exact_mut(4).zip(state256) {
        dst.copy_from_slice(&word.to_le_bytes());
    }

    let mut state512 = std::array::from_fn::<_, 8, _>(|i| {
        u64::from_le_bytes(state[8 * i..8 * i + 8].try_into().unwrap())
    });
    let block512 = GenericArray::clone_from_slice(input);
    compress512(&mut state512, &[block512]);
    let mut result512 = [0u8; 64];
    for (dst, word) in result512.chunks_exact_mut(8).zip(state512) {
        dst.copy_from_slice(&word.to_le_bytes());
    }
    (result256, result512)
}

fn fixture(corrupt_sha256_register_event: bool) -> (Program<F>, VmExe<F>, RvrPreflightTranscript) {
    let instructions = [
        Instruction::<F>::from_usize(
            BaseAluImmOpcode::ADDI.global_opcode(),
            [
                reg(4) as usize,
                reg(0) as usize,
                7,
                RV64_REGISTER_AS as usize,
                RV64_IMM_AS as usize,
            ],
        ),
        Instruction::<F>::from_usize(
            Rv64Sha2Opcode::SHA256.global_opcode(),
            [
                reg(1) as usize,
                reg(2) as usize,
                reg(3) as usize,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::<F>::from_usize(
            Rv64Sha2Opcode::SHA512.global_opcode(),
            [
                reg(1) as usize,
                reg(2) as usize,
                reg(3) as usize,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::<F>::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
    ];
    let program = Program::from_instructions(&instructions);
    let state = std::array::from_fn::<_, 64, _>(|i| (i as u8).wrapping_mul(11).wrapping_add(5));
    let input = std::array::from_fn::<_, 128, _>(|i| (i as u8).wrapping_mul(17).wrapping_add(3));
    let (result256, result512) = sha_results(&state, &input);

    let mut init_memory = SparseMemoryImage::default();
    for (register, pointer) in [(1, DST_PTR), (2, STATE_PTR), (3, INPUT_PTR)] {
        init_memory.extend(
            u64::from(pointer)
                .to_le_bytes()
                .into_iter()
                .enumerate()
                .map(|(offset, byte)| ((RV64_REGISTER_AS, reg(register) + offset as u32), byte)),
        );
    }
    init_memory.extend(
        state
            .iter()
            .copied()
            .enumerate()
            .map(|(offset, byte)| ((RV64_MEMORY_AS, STATE_PTR + offset as u32), byte)),
    );
    init_memory.extend(
        input
            .iter()
            .copied()
            .enumerate()
            .map(|(offset, byte)| ((RV64_MEMORY_AS, INPUT_PTR + offset as u32), byte)),
    );
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory);

    let mut memory_log = vec![
        event(1, RV64_REGISTER_AS, reg(0), false, &[0; 8]),
        event(2, RV64_REGISTER_AS, reg(4), true, &7u64.to_le_bytes()),
    ];
    append_sha_events(&mut memory_log, 3, &input[..64], &state[..32], &result256);
    append_sha_events(&mut memory_log, 22, &input, &state, &result512);
    if corrupt_sha256_register_event {
        memory_log[2].pointer += 4;
    }

    // Initial-write seeds exist only for blocks whose first timed event is a write.
    // First reads resolve directly against the segment's initial memory.
    let mut initial_write_log = vec![seed(RV64_REGISTER_AS, reg(4), &[0; 8])];
    for index in 0..8 {
        initial_write_log.push(seed(RV64_MEMORY_AS, DST_PTR + index * 8, &[0; 8]));
    }

    let transcript = RvrPreflightTranscript {
        program_log: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 3,
            },
            PreflightProgramEvent {
                pc: 8,
                timestamp: 22,
            },
            PreflightProgramEvent {
                pc: 12,
                timestamp: 57,
            },
            PreflightProgramEvent {
                pc: 12,
                timestamp: 57,
            },
        ],
        memory_log,
        initial_write_log,
    };
    (program, exe, transcript)
}

#[test]
fn sha_checkpoint_registry_rejects_native_opcode_collision() {
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

#[test]
fn mixed_rv64_sha_checkpoint_expansion_proves() {
    let (program, exe, _) = fixture(false);
    let config = Sha2Rv64Config {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let checkpoint = executor.checkpoint_preflight_instance(&exe, None).unwrap();
    let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Sha2Rv64GpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&state.memory);
    let execution = checkpoint
        .execute_from_state(state, RvrCheckpointPreflightLimits::new(4, 12, 1))
        .unwrap();
    assert_eq!(execution.to_state.timestamp, 57);
    assert_eq!(execution.transcript.residuals.len(), 12);
    let gpu_program = Sha2RvrGpuTracegen::upload_checkpoint_program(
        &program,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .unwrap();
    let (gpu_transcript, replay_plan) = Sha2RvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .unwrap();
    let tracegen = Sha2RvrGpuTracegen::new(&gpu_program, &gpu_transcript, &replay_plan);
    let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
    drop(replay_plan);
    drop(gpu_transcript);
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();
}

#[test]
fn mixed_rv64_sha_manual_transcript_rejects_corruption() {
    let (program, exe, corrupt) = fixture(true);
    let config = Sha2Rv64Config {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let state = executor
        .interpreter_instance(&exe)
        .unwrap()
        .create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut vm, _) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Sha2Rv64GpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&state.memory);
    let gpu_program = GpuRvrProgram::upload(
        &program,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .unwrap();
    let (gpu_corrupt, corrupt_plan) = gpu_program
        .upload_transcript(&corrupt, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let error = Sha2RvrGpuTracegen::new(&gpu_program, &gpu_corrupt, &corrupt_plan)
        .generate_proving_ctx(&mut vm)
        .err()
        .expect("corrupt SHA register event must fail closed");
    assert!(error.to_string().contains("code 901"), "{error}");

    let retry = Sha2RvrGpuTracegen::new(&gpu_program, &gpu_corrupt, &corrupt_plan)
        .generate_proving_ctx(&mut vm)
        .err()
        .expect("a VM with partially updated lookup counts must reject retry");
    assert!(retry.to_string().contains("poisoned"), "{retry}");
}

#[test]
fn mixed_rv64_sha_manual_transcript_rejects_corrupt_outputs() {
    let (program, exe, _) = fixture(false);
    let config = Sha2Rv64Config {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let state = executor
        .interpreter_instance(&exe)
        .unwrap()
        .create_initial_vm_state(Vec::<Vec<u8>>::new());

    for (variant, first_write_timestamp) in [("SHA-256", 18), ("SHA-512", 49)] {
        let (_, _, mut corrupt) = fixture(false);
        let output = corrupt
            .memory_log
            .iter_mut()
            .find(|event| {
                event.timestamp == first_write_timestamp
                    && event.address_space_and_kind == (RV64_MEMORY_AS | PREFLIGHT_WRITE_BIT)
                    && event.pointer == DST_PTR / 2
            })
            .expect("fixture must contain the first deterministic output write");
        output.value[0] ^= 1;

        let (mut vm, _) =
            VirtualMachine::new_with_keygen(test_gpu_engine(), Sha2Rv64GpuBuilder, config.clone())
                .unwrap();
        let cached_program = vm.commit_program_on_device(&program);
        vm.load_program(cached_program);
        vm.transport_init_memory_to_device(&state.memory);
        let gpu_program = GpuRvrProgram::upload(
            &program,
            &config.system.memory_config,
            &vm.engine.device().device_ctx,
        )
        .unwrap();
        let (gpu_corrupt, corrupt_plan) = gpu_program
            .upload_transcript(&corrupt, RvrPreflightEndpoint::Terminated)
            .unwrap_or_else(|error| panic!("{variant}: {error}"));
        let error = Sha2RvrGpuTracegen::new(&gpu_program, &gpu_corrupt, &corrupt_plan)
            .generate_proving_ctx(&mut vm)
            .err()
            .unwrap_or_else(|| panic!("{variant} corrupt output must fail closed"));
        assert!(error.to_string().contains("code 901"), "{variant}: {error}");
    }
}

#[test]
fn sha_coordinator_requires_both_producers_per_executed_opcode() {
    let (program, _, transcript) = fixture(false);
    let config = Sha2Rv64Config {
        system: test_system_config(),
        ..Default::default()
    };
    let engine = test_gpu_engine();
    let gpu_program = GpuRvrProgram::upload(
        &program,
        &config.system.memory_config,
        &engine.device().device_ctx,
    )
    .unwrap();
    let (gpu_transcript, replay_plan) = gpu_program
        .upload_transcript(&transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let error = Sha2RvrGpuTracegen::new(&gpu_program, &gpu_transcript, &replay_plan)
        .finish()
        .expect_err("unvisited SHA producers must fail closed");
    let message = error.to_string();
    for producer in [
        "Sha256Main",
        "Sha256BlockHasher",
        "Sha512Main",
        "Sha512BlockHasher",
    ] {
        assert!(message.contains(producer), "{message}");
    }
}
