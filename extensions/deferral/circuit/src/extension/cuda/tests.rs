use std::sync::Arc;

use openvm_circuit::{
    arch::{
        rvr::{cuda::GpuRvrProgram, RvrPreflightEndpoint, RvrPreflightTranscript},
        Streams, VirtualMachine, VmState, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES,
    },
    utils::{test_gpu_engine, test_system_config},
};
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    exe::SparseMemoryImage,
    instruction::Instruction,
    program::Program,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, SystemOpcode,
};
use openvm_riscv_circuit::{Rv64I, Rv64Io, Rv64M};
use openvm_stark_backend::StarkEngine;
use openvm_stark_sdk::{config::baby_bear_poseidon2::DIGEST_SIZE, p3_baby_bear::BabyBear};
use rvr_state::{
    PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent, PREFLIGHT_WRITE_BIT,
};

use super::{DeferralRvrCoverage, DeferralRvrGpuTracegen, Rv64DeferralGpuBuilder};
use crate::{
    generate_deferral_results,
    poseidon2::deferral_poseidon2_chip,
    utils::{combine_output, COMMIT_NUM_BYTES},
    DeferralExtension, DeferralFn, RawDeferralResult, Rv64DeferralConfig,
};

type F = BabyBear;

fn block(bytes: &[u8]) -> [u16; BLOCK_FE_WIDTH] {
    std::array::from_fn(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
}

fn insert_bytes(memory: &mut SparseMemoryImage, address_space: u32, pointer: u32, bytes: &[u8]) {
    memory.extend(
        bytes
            .iter()
            .copied()
            .enumerate()
            .map(|(offset, byte)| ((address_space, pointer + offset as u32), byte)),
    );
}

#[test]
fn deferral_coverage_rejects_missing_and_duplicate_producers() {
    let missing = DeferralRvrCoverage::new().finish().unwrap_err();
    assert!(missing.to_string().contains("Output"), "{missing}");
    assert!(missing.to_string().contains("Count"), "{missing}");

    let mut coverage = DeferralRvrCoverage::new();
    DeferralRvrCoverage::claim(&mut coverage.pending_output, "Output").unwrap();
    let duplicate = DeferralRvrCoverage::claim(&mut coverage.pending_output, "Output").unwrap_err();
    assert!(
        duplicate.to_string().contains("duplicate Output"),
        "{duplicate}"
    );

    DeferralRvrCoverage::claim(&mut coverage.pending_call, "Call").unwrap();
    DeferralRvrCoverage::claim(&mut coverage.pending_poseidon2, "Poseidon2").unwrap();
    DeferralRvrCoverage::claim(&mut coverage.pending_count, "Count").unwrap();
    coverage.finish().unwrap();
}

#[test]
fn deferral_output_coordinator_proves_without_record_arenas_and_call_fails_closed() {
    let rd = 8u32;
    let rs = 16u32;
    let output_ptr = 0x100u32;
    let input_ptr = 0x200u32;
    let output_raw = (0..2 * DIGEST_SIZE)
        .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
        .collect::<Vec<_>>();
    let result = generate_deferral_results(
        vec![RawDeferralResult::new(
            vec![0; COMMIT_NUM_BYTES],
            output_raw.clone(),
        )],
        0,
        &deferral_poseidon2_chip::<F>(),
    )
    .pop()
    .unwrap();
    let output_commit: [u8; COMMIT_NUM_BYTES] = result.output_commit.try_into().unwrap();
    let output_key = combine_output(output_commit, (output_raw.len() as u64).to_le_bytes());
    let output = Instruction::from_usize(
        DeferralOpcode::OUTPUT.global_opcode(),
        [
            rd as usize,
            rs as usize,
            0,
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
        ],
    );
    let program = Program::from_instructions(&[
        output,
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
    ]);

    let mut init_memory = SparseMemoryImage::default();
    insert_bytes(
        &mut init_memory,
        RV64_REGISTER_AS,
        rd,
        &(output_ptr as u64).to_le_bytes(),
    );
    insert_bytes(
        &mut init_memory,
        RV64_REGISTER_AS,
        rs,
        &(input_ptr as u64).to_le_bytes(),
    );
    insert_bytes(&mut init_memory, RV64_MEMORY_AS, input_ptr, &output_key);

    let config = Rv64DeferralConfig {
        system: test_system_config(),
        rv64i: Rv64I,
        rv64m: Rv64M::default(),
        io: Rv64Io,
        deferral: DeferralExtension::new(
            vec![Arc::new(DeferralFn::new(|_| Vec::new()))],
            vec![[0; COMMIT_NUM_BYTES]],
        ),
    };
    let initial_state = VmState::initial(&config.system, &init_memory, 0, Streams::default());
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64DeferralGpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&initial_state.memory);
    let device_ctx = vm.engine.device().device_ctx.clone();
    let gpu_program =
        GpuRvrProgram::upload(&program, &config.system.memory_config, &device_ctx).unwrap();

    let mut memory_log = vec![
        PreflightMemoryEvent {
            timestamp: 1,
            address_space_and_kind: RV64_REGISTER_AS,
            pointer: rd / 2,
            value: block(&(output_ptr as u64).to_le_bytes()),
        },
        PreflightMemoryEvent {
            timestamp: 2,
            address_space_and_kind: RV64_REGISTER_AS,
            pointer: rs / 2,
            value: block(&(input_ptr as u64).to_le_bytes()),
        },
    ];
    memory_log.extend(output_key.chunks_exact(MEMORY_BLOCK_BYTES).enumerate().map(
        |(chunk_idx, chunk)| PreflightMemoryEvent {
            timestamp: 3 + chunk_idx as u32,
            address_space_and_kind: RV64_MEMORY_AS,
            pointer: input_ptr / 2 + (chunk_idx * BLOCK_FE_WIDTH) as u32,
            value: block(chunk),
        },
    ));
    memory_log.extend(output_raw.chunks_exact(MEMORY_BLOCK_BYTES).enumerate().map(
        |(chunk_idx, chunk)| PreflightMemoryEvent {
            timestamp: 8 + chunk_idx as u32,
            address_space_and_kind: RV64_MEMORY_AS | PREFLIGHT_WRITE_BIT,
            pointer: output_ptr / 2 + (chunk_idx * BLOCK_FE_WIDTH) as u32,
            value: block(chunk),
        },
    ));
    let final_timestamp = 8 + (output_raw.len() / MEMORY_BLOCK_BYTES) as u32;
    let transcript = RvrPreflightTranscript {
        program_log: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: final_timestamp,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: final_timestamp,
            },
        ],
        memory_log,
        initial_write_log: (0..output_raw.len() / MEMORY_BLOCK_BYTES)
            .map(|chunk_idx| PreflightInitialWrite {
                address_space: RV64_MEMORY_AS,
                pointer: output_ptr / 2 + (chunk_idx * BLOCK_FE_WIDTH) as u32,
                initial_value: [0; BLOCK_FE_WIDTH],
            })
            .collect(),
    };
    let (gpu_transcript, replay_plan) = gpu_program
        .upload_transcript(&transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let proving_ctx =
        DeferralRvrGpuTracegen::new(&gpu_program, &gpu_transcript, &replay_plan, 1 << 20)
            .unwrap()
            .generate_proving_ctx(&mut vm)
            .unwrap();
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();

    let call_program = Program::from_instructions(&[
        Instruction::<F>::from_usize(
            DeferralOpcode::CALL.global_opcode(),
            [
                rd as usize,
                rs as usize,
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
    ]);
    let call_gpu =
        GpuRvrProgram::upload(&call_program, &config.system.memory_config, &device_ctx).unwrap();
    let call_transcript = RvrPreflightTranscript {
        program_log: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 1,
            },
        ],
        memory_log: vec![],
        initial_write_log: vec![],
    };
    let (call_transcript, call_plan) = call_gpu
        .upload_transcript(&call_transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let error = DeferralRvrGpuTracegen::new(&call_gpu, &call_transcript, &call_plan, 1 << 20)
        .err()
        .expect("CALL must remain blocked until Phase-B AS4 production");
    assert!(error.to_string().contains("Phase-B AS4"), "{error}");
}
