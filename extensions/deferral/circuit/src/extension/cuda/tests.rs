use std::sync::Arc;

use openvm_circuit::{
    arch::{
        deferral::DeferralState, rvr::RvrCheckpointPreflightLimits, Streams, VirtualMachine,
        VmExecutor, BLOCK_FE_WIDTH,
    },
    system::memory::online::LinearMemory,
    utils::{test_gpu_engine, test_system_config},
};
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    instruction::Instruction,
    program::Program,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, SystemOpcode, DEFERRAL_AS,
};
use openvm_riscv_circuit::{Rv64I, Rv64Io, Rv64M};
use openvm_stark_backend::{p3_field::PrimeField32, StarkEngine};
use openvm_stark_sdk::{config::baby_bear_poseidon2::DIGEST_SIZE, p3_baby_bear::BabyBear};

use super::{DeferralRvrCoverage, DeferralRvrGpuTracegen, Rv64DeferralGpuBuilder};
use crate::{
    generate_deferral_results,
    poseidon2::deferral_poseidon2_chip,
    utils::{combine_output, COMMIT_NUM_BYTES},
    DeferralExtension, DeferralFn, RawDeferralResult, Rv64DeferralConfig,
};

type F = BabyBear;

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
fn deferral_output_coordinator_proves_without_record_arenas() {
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
    let output_commit: [u8; COMMIT_NUM_BYTES] = result.output_commit.clone().try_into().unwrap();
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
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
    let executor = VmExecutor::new(config.clone()).unwrap();
    let checkpoint = executor.checkpoint_preflight_instance(&exe, None).unwrap();
    let initial_state = checkpoint.create_initial_vm_state(Streams {
        deferrals: vec![DeferralState::new(vec![result])],
        ..Default::default()
    });
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64DeferralGpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&initial_state.memory);
    let mut execution = checkpoint
        .execute_from_state(initial_state, RvrCheckpointPreflightLimits::new(2, 3, 1))
        .unwrap();
    assert_eq!(execution.retired, 2);
    assert_eq!(execution.to_state.timestamp, 10);
    assert_eq!(
        execution.transcript.residuals,
        vec![
            2,
            u64::from_le_bytes(output_raw[..DIGEST_SIZE].try_into().unwrap()),
            u64::from_le_bytes(output_raw[DIGEST_SIZE..].try_into().unwrap()),
        ]
    );
    assert_eq!(
        &execution.state.memory.memory.mem[RV64_MEMORY_AS as usize].as_slice()
            [output_ptr as usize..output_ptr as usize + output_raw.len()],
        output_raw
    );

    let gpu_program = DeferralRvrGpuTracegen::upload_checkpoint_program(
        &program,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .unwrap();
    let missing = execution.transcript.residuals.pop().unwrap();
    let error = DeferralRvrGpuTracegen::expand_checkpoint_replay(&vm, &gpu_program, &execution, 2)
        .err()
        .expect("missing OUTPUT residual must be rejected");
    assert!(error.to_string().contains("code 306"), "{error}");
    execution.transcript.residuals.push(missing);

    let (gpu_transcript, replay_plan) =
        DeferralRvrGpuTracegen::expand_checkpoint_replay(&vm, &gpu_program, &execution, 2).unwrap();
    assert_eq!(
        gpu_transcript
            .program_log_host()
            .unwrap()
            .iter()
            .map(|event| (event.pc, event.timestamp))
            .collect::<Vec<_>>(),
        [(0, 1), (4, 10), (4, 10)]
    );
    let memory = gpu_transcript.memory_log_host().unwrap();
    assert_eq!(memory.len(), 9);
    let expected = [
        (1, RV64_REGISTER_AS, rd / 2, false),
        (2, RV64_REGISTER_AS, rs / 2, false),
        (3, RV64_MEMORY_AS, input_ptr / 2, false),
        (4, RV64_MEMORY_AS, input_ptr / 2 + 4, false),
        (5, RV64_MEMORY_AS, input_ptr / 2 + 8, false),
        (6, RV64_MEMORY_AS, input_ptr / 2 + 12, false),
        (7, RV64_MEMORY_AS, input_ptr / 2 + 16, false),
        (8, RV64_MEMORY_AS, output_ptr / 2, true),
        (9, RV64_MEMORY_AS, output_ptr / 2 + 4, true),
    ];
    for (event, &(timestamp, address_space, pointer, is_write)) in memory.iter().zip(&expected) {
        assert_eq!(event.timestamp, timestamp);
        assert_eq!(event.address_space(), address_space);
        assert_eq!(event.pointer, pointer);
        assert_eq!(event.is_write(), is_write);
    }

    let proving_ctx =
        DeferralRvrGpuTracegen::new(&gpu_program, &gpu_transcript, &replay_plan, 1 << 20)
            .unwrap()
            .generate_proving_ctx(&mut vm)
            .unwrap();
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();
}

#[test]
fn deferral_call_checkpoint_expands_exact_as4_chronology_and_proves_without_records() {
    let rd = 8u32;
    let rs = 16u32;
    let output_ptr = 0x100u32;
    let input_ptr = 0x200u32;
    let input_commit = [0u8; COMMIT_NUM_BYTES];
    let output_raw = vec![7u8; 2 * DIGEST_SIZE];
    let call = Instruction::from_usize(
        DeferralOpcode::CALL.global_opcode(),
        [
            rd as usize,
            rs as usize,
            0,
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
        ],
    );
    let program = Program::from_instructions(&[
        call,
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
    insert_bytes(&mut init_memory, RV64_MEMORY_AS, input_ptr, &input_commit);

    let mut system = test_system_config();
    system.memory_config.addr_spaces[DEFERRAL_AS as usize].num_cells = 1 << 20;
    let config = Rv64DeferralConfig {
        system,
        rv64i: Rv64I,
        rv64m: Rv64M::default(),
        io: Rv64Io,
        deferral: DeferralExtension::new(
            vec![Arc::new(DeferralFn::new({
                let output_raw = output_raw.clone();
                move |_| output_raw.clone()
            }))],
            vec![[0; COMMIT_NUM_BYTES]],
        ),
    };
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
    let mut deferral = DeferralState::default();
    deferral.store_input(input_commit.to_vec(), vec![3u8; DIGEST_SIZE]);
    let streams = Streams {
        deferrals: vec![deferral],
        ..Default::default()
    };
    let executor = VmExecutor::new(config.clone()).unwrap();
    let checkpoint = executor.checkpoint_preflight_instance(&exe, None).unwrap();
    let state = checkpoint.create_initial_vm_state(streams);
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64DeferralGpuBuilder, config.clone())
            .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&state.memory);
    let mut execution = checkpoint
        .execute_from_state(state, RvrCheckpointPreflightLimits::new(2, 13, 1))
        .unwrap();
    assert_eq!(execution.retired, 2);
    assert_eq!(execution.to_state.timestamp, 20);
    assert_eq!(execution.transcript.residuals.len(), 13);

    let gpu_program = DeferralRvrGpuTracegen::upload_checkpoint_program(
        &program,
        &config.system.memory_config,
        &vm.engine.device().device_ctx,
    )
    .unwrap();

    let original = execution.transcript.residuals[5];
    execution.transcript.residuals[5] = u64::from(F::ORDER_U32) << 32;
    let error = DeferralRvrGpuTracegen::expand_checkpoint_replay(&vm, &gpu_program, &execution, 2)
        .err()
        .expect("non-canonical CALL residual must be rejected");
    assert!(error.to_string().contains("code 306"), "{error}");
    execution.transcript.residuals[5] = original;

    let missing = execution.transcript.residuals.pop().unwrap();
    let error = DeferralRvrGpuTracegen::expand_checkpoint_replay(&vm, &gpu_program, &execution, 2)
        .err()
        .expect("missing CALL residual must be rejected");
    assert!(error.to_string().contains("code 306"), "{error}");
    execution.transcript.residuals.push(missing);

    let (transcript, replay_plan) =
        DeferralRvrGpuTracegen::expand_checkpoint_replay(&vm, &gpu_program, &execution, 2).unwrap();
    let program_log = transcript.program_log_host().unwrap();
    assert_eq!(
        program_log
            .iter()
            .map(|event| (event.pc, event.timestamp))
            .collect::<Vec<_>>(),
        [(0, 1), (4, 20), (4, 20)]
    );
    let memory = transcript.memory_log_host().unwrap();
    assert_eq!(memory.len(), 19);
    let expected = [
        (1, RV64_REGISTER_AS, rd / 2, false),
        (2, RV64_REGISTER_AS, rs / 2, false),
        (3, RV64_MEMORY_AS, input_ptr / 2, false),
        (4, RV64_MEMORY_AS, input_ptr / 2 + 4, false),
        (5, RV64_MEMORY_AS, input_ptr / 2 + 8, false),
        (6, RV64_MEMORY_AS, input_ptr / 2 + 12, false),
        (7, DEFERRAL_AS, 0, false),
        (8, DEFERRAL_AS, 4, false),
        (9, DEFERRAL_AS, 8, false),
        (10, DEFERRAL_AS, 12, false),
        (11, RV64_MEMORY_AS, output_ptr / 2, true),
        (12, RV64_MEMORY_AS, output_ptr / 2 + 4, true),
        (13, RV64_MEMORY_AS, output_ptr / 2 + 8, true),
        (14, RV64_MEMORY_AS, output_ptr / 2 + 12, true),
        (15, RV64_MEMORY_AS, output_ptr / 2 + 16, true),
        (16, DEFERRAL_AS, 0, true),
        (17, DEFERRAL_AS, 4, true),
        (18, DEFERRAL_AS, 8, true),
        (19, DEFERRAL_AS, 12, true),
    ];
    for (event, &(timestamp, address_space, pointer, is_write)) in memory.iter().zip(&expected) {
        assert_eq!(event.timestamp, timestamp);
        assert_eq!(event.address_space(), address_space);
        assert_eq!(event.pointer, pointer);
        assert_eq!(event.is_write(), is_write);
    }
    let field_values = transcript.field_values_host().unwrap();
    assert_eq!(field_values.len(), 8);
    for (reference, event) in memory[6..10].iter().chain(&memory[15..19]).enumerate() {
        assert_eq!(
            event.value,
            [reference as u16, (reference >> 16) as u16, 0, 0]
        );
    }
    assert!(field_values[..4]
        .iter()
        .all(|block| block.values == [0; BLOCK_FE_WIDTH]));
    assert!(field_values[4..]
        .iter()
        .any(|block| block.values != [0; BLOCK_FE_WIDTH]));
    assert_eq!(
        replay_plan
            .opcode_range(DeferralOpcode::CALL.global_opcode())
            .len(),
        1
    );

    let proving_ctx = DeferralRvrGpuTracegen::new(&gpu_program, &transcript, &replay_plan, 1 << 20)
        .unwrap()
        .generate_proving_ctx(&mut vm)
        .unwrap();
    drop(replay_plan);
    drop(transcript);
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();
}
