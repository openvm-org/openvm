use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_instructions::{
    instruction::Instruction,
    riscv::{MEMORY_AS, REGISTER_AS},
    LocalOpcode, SystemOpcode, VmOpcode, PUBLIC_VALUES_AS,
};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rvr_state::{
    PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent, PREFLIGHT_WRITE_BIT,
};

use super::{
    testing::{
        build_memory_chronology_for_test as gpu_chronology_with_fields,
        empty_chronology_counts_for_test,
    },
    *,
};
use crate::arch::{
    preflight::encode_u8_block, rvr::PreflightEndpoint, Postflight,
    POSTFLIGHT_PREDECESSOR_INDEX_LIMIT,
};

fn configured_byte_lengths(config: &MemoryConfig) -> Vec<usize> {
    config
        .addr_spaces
        .iter()
        .map(|address_space| address_space.num_cells * address_space.layout.size())
        .collect()
}

#[test]
fn field_cells_are_restricted_to_deferral_address_space() {
    let mut config = MemoryConfig::default();
    assert!(validate_field_address_spaces(&config).is_ok());

    config.addr_spaces[PUBLIC_VALUES_AS as usize].layout = MemoryCellType::field32();
    assert!(matches!(
        validate_field_address_spaces(&config),
        Err(GpuPostflightError::InvalidMemoryConfig(_))
    ));
}

#[test]
fn initial_memory_must_match_every_configured_address_space() {
    let config = MemoryConfig::default();
    let mut byte_lengths = configured_byte_lengths(&config);
    assert!(validate_initial_memory_lengths(&config, &byte_lengths).is_ok());

    byte_lengths.pop();
    assert!(matches!(
        validate_initial_memory_lengths(&config, &byte_lengths),
        Err(GpuPostflightError::InvalidTranscript(_))
    ));

    let mut byte_lengths = configured_byte_lengths(&config);
    byte_lengths[PUBLIC_VALUES_AS as usize] -= 1;
    assert!(matches!(
        validate_initial_memory_lengths(&config, &byte_lengths),
        Err(GpuPostflightError::InvalidTranscript(_))
    ));
}

#[test]
fn history_write_masks_reject_unsupported_cell_types() {
    let mut config = MemoryConfig::default();
    config.addr_spaces[MEMORY_AS as usize].layout = MemoryCellType::U32;
    let mut history = PreflightHistory {
        program: vec![PreflightProgramEvent {
            pc: 0,
            timestamp: 1,
        }],
        ..Default::default()
    };
    history
        .memory
        .accesses
        .push(event_value(1, MEMORY_AS, 0, true, [0; 4]));

    assert!(matches!(
        validated_history_write_masks(&history, &config),
        Err(GpuPostflightError::InvalidTranscript(_))
    ));
}

fn event_value(
    timestamp: u32,
    address_space: u32,
    pointer: u32,
    is_write: bool,
    value: [u16; 4],
) -> PreflightMemoryEvent {
    PreflightMemoryEvent {
        timestamp,
        address_space_and_kind: address_space | if is_write { PREFLIGHT_WRITE_BIT } else { 0 },
        pointer,
        value,
    }
}

fn field_event(
    timestamp: u32,
    pointer: u32,
    is_write: bool,
    value_index: u32,
) -> PreflightMemoryEvent {
    PreflightMemoryEvent {
        timestamp,
        address_space_and_kind: DEFERRAL_AS | if is_write { PREFLIGHT_WRITE_BIT } else { 0 },
        pointer,
        value: [value_index as u16, (value_index >> 16) as u16, 0, 0],
    }
}

fn raw_baby_bear(value: BabyBear) -> u32 {
    // BabyBear and the CUDA `Fp` ABI are both one raw Montgomery u32.
    unsafe { std::mem::transmute(value) }
}

#[test]
fn empty_gpu_chronology_zeroes_every_counter() {
    assert_eq!(empty_chronology_counts_for_test(false).unwrap(), vec![0; 3]);
    assert_eq!(empty_chronology_counts_for_test(true).unwrap(), vec![0; 7]);
}

fn gpu_program(opcodes: &[u32], device_ctx: &GpuDeviceCtx) -> GpuPostflightProgram {
    GpuPostflightProgram::synthetic_for_test(
        opcodes,
        0,
        MemoryConfig::default().timestamp_max_bits as u32,
        device_ctx,
    )
    .unwrap()
}

fn gpu_plan(
    program: &GpuPostflightProgram,
    history: &PreflightHistory,
    endpoint: PreflightEndpoint,
) -> Result<GpuPostflightPlan, GpuPostflightError> {
    assert!(history.memory.accesses.is_empty());
    assert!(history.memory.initial_writes.is_empty());
    let first = history.program.first().unwrap();
    let last = history.program.last().unwrap();
    let boundary = GpuPostflightBoundary::new(
        ExecutionState::new(first.pc, first.timestamp),
        ExecutionState::new(last.pc, last.timestamp),
        matches!(endpoint, PreflightEndpoint::Terminated).then_some(0),
    );
    program
        .index_program_log_for_test(&history.program, boundary)
        .map(|(_, plan)| plan)
}

#[test]
fn empty_program_cannot_terminate_without_a_terminate_step() {
    let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
    let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
    let program = gpu_program(&[terminate], &device_ctx);
    let history = PreflightHistory {
        program: vec![PreflightProgramEvent {
            pc: 0,
            timestamp: 1,
        }],
        ..Default::default()
    };
    let state = ExecutionState::new(0u32, 1u32);
    let error = match program.index_program_log_for_test(
        &history.program,
        GpuPostflightBoundary::new(state, state, Some(0)),
    ) {
        Ok(_) => panic!("empty program must not terminate without a terminate step"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("code 115"));
}

#[test]
fn interpreter_history_uses_the_standard_gpu_indexes() {
    let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
    let opcode = 17;
    let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
    let program = gpu_program(&[opcode, terminate], &device_ctx);
    let first = [1u16, 2, 3, 4];
    let memory_read = [21u16, 22, 23, 24];
    let initial_second = [5u16, 6, 7, 8];
    let written_second = [9u16, 10, 11, 12];
    let history = PreflightHistory {
        program: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 4,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 4,
            },
        ],
        memory: openvm_circuit::arch::PreflightMemoryLog {
            accesses: vec![
                event_value(1, REGISTER_AS, 0, false, first),
                event_value(2, MEMORY_AS, 0, false, memory_read),
                event_value(3, REGISTER_AS, BLOCK_FE_WIDTH as u32, true, written_second),
            ],
            initial_writes: vec![PreflightInitialWrite {
                address_space: REGISTER_AS,
                pointer: BLOCK_FE_WIDTH as u32,
                initial_value: initial_second,
            }],
            field_values: vec![],
            field_initial_values: vec![],
        },
    };
    let mut initial_registers = Vec::new();
    for value in first.into_iter().chain(initial_second) {
        initial_registers.extend_from_slice(&value.to_le_bytes());
    }
    let initial_memory_values = memory_read
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let initial_memory = (0..MemoryConfig::default().addr_spaces.len())
        .map(|address_space| {
            let image = if address_space == REGISTER_AS as usize {
                initial_registers.as_slice()
            } else if address_space == MEMORY_AS as usize {
                initial_memory_values.as_slice()
            } else {
                &[]
            };
            upload(image, &device_ctx).unwrap()
        })
        .collect::<Vec<_>>();
    let initial_memory_views = initial_memory
        .iter()
        .map(|image| image.view())
        .collect::<Vec<_>>();

    let (transcript, plan) = program
        .upload_history_with_initial_memory_for_test(
            &history,
            GpuPostflightBoundary::new(
                ExecutionState::new(0u32, 1u32),
                ExecutionState::new(4u32, 4u32),
                Some(0),
            ),
            &initial_memory_views,
        )
        .unwrap();

    assert_eq!(
        transcript.initial_write_log_host().unwrap(),
        history.memory.initial_writes
    );
    assert_eq!(
        transcript.memory_predecessors_host().unwrap(),
        vec![0, 0, POSTFLIGHT_PREDECESSOR_INDEX_LIMIT]
    );
    assert_eq!(
        plan.opcode_range(VmOpcode::from_usize(opcode as usize))
            .len(),
        1
    );
    assert_eq!(
        plan.opcode_range(SystemOpcode::TERMINATE.global_opcode())
            .len(),
        1
    );
}

fn mixed_chronology_fixture() -> (MemoryConfig, Vec<Vec<u8>>) {
    let mut config = MemoryConfig::default();
    for address_space in &mut config.addr_spaces {
        address_space.num_cells = 0;
    }
    config.addr_spaces[MEMORY_AS as usize].num_cells = 8;
    config.addr_spaces[DEFERRAL_AS as usize].num_cells = 8;
    let mut images = config
        .addr_spaces
        .iter()
        .map(|space| {
            let cell_bytes = match space.layout {
                MemoryCellType::Null | MemoryCellType::U8 => 1,
                MemoryCellType::U16 => 2,
                MemoryCellType::U32 | MemoryCellType::FIELD32 => 4,
                MemoryCellType::F { size } => size as usize,
            };
            vec![0u8; space.num_cells * cell_bytes]
        })
        .collect::<Vec<_>>();
    for (index, value) in [1u16, 2, 3, 4].into_iter().enumerate() {
        images[MEMORY_AS as usize][2 * index..2 * index + 2].copy_from_slice(&value.to_le_bytes());
    }
    for (index, value) in [11u32, 12, 13, 14, 21, 22, 23, 24].into_iter().enumerate() {
        images[DEFERRAL_AS as usize][4 * index..4 * index + 4]
            .copy_from_slice(&raw_baby_bear(BabyBear::from_u32(value)).to_le_bytes());
    }
    (config, images)
}

#[test]
fn gpu_chronology_resolves_mixed_u16_and_field_blocks_with_one_predecessor_order() {
    let (config, initial_memory) = mixed_chronology_fixture();
    let memory = [
        field_event(1, 0, false, 0),
        event_value(2, MEMORY_AS, 0, true, [0x00aa, 0, 0, 0]),
        field_event(3, 0, true, 1),
        field_event(4, 0, false, 2),
        event_value(5, MEMORY_AS, 0, false, [0; 4]),
        field_event(6, 4, true, 3),
        field_event(7, 4, false, 4),
    ];
    let first_write = PreflightFieldBlock {
        values: [31, 32, 33, 34],
    };
    let second_write = PreflightFieldBlock {
        values: [41, 42, 43, 44],
    };
    let field_values = [
        PreflightFieldBlock::default(),
        first_write,
        PreflightFieldBlock::default(),
        second_write,
        PreflightFieldBlock::default(),
    ];
    let (resolved, seeds, resolved_fields, field_seeds, predecessors, touched) =
        gpu_chronology_with_fields(
            &memory,
            &[0, 0x01, 0xff, 0, 0, 0xff, 0],
            &field_values,
            &initial_memory,
            &config,
        )
        .unwrap();

    assert_eq!(
        predecessors,
        [
            0,
            POSTFLIGHT_PREDECESSOR_INDEX_LIMIT,
            1,
            3,
            2,
            POSTFLIGHT_PREDECESSOR_INDEX_LIMIT | 1,
            6,
        ]
    );
    assert_eq!(resolved[1].value, [0x00aa, 2, 3, 4]);
    assert_eq!(resolved[4].value, [0x00aa, 2, 3, 4]);
    assert_eq!(resolved_fields[0].values, [11, 12, 13, 14]);
    assert_eq!(resolved_fields[1], first_write);
    assert_eq!(resolved_fields[2], first_write);
    assert_eq!(resolved_fields[3], second_write);
    assert_eq!(resolved_fields[4], second_write);

    assert_eq!(seeds.len(), 2);
    assert_eq!(seeds[0].address_space, MEMORY_AS);
    assert_eq!(seeds[0].initial_value, [1, 2, 3, 4]);
    assert_eq!(seeds[1].address_space, DEFERRAL_AS);
    assert_eq!(seeds[1].initial_value, [0, 0, 0, 0]);
    assert_eq!(
        field_seeds,
        [PreflightFieldBlock {
            values: [21, 22, 23, 24]
        }]
    );

    assert_eq!(
        touched
            .iter()
            .map(|block| (block.address_space, block.ptr, block.timestamp))
            .collect::<Vec<_>>(),
        [(MEMORY_AS, 0, 5), (DEFERRAL_AS, 0, 4), (DEFERRAL_AS, 4, 7),]
    );
    assert_eq!(touched[0].values, [0x00aa, 2, 3, 4]);
    assert_eq!(touched[1].values, first_write.values);
    assert_eq!(touched[2].values, second_write.values);
    assert_eq!(
        touched
            .iter()
            .map(|block| block.is_dirty)
            .collect::<Vec<_>>(),
        [1, 1, 1]
    );
}

#[test]
fn gpu_chronology_keeps_narrow_u16_only_path() {
    let mut config = MemoryConfig {
        addr_space_height: 1,
        ..Default::default()
    };
    config.addr_spaces.truncate(3);
    config.addr_spaces[MEMORY_AS as usize].num_cells = 4;
    let mut initial_memory = vec![Vec::new(), Vec::new(), vec![0u8; 8]];
    for (index, value) in [1u16, 2, 3, 4].into_iter().enumerate() {
        initial_memory[MEMORY_AS as usize][2 * index..2 * index + 2]
            .copy_from_slice(&value.to_le_bytes());
    }
    let read = event_value(1, MEMORY_AS, 0, false, [0; 4]);
    let (resolved, seeds, field_values, field_seeds, predecessors, touched) =
        gpu_chronology_with_fields(&[read], &[0], &[], &initial_memory, &config).unwrap();

    assert_eq!(resolved[0].value, [1, 2, 3, 4]);
    assert!(seeds.is_empty());
    assert!(field_values.is_empty());
    assert!(field_seeds.is_empty());
    assert_eq!(predecessors, [0]);
    assert_eq!(touched.len(), 1);
    assert_eq!(touched[0].is_dirty, 0);

    let observed_read = event_value(1, MEMORY_AS, 0, false, [1, 2, 3, 4]);
    assert!(
        gpu_chronology_with_fields(&[observed_read], &[0], &[], &initial_memory, &config).is_ok()
    );
    let incorrect_read = event_value(1, MEMORY_AS, 0, false, [9, 2, 3, 4]);
    assert!(
        gpu_chronology_with_fields(&[incorrect_read], &[0], &[], &initial_memory, &config).is_err()
    );

    // Dirtiness records the write itself, even when the value is unchanged
    // and a later read is the block's final event.
    let write = event_value(1, MEMORY_AS, 0, true, [1, 2, 3, 4]);
    let read = event_value(2, MEMORY_AS, 0, false, [0; 4]);
    let (_, _, _, _, _, touched) =
        gpu_chronology_with_fields(&[write, read], &[0xff, 0], &[], &initial_memory, &config)
            .unwrap();
    assert_eq!(touched.len(), 1);
    assert_eq!(touched[0].is_dirty, 1);
}

#[test]
fn gpu_chronology_handles_u8_cell_blocks() {
    let mut config = MemoryConfig::default();
    for address_space in &mut config.addr_spaces {
        address_space.num_cells = 0;
    }
    config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 8;
    let mut initial_memory = config
        .addr_spaces
        .iter()
        .map(|space| vec![0u8; space.num_cells * space.layout.size()])
        .collect::<Vec<_>>();
    initial_memory[PUBLIC_VALUES_AS as usize].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);

    let write = event_value(
        1,
        PUBLIC_VALUES_AS,
        0,
        true,
        encode_u8_block([9, 10, 11, 12]),
    );
    let read = event_value(2, PUBLIC_VALUES_AS, 0, false, [0; 4]);
    let (resolved, seeds, fields, field_seeds, predecessors, touched) =
        gpu_chronology_with_fields(&[write, read], &[0x0f, 0], &[], &initial_memory, &config)
            .unwrap();

    assert_eq!(resolved[0].value, encode_u8_block([9, 10, 11, 12]));
    assert_eq!(resolved[1].value, encode_u8_block([9, 10, 11, 12]));
    assert_eq!(seeds[0].initial_value, encode_u8_block([1, 2, 3, 4]));
    assert!(fields.is_empty());
    assert!(field_seeds.is_empty());
    assert_eq!(predecessors, [POSTFLIGHT_PREDECESSOR_INDEX_LIMIT, 1]);
    assert_eq!(touched.len(), 1);
    assert_eq!(
        touched[0].values.map(|value| value.as_canonical_u32()),
        [9, 10, 11, 12]
    );

    let invalid_padding = event_value(1, PUBLIC_VALUES_AS, 0, true, [0, 0, 1, 0]);
    assert!(
        gpu_chronology_with_fields(&[invalid_padding], &[0x0f], &[], &initial_memory, &config,)
            .is_err()
    );
}

#[test]
fn gpu_chronology_rejects_partial_or_noncanonical_field_values() {
    let (config, initial_memory) = mixed_chronology_fixture();
    let write = field_event(1, 0, true, 0);
    let valid = [PreflightFieldBlock {
        values: [1, 2, 3, 4],
    }];

    assert!(
        gpu_chronology_with_fields(&[write], &[0x0f], &valid, &initial_memory, &config,).is_err()
    );

    let invalid = [PreflightFieldBlock {
        values: [BabyBear::ORDER_U32, 2, 3, 4],
    }];
    assert!(
        gpu_chronology_with_fields(&[write], &[0xff], &invalid, &initial_memory, &config,).is_err()
    );

    let malformed_reference = PreflightMemoryEvent {
        value: [0, 0, 1, 0],
        ..write
    };
    assert!(gpu_chronology_with_fields(
        &[malformed_reference],
        &[0xff],
        &valid,
        &initial_memory,
        &config,
    )
    .is_err());

    let out_of_bounds_reference = field_event(1, 0, true, 1);
    assert!(gpu_chronology_with_fields(
        &[out_of_bounds_reference],
        &[0xff],
        &valid,
        &initial_memory,
        &config,
    )
    .is_err());

    let nonzero_read = field_event(1, 0, false, 0);
    let observed = [PreflightFieldBlock {
        values: [11, 12, 13, 14],
    }];
    assert!(
        gpu_chronology_with_fields(&[nonzero_read], &[0], &observed, &initial_memory, &config,)
            .is_ok()
    );
    assert!(
        gpu_chronology_with_fields(&[nonzero_read], &[0], &valid, &initial_memory, &config,)
            .is_err()
    );

    let mut short_initial_memory = initial_memory.clone();
    short_initial_memory[DEFERRAL_AS as usize].truncate(8);
    assert!(
        gpu_chronology_with_fields(&[write], &[0xff], &valid, &short_initial_memory, &config,)
            .is_err()
    );

    let mut noncanonical_initial_memory = initial_memory.clone();
    noncanonical_initial_memory[DEFERRAL_AS as usize][0..4]
        .copy_from_slice(&BabyBear::ORDER_U32.to_le_bytes());
    assert!(gpu_chronology_with_fields(
        &[write],
        &[0xff],
        &valid,
        &noncanonical_initial_memory,
        &config,
    )
    .is_err());

    let mut wrong_field_space = config.clone();
    let wrong_address_space = DEFERRAL_AS + 1;
    wrong_field_space.addr_spaces[wrong_address_space as usize].num_cells = 4;
    let mut wrong_field_memory = initial_memory;
    wrong_field_memory[wrong_address_space as usize].resize(16, 0);
    let wrong_space_event = PreflightMemoryEvent {
        address_space_and_kind: wrong_address_space | PREFLIGHT_WRITE_BIT,
        ..write
    };
    assert!(gpu_chronology_with_fields(
        &[wrong_space_event],
        &[0xff],
        &valid,
        &wrong_field_memory,
        &wrong_field_space,
    )
    .is_err());
}

#[test]
fn gpu_program_rejects_memory_configs_outside_the_compact_key_abi() {
    let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
    let program = Program::from_instructions(&[]);
    let assert_invalid = |config: &MemoryConfig| {
        assert!(matches!(
            GpuPostflightProgram::upload(&program, config, &device_ctx),
            Err(GpuPostflightError::InvalidMemoryConfig(_))
        ));
    };

    let ordinary = MemoryConfig::default();
    let uploaded = GpuPostflightProgram::upload(&program, &ordinary, &device_ctx).unwrap();
    assert_eq!(
        uploaded.cell_pointer_max_bits(),
        ordinary.pointer_max_bits as u32
    );

    for pointer_max_bits in [1, 33] {
        let config = MemoryConfig {
            pointer_max_bits,
            ..MemoryConfig::default()
        };
        assert_invalid(&config);
    }

    let timestamp_too_wide = MemoryConfig {
        timestamp_max_bits: 32,
        ..MemoryConfig::default()
    };
    assert_invalid(&timestamp_too_wide);

    let label_too_wide = MemoryConfig {
        pointer_max_bits: 32,
        ..MemoryConfig::default()
    };
    assert_invalid(&label_too_wide);

    let mut malformed_layout = MemoryConfig::default();
    malformed_layout.addr_spaces.pop();
    assert_invalid(&malformed_layout);

    let mut maximum = MemoryConfig {
        addr_space_height: 2,
        pointer_max_bits: 32,
        ..MemoryConfig::default()
    };
    maximum
        .addr_spaces
        .truncate(ADDR_SPACE_OFFSET as usize + (1 << maximum.addr_space_height));
    GpuPostflightProgram::upload(&program, &maximum, &device_ctx).unwrap();
}

#[test]
fn gpu_program_index_matches_cpu_oracle_and_preserves_order() {
    let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
    let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
    let opcodes = [100, 200, terminate];
    let program = gpu_program(&opcodes, &device_ctx);
    let history = PreflightHistory {
        program: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 2,
            },
            PreflightProgramEvent {
                pc: 0,
                timestamp: 3,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 4,
            },
            PreflightProgramEvent {
                pc: 8,
                timestamp: 5,
            },
            PreflightProgramEvent {
                pc: 8,
                timestamp: 5,
            },
        ],
        ..Default::default()
    };
    let endpoint = PreflightEndpoint::Terminated;
    let cpu_program = Program::new_without_debug_infos(
        &[
            Instruction::from_usize(VmOpcode::from_usize(100), [0; 5]),
            Instruction::from_usize(VmOpcode::from_usize(200), [0; 5]),
            Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
        ],
        0,
    );
    let expected =
        Postflight::new(&cpu_program, &history, &MemoryConfig::default(), Some(0)).unwrap();
    let actual = gpu_plan(&program, &history, endpoint).unwrap();
    let actual_steps = actual.steps_host().unwrap();
    let expected_steps = expected
        .replay_steps_for_test()
        .map(|(program_index, memory_start)| [program_index, memory_start])
        .collect::<Vec<_>>();
    assert_eq!(actual_steps, expected_steps);
    for &opcode in &[100, 200, terminate] {
        assert_eq!(
            actual.opcode_range(VmOpcode::from_usize(opcode as usize)),
            expected
                .opcode_ranges_for_test()
                .get(&opcode)
                .cloned()
                .unwrap_or(0..0)
        );
    }
    let opcode_100 = actual.opcode_range(VmOpcode::from_usize(100));
    let opcode_200 = actual.opcode_range(VmOpcode::from_usize(200));
    assert_eq!(
        actual_steps[opcode_100]
            .iter()
            .map(|step| step[0])
            .collect::<Vec<_>>(),
        vec![0, 2]
    );
    assert_eq!(
        actual_steps[opcode_200]
            .iter()
            .map(|step| step[0])
            .collect::<Vec<_>>(),
        vec![1, 3]
    );
}

#[test]
fn gpu_program_frequencies_are_dense_and_exclude_the_sentinel() {
    let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
    let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
    let program = GpuPostflightProgram::synthetic_for_test(
        &[100, u32::MAX, 200, 300, terminate],
        0x100,
        MemoryConfig::default().timestamp_max_bits as u32,
        &device_ctx,
    )
    .unwrap();
    let history = PreflightHistory {
        program: vec![
            PreflightProgramEvent {
                pc: 0x100,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 0x108,
                timestamp: 2,
            },
            PreflightProgramEvent {
                pc: 0x100,
                timestamp: 3,
            },
            PreflightProgramEvent {
                pc: 0x110,
                timestamp: 4,
            },
            PreflightProgramEvent {
                pc: 0x110,
                timestamp: 4,
            },
        ],
        ..Default::default()
    };
    let plan = gpu_plan(&program, &history, PreflightEndpoint::Terminated).unwrap();
    assert_eq!(plan.program_frequencies_host().unwrap(), vec![2, 1, 0, 1]);
    assert_eq!(
        plan.connector_boundary_for_test(),
        GpuPostflightBoundary::new(
            ExecutionState::new(0x100u32, 1u32),
            ExecutionState::new(0x110u32, 4u32),
            Some(0),
        )
    );

    let suspended = PreflightHistory {
        program: vec![
            PreflightProgramEvent {
                pc: 0x100,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 0x108,
                timestamp: 2,
            },
        ],
        ..Default::default()
    };
    let plan = gpu_plan(&program, &suspended, PreflightEndpoint::Suspended).unwrap();
    assert_eq!(plan.program_frequencies_host().unwrap(), vec![1, 0, 0, 0]);
    assert_eq!(
        plan.connector_boundary_for_test(),
        GpuPostflightBoundary::new(
            ExecutionState::new(0x100u32, 1u32),
            ExecutionState::new(0x108u32, 2u32),
            None,
        )
    );

    let empty = PreflightHistory {
        program: vec![PreflightProgramEvent {
            pc: 0x100,
            timestamp: 1,
        }],
        ..Default::default()
    };
    let plan = gpu_plan(&program, &empty, PreflightEndpoint::Suspended).unwrap();
    assert_eq!(plan.program_frequencies_host().unwrap(), vec![0; 4]);
}

#[test]
fn gpu_program_frequency_input_rejects_invalid_program_counters() {
    let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
    let program = GpuPostflightProgram::synthetic_for_test(
        &[100, u32::MAX, 200],
        0x100,
        MemoryConfig::default().timestamp_max_bits as u32,
        &device_ctx,
    )
    .unwrap();
    for invalid_pc in [0xfc, 0x102, 0x104, 0x10c] {
        let history = PreflightHistory {
            program: vec![
                PreflightProgramEvent {
                    pc: invalid_pc,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: invalid_pc,
                    timestamp: 2,
                },
            ],
            ..Default::default()
        };
        assert!(gpu_plan(&program, &history, PreflightEndpoint::Suspended,).is_err());
    }
}

#[test]
fn gpu_program_index_accepts_an_empty_suspended_segment() {
    let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
    let program = gpu_program(&[100], &device_ctx);
    let history = PreflightHistory {
        program: vec![PreflightProgramEvent {
            pc: 0,
            timestamp: 1,
        }],
        ..Default::default()
    };
    let endpoint = PreflightEndpoint::Suspended;
    let plan = gpu_plan(&program, &history, endpoint).unwrap();
    assert!(plan.steps_host().unwrap().is_empty());
    assert_eq!(plan.executed_opcodes().count(), 0);
}

#[test]
fn gpu_program_index_rejects_the_timestamp_domain_limit() {
    let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
    let program = GpuPostflightProgram::synthetic_for_test(&[100], 0, 2, &device_ctx).unwrap();
    let history = |final_timestamp| PreflightHistory {
        program: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 0,
                timestamp: final_timestamp,
            },
        ],
        ..Default::default()
    };
    gpu_plan(&program, &history(3), PreflightEndpoint::Suspended).unwrap();
    assert!(gpu_plan(&program, &history(4), PreflightEndpoint::Suspended,).is_err());
}

#[test]
fn gpu_program_index_rejects_malformed_boundaries() {
    let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
    let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
    let program = gpu_program(&[100, terminate], &device_ctx);
    let history = |program| PreflightHistory {
        program,
        ..Default::default()
    };

    let undefined_pc = history(vec![
        PreflightProgramEvent {
            pc: 12,
            timestamp: 1,
        },
        PreflightProgramEvent {
            pc: 12,
            timestamp: 2,
        },
    ]);
    assert!(gpu_plan(&program, &undefined_pc, PreflightEndpoint::Suspended,).is_err());

    let missing_terminate = history(vec![
        PreflightProgramEvent {
            pc: 0,
            timestamp: 1,
        },
        PreflightProgramEvent {
            pc: 0,
            timestamp: 2,
        },
    ]);
    assert!(gpu_plan(&program, &missing_terminate, PreflightEndpoint::Terminated,).is_err());

    let timestamp_regression = history(vec![
        PreflightProgramEvent {
            pc: 0,
            timestamp: 2,
        },
        PreflightProgramEvent {
            pc: 4,
            timestamp: 1,
        },
    ]);
    assert!(gpu_plan(
        &program,
        &timestamp_regression,
        PreflightEndpoint::Terminated,
    )
    .is_err());
}
