use openvm_instructions::{
    instruction::InstructionOperand, program::Program, riscv::REGISTER_AS, SystemOpcode,
    DEFERRAL_AS, PUBLIC_VALUES_AS,
};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rvr_state::PREFLIGHT_WRITE_BIT;

use super::*;
use crate::arch::{
    preflight::encode_u8_block, PreflightFieldBlock, PreflightInitialWrite, PreflightMemoryEvent,
    PreflightMemoryLog, PreflightProgramEvent,
};

#[test]
fn fill_trace_rows_skips_empty_ranges() {
    let mut trace = RowMajorMatrix::<BabyBear>::new(Vec::new(), 1);
    fill_trace_rows::<_, ()>(&mut trace, usize::MAX, &[], |_, _| unreachable!()).unwrap();
}

#[test]
fn peek_uses_the_already_consumed_timed_event_prefix() {
    let instruction =
        Instruction::from_usize(SystemOpcode::PHANTOM.global_opcode(), [0, 0, 0, 0, 0]);
    let program = Program::new_without_debug_infos(&[instruction.clone(), instruction], 0);
    let history = PreflightHistory {
        program: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 3,
            },
        ],
        memory: PreflightMemoryLog {
            accesses: vec![
                PreflightMemoryEvent {
                    timestamp: 1,
                    address_space_and_kind: 1 | PREFLIGHT_WRITE_BIT,
                    pointer: 0,
                    value: [2, 0, 0, 0],
                },
                PreflightMemoryEvent {
                    timestamp: 2,
                    address_space_and_kind: 1 | PREFLIGHT_WRITE_BIT,
                    pointer: 0,
                    value: [3, 0, 0, 0],
                },
            ],
            initial_writes: vec![PreflightInitialWrite {
                address_space: 1,
                pointer: 0,
                initial_value: [1, 0, 0, 0],
            }],
            ..Default::default()
        },
    };
    let memory_config = MemoryConfig::default();
    let postflight = Postflight::<BabyBear>::new(&program, &history, &memory_config, None).unwrap();
    let step = postflight.steps(SystemOpcode::PHANTOM.global_opcode())[0];
    let mut replay = postflight.replay(step);

    assert_eq!(replay.peek_u16(1, 0).unwrap(), [1, 0, 0, 0]);
    assert_eq!(replay.write_observed_u16(1, 0).unwrap().value, [2, 0, 0, 0]);
    assert_eq!(replay.peek_u16(1, 0).unwrap(), [2, 0, 0, 0]);
    assert_eq!(replay.write_observed_u16(1, 0).unwrap().value, [3, 0, 0, 0]);
    assert_eq!(replay.peek_u16(1, 0).unwrap(), [3, 0, 0, 0]);
    replay.finish(4).unwrap();
}

#[test]
fn peek_before_a_first_read_uses_the_read_value() {
    let instruction =
        Instruction::from_usize(SystemOpcode::PHANTOM.global_opcode(), [0, 0, 0, 0, 0]);
    let program = Program::new_without_debug_infos(&[instruction.clone(), instruction], 0);
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
        ],
        memory: PreflightMemoryLog {
            accesses: vec![PreflightMemoryEvent {
                timestamp: 1,
                address_space_and_kind: 1,
                pointer: 0,
                value: [7, 0, 0, 0],
            }],
            ..Default::default()
        },
    };
    let memory_config = MemoryConfig::default();
    let postflight = Postflight::<BabyBear>::new(&program, &history, &memory_config, None).unwrap();
    let step = postflight.steps(SystemOpcode::PHANTOM.global_opcode())[0];
    let mut replay = postflight.replay(step);

    assert_eq!(replay.peek_u16(1, 0).unwrap(), [7, 0, 0, 0]);
    assert!(replay.peek_u16(1, 4).is_err());
    assert_eq!(replay.read_u16(1, 0).unwrap().value, [7, 0, 0, 0]);
    assert_eq!(replay.peek_u16(1, 0).unwrap(), [7, 0, 0, 0]);
    replay.finish(4).unwrap();
}

fn field_block(values: [u32; BLOCK_FE_WIDTH]) -> PreflightFieldBlock {
    PreflightFieldBlock { values }
}

#[test]
fn field_sidecars_are_canonical_not_field_representations() {
    let canonical = [1, 2, 3, BabyBear::ORDER_U32 - 1];
    let block = field_block(canonical);
    assert_eq!(block.values, canonical);
    assert_eq!(
        decode_field_block::<BabyBear>(block).map(|value| value.as_canonical_u32()),
        canonical
    );
}

fn compact_reference(index: u32) -> [u16; BLOCK_FE_WIDTH] {
    [index as u16, (index >> 16) as u16, 0, 0]
}

fn mixed_history() -> (Program, PreflightHistory) {
    let instruction =
        Instruction::from_usize(SystemOpcode::PHANTOM.global_opcode(), [0, 0, 0, 0, 0]);
    let program = Program::new_without_debug_infos(&[instruction.clone(), instruction], 0);
    let history = PreflightHistory {
        program: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 5,
            },
        ],
        memory: PreflightMemoryLog {
            accesses: vec![
                PreflightMemoryEvent {
                    timestamp: 1,
                    address_space_and_kind: REGISTER_AS,
                    pointer: 0,
                    value: [1, 2, 3, 4],
                },
                PreflightMemoryEvent {
                    timestamp: 2,
                    address_space_and_kind: REGISTER_AS | PREFLIGHT_WRITE_BIT,
                    pointer: 0,
                    value: [5, 6, 7, 8],
                },
                PreflightMemoryEvent {
                    timestamp: 3,
                    address_space_and_kind: DEFERRAL_AS | PREFLIGHT_WRITE_BIT,
                    pointer: 0,
                    value: compact_reference(0),
                },
                PreflightMemoryEvent {
                    timestamp: 4,
                    address_space_and_kind: DEFERRAL_AS,
                    pointer: 0,
                    value: compact_reference(1),
                },
            ],
            initial_writes: vec![PreflightInitialWrite {
                address_space: DEFERRAL_AS,
                pointer: 0,
                initial_value: compact_reference(0),
            }],
            field_values: vec![field_block([21, 22, 23, 24]), field_block([31, 32, 33, 34])],
            field_initial_values: vec![field_block([11, 12, 13, 14])],
        },
    };
    (program, history)
}

#[test]
fn u8_history_replays_packed_event_and_seed() {
    let instruction =
        Instruction::from_usize(SystemOpcode::PHANTOM.global_opcode(), [0, 0, 0, 0, 0]);
    let program = Program::new_without_debug_infos(&[instruction.clone(), instruction], 0);
    let initial = [1, 2, 3, 4];
    let written = [5, 6, 7, 8];
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
        ],
        memory: PreflightMemoryLog {
            accesses: vec![PreflightMemoryEvent {
                timestamp: 1,
                address_space_and_kind: PUBLIC_VALUES_AS | PREFLIGHT_WRITE_BIT,
                pointer: 0,
                value: encode_u8_block(written),
            }],
            initial_writes: vec![PreflightInitialWrite {
                address_space: PUBLIC_VALUES_AS,
                pointer: 0,
                initial_value: encode_u8_block(initial),
            }],
            ..Default::default()
        },
    };

    let postflight =
        Postflight::<BabyBear>::new(&program, &history, &MemoryConfig::default(), None).unwrap();
    let step = postflight.steps(SystemOpcode::PHANTOM.global_opcode())[0];
    let mut replay = postflight.replay(step);
    let access = replay.write_u8(PUBLIC_VALUES_AS, 0, written).unwrap();
    assert_eq!(access.previous_value, initial);
    replay.finish(4).unwrap();

    let touched = &postflight.touched_memory()[0];
    assert_eq!(touched.address_space, PUBLIC_VALUES_AS);
    assert_eq!(
        touched.values.map(|value| value.as_canonical_u32()),
        written.map(u32::from)
    );
}

#[test]
fn rejects_invalid_program_boundaries() {
    let memory_config = MemoryConfig::default();
    let (program, mut history) = mixed_history();
    history.program[0].pc = 2;
    assert!(
        Postflight::<BabyBear>::new(&program, &history, &memory_config, None)
            .err()
            .unwrap()
            .to_string()
            .contains("instruction-aligned")
    );

    let terminate =
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]);
    let phantom = Instruction::from_usize(SystemOpcode::PHANTOM.global_opcode(), [0, 0, 0, 0, 0]);
    let program = Program::new_without_debug_infos(&[terminate, phantom], 0);
    let history = PreflightHistory {
        program: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 1,
            },
        ],
        ..Default::default()
    };
    assert!(
        Postflight::<BabyBear>::new(&program, &history, &memory_config, Some(0))
            .err()
            .unwrap()
            .to_string()
            .contains("duplicate the sentinel")
    );
}

#[test]
fn derives_boundary_frequencies_and_mixed_touched_memory() {
    let (program, history) = mixed_history();
    let memory_config = MemoryConfig::default();
    let postflight = Postflight::<BabyBear>::new(&program, &history, &memory_config, None).unwrap();

    assert_eq!(postflight.from_state(), ExecutionState::new(0u32, 1u32));
    assert_eq!(postflight.to_state(), ExecutionState::new(4u32, 5u32));
    assert_eq!(postflight.exit_code(), None);
    assert_eq!(postflight.filtered_exec_frequencies(), [1, 0]);
    assert_eq!(
        postflight
            .touched_memory()
            .iter()
            .map(|block| (
                block.address_space,
                block.ptr,
                block.is_dirty,
                block.timestamp
            ))
            .collect::<Vec<_>>(),
        [(REGISTER_AS, 0, 1, 2), (DEFERRAL_AS, 0, 1, 4),]
    );
    assert_eq!(
        postflight.touched_memory()[0]
            .values
            .map(|value| value.as_canonical_u32()),
        [5, 6, 7, 8]
    );
    assert_eq!(
        postflight.touched_memory()[1]
            .values
            .map(|value| value.as_canonical_u32()),
        [31, 32, 33, 34]
    );

    let step = postflight.steps(SystemOpcode::PHANTOM.global_opcode())[0];
    let mut replay = postflight.replay(step);
    let read = replay.read_u16(REGISTER_AS, 0).unwrap();
    assert_eq!(read.value, [1, 2, 3, 4]);
    assert_eq!(read.previous_value, read.value);
    let write = replay.write_u16(REGISTER_AS, 0, [5, 6, 7, 8]).unwrap();
    assert_eq!(write.previous_value, [1, 2, 3, 4]);
    let field_write = replay
        .write_field32(DEFERRAL_AS, 0, [21, 22, 23, 24].map(BabyBear::from_u32))
        .unwrap();
    assert_eq!(
        field_write
            .previous_value
            .map(|value| value.as_canonical_u32()),
        [11, 12, 13, 14]
    );
    let field_read = replay.read_field32(DEFERRAL_AS, 0).unwrap();
    assert_eq!(
        field_read
            .previous_value
            .map(|value| value.as_canonical_u32()),
        [21, 22, 23, 24]
    );
    replay.finish(4).unwrap();
}

#[test]
fn retains_a_terminated_boundary_and_frequency() {
    let exit_code = 7;
    let terminate = Instruction::from_usize(
        SystemOpcode::TERMINATE.global_opcode(),
        [0, 0, exit_code as usize, 0, 0],
    );
    let program = Program::new_without_debug_infos(&[terminate], 0);
    let boundary = PreflightProgramEvent {
        pc: 0,
        timestamp: 1,
    };
    let history = PreflightHistory {
        program: vec![boundary, boundary],
        ..Default::default()
    };
    let memory_config = MemoryConfig::default();
    let postflight =
        Postflight::<BabyBear>::new(&program, &history, &memory_config, Some(exit_code)).unwrap();

    assert_eq!(postflight.from_state(), ExecutionState::new(0u32, 1u32));
    assert_eq!(postflight.to_state(), ExecutionState::new(0u32, 1u32));
    assert_eq!(postflight.exit_code(), Some(exit_code));
    assert_eq!(postflight.filtered_exec_frequencies(), [1]);
}

#[test]
fn rejects_misaligned_program_base_and_pc() {
    let terminate =
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]);
    let program = Program::new_without_debug_infos(&[terminate], 2);
    let boundary = PreflightProgramEvent {
        pc: 2,
        timestamp: 1,
    };
    let history = PreflightHistory {
        program: vec![boundary, boundary],
        ..Default::default()
    };

    let error = Postflight::<BabyBear>::new(&program, &history, &MemoryConfig::default(), Some(0))
        .err()
        .unwrap();
    assert!(error.to_string().contains("is not instruction-aligned"));
}

#[test]
fn rejects_negative_terminate_exit_code() {
    let terminate = Instruction {
        opcode: SystemOpcode::TERMINATE.global_opcode(),
        c: InstructionOperand::from_i32(-1),
        ..Default::default()
    };
    let program = Program::new_without_debug_infos(&[terminate], 0);
    let boundary = PreflightProgramEvent {
        pc: 0,
        timestamp: 1,
    };
    let history = PreflightHistory {
        program: vec![boundary, boundary],
        ..Default::default()
    };

    let error =
        Postflight::<BabyBear>::new(&program, &history, &MemoryConfig::default(), Some(u32::MAX))
            .err()
            .unwrap();
    assert!(error.to_string().contains("must be non-negative"));
}

#[cfg(feature = "metrics")]
#[test]
fn derives_opcode_counts_from_validated_history() {
    let phantom = Instruction::from_usize(SystemOpcode::PHANTOM.global_opcode(), [0, 0, 0, 0, 0]);
    let program = Program::new_without_debug_infos(&[phantom.clone(), phantom.clone(), phantom], 0);
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
                pc: 8,
                timestamp: 3,
            },
        ],
        ..Default::default()
    };
    let postflight =
        Postflight::<BabyBear>::new(&program, &history, &MemoryConfig::default(), None).unwrap();

    assert_eq!(
        postflight.executed_opcodes().collect::<Vec<_>>(),
        [SystemOpcode::PHANTOM.global_opcode()]
    );
    assert_eq!(
        postflight.opcode_count(SystemOpcode::PHANTOM.global_opcode()),
        2
    );
}

#[test]
fn rejects_invalid_memory_domains_and_field_sidecars() {
    let memory_config = MemoryConfig::default();

    let (program, mut history) = mixed_history();
    history.memory.accesses[0].pointer = 1;
    assert!(
        Postflight::<BabyBear>::new(&program, &history, &memory_config, None)
            .err()
            .unwrap()
            .to_string()
            .contains("misaligned")
    );

    let (program, mut history) = mixed_history();
    history.memory.accesses[0].address_space_and_kind = PUBLIC_VALUES_AS;
    history.memory.accesses[0].value = encode_u8_block([1, 2, 3, 4]);
    history.memory.accesses[0].value[2] = 1;
    assert!(
        Postflight::<BabyBear>::new(&program, &history, &memory_config, None)
            .err()
            .unwrap()
            .to_string()
            .contains("u8 memory event has nonzero padding")
    );

    let (program, mut history) = mixed_history();
    history.memory.initial_writes[0].address_space = PUBLIC_VALUES_AS;
    history.memory.initial_writes[0].initial_value = encode_u8_block([1, 2, 3, 4]);
    history.memory.initial_writes[0].initial_value[3] = 1;
    assert!(
        Postflight::<BabyBear>::new(&program, &history, &memory_config, None)
            .err()
            .unwrap()
            .to_string()
            .contains("u8 initial-write seed has nonzero padding")
    );

    let (program, mut history) = mixed_history();
    history.memory.accesses[2].value = compact_reference(1);
    assert!(
        Postflight::<BabyBear>::new(&program, &history, &memory_config, None)
            .err()
            .unwrap()
            .to_string()
            .contains("dense ordered")
    );

    let (program, mut history) = mixed_history();
    history.memory.field_values[0].values[0] = BabyBear::ORDER_U32;
    assert!(
        Postflight::<BabyBear>::new(&program, &history, &memory_config, None)
            .err()
            .unwrap()
            .to_string()
            .contains("non-canonical")
    );

    let (program, mut history) = mixed_history();
    history.program[1].timestamp = 1 << memory_config.timestamp_max_bits;
    assert!(
        Postflight::<BabyBear>::new(&program, &history, &memory_config, None)
            .err()
            .unwrap()
            .to_string()
            .contains("timestamp exceeds")
    );

    let (program, history) = mixed_history();
    let postflight = Postflight::<BabyBear>::new(&program, &history, &memory_config, None).unwrap();
    let step = postflight.steps(SystemOpcode::PHANTOM.global_opcode())[0];
    assert!(postflight
        .replay(step)
        .read_u16(DEFERRAL_AS, 0)
        .err()
        .unwrap()
        .to_string()
        .contains("wrong cell layout"));
}

#[test]
fn rejects_invalid_memory_chronology() {
    let read = |timestamp, pointer| PreflightMemoryEvent {
        timestamp,
        address_space_and_kind: REGISTER_AS,
        pointer,
        value: [0; BLOCK_FE_WIDTH],
    };
    let write = |timestamp, pointer| PreflightMemoryEvent {
        address_space_and_kind: REGISTER_AS | PREFLIGHT_WRITE_BIT,
        ..read(timestamp, pointer)
    };
    let seed = PreflightInitialWrite {
        address_space: REGISTER_AS,
        pointer: 0,
        initial_value: [0; BLOCK_FE_WIDTH],
    };
    let history = |accesses, initial_writes| PreflightHistory {
        memory: PreflightMemoryLog {
            accesses,
            initial_writes,
            ..Default::default()
        },
        ..Default::default()
    };
    let error = |history: &PreflightHistory, config: &MemoryConfig| {
        memory_index::<BabyBear>(history, config)
            .unwrap_err()
            .to_string()
    };
    let config = MemoryConfig::default();

    assert!(error(&history(vec![write(1, 0)], vec![]), &config).contains("without a seed"));
    assert!(error(&history(vec![write(1, 0)], vec![seed, seed]), &config).contains("duplicate"));
    assert!(error(&history(vec![read(1, 0)], vec![seed]), &config).contains("not referenced"));

    let invalid_seed = PreflightInitialWrite {
        address_space: REGISTER_AS | PREFLIGHT_WRITE_BIT,
        ..seed
    };
    assert!(
        error(&history(vec![write(1, 0)], vec![invalid_seed]), &config)
            .contains("contains the write bit")
    );

    let invalid_address_space = PreflightMemoryEvent {
        address_space_and_kind: 0,
        ..read(1, 0)
    };
    assert!(error(&history(vec![invalid_address_space], vec![]), &config).contains("out of range"));

    let (program, mut history) = mixed_history();
    history.memory.accesses[1].timestamp = history.memory.accesses[0].timestamp;
    assert!(
        Postflight::<BabyBear>::new(&program, &history, &config, None)
            .err()
            .unwrap()
            .to_string()
            .contains("not strictly increasing")
    );
}
