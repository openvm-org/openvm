//! Small derived indexes for replaying a preflight transcript.

use std::collections::{hash_map::Entry, BTreeMap};
#[cfg(test)]
use std::ops::Range;

#[cfg(test)]
use openvm_instructions::{program::DEFAULT_PC_STEP, LocalOpcode, SystemOpcode};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use p3_baby_bear::BabyBear;
use rustc_hash::FxHashMap;
#[cfg(test)]
use rvr_state::PreflightProgramEvent;
use rvr_state::{PreflightInitialWrite, PreflightMemoryEvent, PREFLIGHT_WRITE_BIT};
use thiserror::Error;

#[cfg(test)]
use super::{cuda::RvrReplayStep, PreflightEndpoint, PreflightEventLog};
use crate::{
    arch::{
        postflight::memory_key, MemoryCellType, MemoryConfig, ADDR_SPACE_OFFSET, BLOCK_FE_WIDTH,
    },
    system::TouchedBlock,
};

/// No earlier timed event exists for this block. This is valid for a first read,
/// whose logged value is the segment's initial value.
pub(crate) const MEMORY_PREDECESSOR_BASELINE: u32 = 0;
/// The high bit distinguishes an initial-write seed index from an event index.
pub(crate) const MEMORY_PREDECESSOR_SEED_BIT: u32 = 1 << 31;
const MEMORY_PREDECESSOR_INDEX_MASK: u32 = !MEMORY_PREDECESSOR_SEED_BIT;

#[derive(Debug, Error)]
#[error("invalid RVR preflight transcript: {0}")]
pub(crate) struct PreflightIndexError(String);

impl PreflightIndexError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

#[cfg(test)]
fn build_step_memory_starts(
    program: &[PreflightProgramEvent],
    memory: &[PreflightMemoryEvent],
) -> Result<Vec<u32>, PreflightIndexError> {
    if program.is_empty() {
        return Err(PreflightIndexError::new(
            "transcript must contain a final sentinel",
        ));
    }
    if program[0].timestamp != 1 {
        return Err(PreflightIndexError::new(
            "segment transcript must start at timestamp 1",
        ));
    }
    let mut memory_cursor = 0usize;
    let mut previous_memory_timestamp = None;
    let mut step_memory_starts = Vec::with_capacity(program.len() - 1);
    for (program_index, boundary) in program.windows(2).enumerate() {
        let [from, to] = boundary else { unreachable!() };
        if to.timestamp < from.timestamp {
            return Err(PreflightIndexError::new(format!(
                "program timestamp moved backwards at step {program_index}"
            )));
        }
        if memory
            .get(memory_cursor)
            .is_some_and(|event| event.timestamp < from.timestamp)
        {
            return Err(PreflightIndexError::new(format!(
                "memory event {memory_cursor} precedes step {program_index}"
            )));
        }

        step_memory_starts.push(
            u32::try_from(memory_cursor).map_err(|_| {
                PreflightIndexError::new("memory log has more than u32::MAX entries")
            })?,
        );
        while memory
            .get(memory_cursor)
            .is_some_and(|event| event.timestamp < to.timestamp)
        {
            let timestamp = memory[memory_cursor].timestamp;
            if previous_memory_timestamp.is_some_and(|previous| previous >= timestamp) {
                return Err(PreflightIndexError::new(
                    "memory timestamps are not strictly increasing",
                ));
            }
            previous_memory_timestamp = Some(timestamp);
            memory_cursor += 1;
        }
    }
    if memory_cursor != memory.len() {
        return Err(PreflightIndexError::new(format!(
            "{} memory events occur at or after the final sentinel",
            memory.len() - memory_cursor
        )));
    }
    Ok(step_memory_starts)
}

/// Cold derived replay data built once from one program/transcript pair.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct ReplayData {
    steps: Vec<RvrReplayStep>,
    opcode_ranges: BTreeMap<u32, Range<usize>>,
}

#[cfg(test)]
impl ReplayData {
    pub(crate) fn build(
        pc_base: u32,
        opcodes: &[u32],
        transcript: &PreflightEventLog,
        endpoint: PreflightEndpoint,
    ) -> Result<Self, PreflightIndexError> {
        let step_memory_starts =
            build_step_memory_starts(&transcript.program_log, &transcript.memory_log)?;
        let mut opcode_counts = BTreeMap::<u32, usize>::new();
        for program_index in 0..step_memory_starts.len() {
            let opcode = resolve_opcode(pc_base, opcodes, &transcript.program_log[program_index])?;
            validate_step_endpoint(opcode, program_index, &transcript.program_log, endpoint)?;
            *opcode_counts.entry(opcode).or_default() += 1;
        }
        validate_endpoint(pc_base, opcodes, transcript, endpoint)?;

        let mut opcode_ranges = BTreeMap::new();
        let mut cursor = 0usize;
        for (&opcode, &count) in &opcode_counts {
            opcode_ranges.insert(opcode, cursor..cursor + count);
            cursor += count;
        }
        let mut next = opcode_ranges
            .iter()
            .map(|(&opcode, range)| (opcode, range.start))
            .collect::<BTreeMap<_, _>>();
        let mut steps = vec![RvrReplayStep::default(); step_memory_starts.len()];
        for (program_index, &memory_start) in step_memory_starts.iter().enumerate() {
            let opcode = resolve_opcode(pc_base, opcodes, &transcript.program_log[program_index])?;
            let destination = next
                .get_mut(&opcode)
                .expect("opcode count was collected in the first pass");
            steps[*destination] = RvrReplayStep {
                program_index: u32::try_from(program_index).map_err(|_| {
                    PreflightIndexError::new("program log has more than u32::MAX entries")
                })?,
                memory_start,
            };
            *destination += 1;
        }
        Ok(Self {
            steps,
            opcode_ranges,
        })
    }

    pub(crate) fn steps(&self) -> &[RvrReplayStep] {
        &self.steps
    }

    pub(crate) fn opcode_ranges(&self) -> &BTreeMap<u32, Range<usize>> {
        &self.opcode_ranges
    }
}

#[cfg(test)]
fn resolve_opcode(
    pc_base: u32,
    opcodes: &[u32],
    event: &PreflightProgramEvent,
) -> Result<u32, PreflightIndexError> {
    let delta = event.pc.checked_sub(pc_base).ok_or_else(|| {
        PreflightIndexError::new(format!(
            "program log PC {:#x} precedes program base",
            event.pc
        ))
    })?;
    if delta % DEFAULT_PC_STEP != 0 {
        return Err(PreflightIndexError::new(format!(
            "program log PC {:#x} is not instruction-aligned",
            event.pc
        )));
    }
    let slot = (delta / DEFAULT_PC_STEP) as usize;
    let opcode = opcodes
        .get(slot)
        .copied()
        .filter(|&opcode| opcode != u32::MAX)
        .ok_or_else(|| {
            PreflightIndexError::new(format!(
                "program log PC {:#x} points to an undefined instruction",
                event.pc
            ))
        })?;
    Ok(opcode)
}

#[cfg(test)]
fn validate_step_endpoint(
    opcode: u32,
    program_index: usize,
    log: &[PreflightProgramEvent],
    endpoint: PreflightEndpoint,
) -> Result<(), PreflightIndexError> {
    let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
    let from = log[program_index];
    let to = log[program_index + 1];
    if opcode == terminate {
        if !matches!(endpoint, PreflightEndpoint::Terminated)
            || program_index + 2 != log.len()
            || from != to
        {
            return Err(PreflightIndexError::new(
                "TERMINATE must be the final fetched instruction and duplicate the final sentinel",
            ));
        }
    } else if to.timestamp == from.timestamp {
        return Err(PreflightIndexError::new(format!(
            "non-TERMINATE instruction {program_index} did not advance the timestamp"
        )));
    }
    Ok(())
}

#[cfg(test)]
fn validate_endpoint(
    pc_base: u32,
    opcodes: &[u32],
    transcript: &PreflightEventLog,
    endpoint: PreflightEndpoint,
) -> Result<(), PreflightIndexError> {
    match endpoint {
        PreflightEndpoint::Terminated => {
            if transcript.program_log.len() < 2 {
                return Err(PreflightIndexError::new(
                    "terminated transcript has no fetched TERMINATE instruction",
                ));
            }
            let last = &transcript.program_log[transcript.program_log.len() - 2];
            if resolve_opcode(pc_base, opcodes, last)?
                != SystemOpcode::TERMINATE.global_opcode().as_usize() as u32
            {
                return Err(PreflightIndexError::new(
                    "terminated transcript does not end with TERMINATE",
                ));
            }
        }
        PreflightEndpoint::Suspended => {
            let sentinel = transcript.program_log.last().unwrap();
            resolve_opcode(pc_base, opcodes, sentinel)?;
        }
    }
    Ok(())
}

pub(crate) fn build_memory_predecessors(
    memory: &[PreflightMemoryEvent],
    seeds: &[PreflightInitialWrite],
) -> Result<Vec<u32>, PreflightIndexError> {
    if memory.len() >= MEMORY_PREDECESSOR_INDEX_MASK as usize
        || seeds.len() >= MEMORY_PREDECESSOR_INDEX_MASK as usize
    {
        return Err(PreflightIndexError::new(
            "memory or initial-write log is too large for packed predecessor indexes",
        ));
    }

    let mut seed_by_block = FxHashMap::with_capacity_and_hasher(seeds.len(), Default::default());
    for (index, seed) in seeds.iter().enumerate() {
        if seed_by_block
            .insert(
                memory_key(seed.address_space, seed.pointer),
                u32::try_from(index).expect("seed count was bounded above"),
            )
            .is_some()
        {
            return Err(PreflightIndexError::new(format!(
                "duplicate initial-write seed for AS={} pointer={}",
                seed.address_space, seed.pointer
            )));
        }
    }

    let mut last_event = FxHashMap::<u64, u32>::default();
    let mut predecessors = Vec::with_capacity(memory.len());
    for (event_index, event) in memory.iter().enumerate() {
        let address_space = event.address_space();
        let key = memory_key(address_space, event.pointer);
        let event_index = u32::try_from(event_index).expect("memory event count was bounded above");
        let predecessor = match last_event.entry(key) {
            Entry::Occupied(mut previous) => {
                let predecessor = *previous.get() + 1;
                previous.insert(event_index);
                predecessor
            }
            Entry::Vacant(vacant) => {
                let predecessor = if event.is_write() {
                    let seed_index = seed_by_block.remove(&key).ok_or_else(|| {
                        PreflightIndexError::new(format!(
                            "first event is a write without a seed for AS={} pointer={}",
                            address_space, event.pointer
                        ))
                    })?;
                    MEMORY_PREDECESSOR_SEED_BIT | seed_index
                } else {
                    MEMORY_PREDECESSOR_BASELINE
                };
                vacant.insert(event_index);
                predecessor
            }
        };
        predecessors.push(predecessor);
    }

    if !seed_by_block.is_empty() {
        return Err(PreflightIndexError::new(format!(
            "{} initial-write seeds were not used",
            seed_by_block.len()
        )));
    }
    Ok(predecessors)
}

pub(crate) struct MemoryReplayIndex {
    pub(crate) predecessors: Vec<u32>,
    pub(crate) touched_blocks: Vec<TouchedBlock<BabyBear>>,
}

/// Builds the derived memory metadata used only when tests upload fabricated
/// expanded logs. Production preflight derives the same metadata on the GPU
/// from compact checkpoints and residuals.
pub(crate) fn build_memory_index(
    memory: &[PreflightMemoryEvent],
    seeds: &[PreflightInitialWrite],
    config: &MemoryConfig,
) -> Result<MemoryReplayIndex, PreflightIndexError> {
    let timestamp_limit = 1u64
        .checked_shl(config.timestamp_max_bits as u32)
        .ok_or_else(|| PreflightIndexError::new("timestamp width exceeds u64"))?;
    let mut previous_timestamp = None;
    let mut final_blocks = BTreeMap::<u64, (PreflightMemoryEvent, bool)>::new();
    for (index, event) in memory.iter().copied().enumerate() {
        validate_memory_block(event.address_space(), event.pointer, config)?;
        if u64::from(event.timestamp) >= timestamp_limit {
            return Err(PreflightIndexError::new(format!(
                "memory event {index} timestamp exceeds the configured domain"
            )));
        }
        if previous_timestamp.is_some_and(|previous| previous >= event.timestamp) {
            return Err(PreflightIndexError::new(
                "memory timestamps are not strictly increasing",
            ));
        }
        previous_timestamp = Some(event.timestamp);
        final_blocks
            .entry(memory_key(event.address_space(), event.pointer))
            .and_modify(|(last, dirty)| {
                *last = event;
                *dirty |= event.is_write();
            })
            .or_insert((event, event.is_write()));
    }
    for seed in seeds {
        if seed.address_space & PREFLIGHT_WRITE_BIT != 0 {
            return Err(PreflightIndexError::new(
                "initial-write seed address space contains the write bit",
            ));
        }
        validate_memory_block(seed.address_space, seed.pointer, config)?;
    }

    let predecessors = build_memory_predecessors(memory, seeds)?;
    let touched_blocks = final_blocks
        .into_values()
        .map(|(event, dirty)| TouchedBlock {
            address_space: event.address_space(),
            ptr: event.pointer,
            is_dirty: u32::from(dirty),
            timestamp: event.timestamp,
            values: event.value.map(BabyBear::from_u16),
        })
        .collect();
    Ok(MemoryReplayIndex {
        predecessors,
        touched_blocks,
    })
}

fn validate_memory_block(
    address_space: u32,
    pointer: u32,
    config: &MemoryConfig,
) -> Result<(), PreflightIndexError> {
    let address_space_count = 1u64
        .checked_shl(config.addr_space_height as u32)
        .ok_or_else(|| PreflightIndexError::new("address-space height exceeds u64"))?;
    let address_space_limit = u64::from(ADDR_SPACE_OFFSET)
        .checked_add(address_space_count)
        .ok_or_else(|| PreflightIndexError::new("address-space domain overflow"))?;
    let pointer_limit = 1u64
        .checked_shl(config.pointer_max_bits as u32)
        .ok_or_else(|| PreflightIndexError::new("pointer width exceeds u64"))?;
    let layout = config
        .addr_spaces
        .get(address_space as usize)
        .filter(|_| {
            address_space >= ADDR_SPACE_OFFSET && u64::from(address_space) < address_space_limit
        })
        .ok_or_else(|| {
            PreflightIndexError::new(format!("address space {address_space} is out of range"))
        })?;
    if layout.layout != MemoryCellType::U16 {
        return Err(PreflightIndexError::new(format!(
            "address space {address_space} does not use u16 cells"
        )));
    }
    let end = u64::from(pointer)
        .checked_add(BLOCK_FE_WIDTH as u64)
        .ok_or_else(|| PreflightIndexError::new("memory block pointer overflow"))?;
    if pointer % BLOCK_FE_WIDTH as u32 != 0
        || u64::from(pointer) >= pointer_limit
        || end > layout.num_cells as u64
    {
        return Err(PreflightIndexError::new(format!(
            "memory block AS={address_space} pointer={pointer} is out of range or misaligned"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rvr_state::{
        PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent, PREFLIGHT_WRITE_BIT,
    };

    use super::*;

    fn read(timestamp: u32, pointer: u32, value: [u16; 4]) -> PreflightMemoryEvent {
        PreflightMemoryEvent {
            timestamp,
            address_space_and_kind: 1,
            pointer,
            value,
        }
    }

    fn write(timestamp: u32, pointer: u32, value: [u16; 4]) -> PreflightMemoryEvent {
        PreflightMemoryEvent {
            timestamp,
            address_space_and_kind: PREFLIGHT_WRITE_BIT | 1,
            pointer,
            value,
        }
    }

    #[test]
    fn builds_step_offsets_and_memory_predecessors() {
        let transcript = PreflightEventLog {
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
                    timestamp: 5,
                },
                PreflightProgramEvent {
                    pc: 8,
                    timestamp: 5,
                },
            ],
            memory_log: vec![
                read(1, 0, [0; 4]),
                write(2, 4, [5, 0, 0, 0]),
                read(3, 4, [5, 0, 0, 0]),
                write(4, 4, [6, 0, 0, 0]),
            ],
            initial_write_log: vec![PreflightInitialWrite {
                address_space: 1,
                pointer: 4,
                initial_value: [9, 8, 7, 6],
            }],
        };

        assert_eq!(
            build_step_memory_starts(&transcript.program_log, &transcript.memory_log).unwrap(),
            vec![0, 2, 4]
        );
        assert_eq!(
            build_memory_predecessors(&transcript.memory_log, &transcript.initial_write_log)
                .unwrap(),
            vec![
                MEMORY_PREDECESSOR_BASELINE,
                MEMORY_PREDECESSOR_SEED_BIT,
                2,
                3,
            ]
        );
    }

    #[test]
    fn rejects_a_first_write_without_exactly_one_seed() {
        let transcript = PreflightEventLog {
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
            memory_log: vec![write(1, 4, [1, 0, 0, 0])],
            initial_write_log: vec![],
        };
        assert!(
            build_memory_predecessors(&transcript.memory_log, &transcript.initial_write_log)
                .unwrap_err()
                .to_string()
                .contains("without a seed")
        );
    }

    #[test]
    fn tracks_interleaved_block_predecessors() {
        let memory = vec![
            read(1, 0, [1; 4]),
            read(2, 4, [2; 4]),
            read(3, 0, [1; 4]),
            write(4, 4, [3; 4]),
        ];
        assert_eq!(
            build_memory_predecessors(&memory, &[]).unwrap(),
            vec![
                MEMORY_PREDECESSOR_BASELINE,
                MEMORY_PREDECESSOR_BASELINE,
                1,
                2,
            ]
        );
    }

    #[test]
    fn rejects_duplicate_and_unused_initial_write_seeds() {
        let seed = PreflightInitialWrite {
            address_space: 1,
            pointer: 4,
            initial_value: [0; 4],
        };
        assert!(
            build_memory_predecessors(&[write(1, 4, [1; 4])], &[seed, seed])
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
        assert!(build_memory_predecessors(&[read(1, 4, [1; 4])], &[seed])
            .unwrap_err()
            .to_string()
            .contains("were not used"));
    }

    #[test]
    fn builds_touched_blocks_and_rejects_invalid_memory_metadata() {
        let config = MemoryConfig::default();
        let memory = [
            read(1, 4, [1, 2, 3, 4]),
            write(2, 0, [5, 6, 7, 8]),
            write(3, 4, [9, 10, 11, 12]),
        ];
        let index = build_memory_index(
            &memory,
            &[PreflightInitialWrite {
                address_space: 1,
                pointer: 0,
                initial_value: [0; 4],
            }],
            &config,
        )
        .unwrap();
        assert_eq!(index.predecessors, [0, MEMORY_PREDECESSOR_SEED_BIT, 1]);
        assert_eq!(
            index.touched_blocks,
            [
                TouchedBlock {
                    address_space: 1,
                    ptr: 0,
                    is_dirty: 1,
                    timestamp: 2,
                    values: [5, 6, 7, 8].map(BabyBear::from_u16),
                },
                TouchedBlock {
                    address_space: 1,
                    ptr: 4,
                    is_dirty: 1,
                    timestamp: 3,
                    values: [9, 10, 11, 12].map(BabyBear::from_u16),
                },
            ]
        );

        for invalid in [
            PreflightMemoryEvent {
                address_space_and_kind: 0,
                ..read(1, 0, [0; 4])
            },
            read(1, 2, [0; 4]),
            PreflightMemoryEvent {
                address_space_and_kind: 4,
                ..read(1, 0, [0; 4])
            },
        ] {
            assert!(build_memory_index(&[invalid], &[], &config).is_err());
        }
        assert!(
            build_memory_index(&[read(1, 0, [0; 4]), read(1, 4, [0; 4])], &[], &config).is_err()
        );
        let mut narrow_timestamp = config.clone();
        narrow_timestamp.timestamp_max_bits = 1;
        assert!(build_memory_index(&[read(2, 0, [0; 4])], &[], &narrow_timestamp).is_err());
        assert!(build_memory_index(
            &[write(1, 0, [0; 4])],
            &[PreflightInitialWrite {
                address_space: PREFLIGHT_WRITE_BIT | 1,
                pointer: 0,
                initial_value: [0; 4],
            }],
            &config,
        )
        .is_err());
    }

    #[test]
    fn rejects_non_increasing_memory_timestamps_across_steps() {
        let program = vec![
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
                timestamp: 5,
            },
        ];
        let memory = vec![read(1, 0, [0; 4]), read(3, 4, [0; 4]), read(3, 8, [0; 4])];
        assert!(build_step_memory_starts(&program, &memory)
            .unwrap_err()
            .to_string()
            .contains("not strictly increasing"));
    }

    #[test]
    fn accepts_a_suspended_segment_boundary() {
        let phantom = SystemOpcode::PHANTOM.global_opcode().as_usize() as u32;
        let transcript = PreflightEventLog {
            program_log: vec![
                PreflightProgramEvent {
                    pc: 8,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 12,
                    timestamp: 3,
                },
            ],
            memory_log: vec![read(1, 0, [0; 4]), read(2, 4, [0; 4])],
            initial_write_log: vec![],
        };
        let replay = ReplayData::build(
            8,
            &[phantom, phantom],
            &transcript,
            PreflightEndpoint::Suspended,
        )
        .unwrap();
        assert_eq!(
            replay.steps,
            vec![RvrReplayStep {
                program_index: 0,
                memory_start: 0
            }]
        );
    }
}
