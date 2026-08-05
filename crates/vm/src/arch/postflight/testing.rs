use openvm_stark_backend::p3_field::PrimeField32;

use super::*;
use crate::arch::{
    testing::memory::{air::MemoryDummyChip, PostflightTestMemory},
    VmField,
};

impl<'a> Postflight<'a> {
    pub fn new_for_test(
        program: &'a Program,
        history: &'a PreflightHistory,
        memory_config: &MemoryConfig,
    ) -> Result<Self, PostflightError> {
        let program_index = PostflightProgramIndex::new(program)?;
        Self::new_inner(program, &program_index, history, memory_config, None, false)
    }

    pub(crate) fn balance_test_memory<F: PrimeField32>(&self, chip: &mut MemoryDummyChip<F>) {
        self.validate_field_values(F::ORDER_U32)
            .expect("postflight field values must be canonical");
        for event_index in 0..self.history.memory.accesses.len() {
            let event = self.history.memory.accesses[event_index];
            let previous_timestamp = self.previous_timestamp(event_index);
            match self.memory_config.addr_spaces[event.address_space() as usize].layout {
                MemoryCellType::U16 => {
                    let value = event.value.map(F::from_u16);
                    let previous = self.previous_u16(event_index).map(F::from_u16);
                    chip.send(
                        event.address_space(),
                        event.pointer,
                        &previous,
                        previous_timestamp,
                    );
                    chip.receive(
                        event.address_space(),
                        event.pointer,
                        &value,
                        event.timestamp,
                    );
                }
                MemoryCellType::FIELD32 => {
                    let value = self.field_value(event_index).map(F::from_u32);
                    let previous = self.previous_field32(event_index).map(F::from_u32);
                    chip.send(
                        event.address_space(),
                        event.pointer,
                        &previous,
                        previous_timestamp,
                    );
                    chip.receive(
                        event.address_space(),
                        event.pointer,
                        &value,
                        event.timestamp,
                    );
                }
                _ => unreachable!("postflight validates every accessed memory layout"),
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn memory_predecessors_for_test(&self) -> &[u32] {
        &self.memory_predecessors
    }

    #[cfg(feature = "cuda")]
    pub fn replay_steps_for_test(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.steps.iter().map(|step| {
            let program_index = step.0;
            (program_index, self.memory_starts[program_index as usize])
        })
    }

    #[cfg(feature = "cuda")]
    pub fn opcode_ranges_for_test(&self) -> &BTreeMap<u32, Range<usize>> {
        &self.opcode_ranges
    }

    pub(crate) fn record_test_writes<F, M>(&self, memory: &mut M)
    where
        F: VmField,
        M: PostflightTestMemory<F>,
    {
        self.validate_field_values(F::ORDER_U32)
            .expect("postflight field values must be canonical");
        let mut first_writes = FxHashMap::<u64, usize>::default();
        for (event_index, event) in self.history.memory.accesses.iter().enumerate() {
            if event.is_write() {
                first_writes
                    .entry(memory_key(event.address_space(), event.pointer))
                    .or_insert(event_index);
            }
        }

        for event_index in first_writes.into_values() {
            let event = self.history.memory.accesses[event_index];
            match self.memory_config.addr_spaces[event.address_space() as usize].layout {
                MemoryCellType::U16 => unsafe {
                    memory.tracing_memory().data.write::<u16, BLOCK_FE_WIDTH>(
                        event.address_space(),
                        event.pointer,
                        self.previous_u16(event_index),
                    );
                },
                MemoryCellType::FIELD32 => unsafe {
                    memory.tracing_memory().data.write::<F, BLOCK_FE_WIDTH>(
                        event.address_space(),
                        event.pointer,
                        self.previous_field32(event_index).map(F::from_u32),
                    );
                },
                _ => unreachable!("postflight validates every accessed memory layout"),
            }
        }

        for (event_index, event) in self.history.memory.accesses.iter().enumerate() {
            if !event.is_write() {
                continue;
            }
            let value = match self.memory_config.addr_spaces[event.address_space() as usize].layout
            {
                MemoryCellType::U16 => event.value.map(F::from_u16),
                MemoryCellType::FIELD32 => self.field_value(event_index).map(F::from_u32),
                _ => unreachable!("postflight validates every accessed memory layout"),
            };
            memory.write_block(
                event.address_space() as usize,
                event.pointer as usize,
                value,
            );
        }
    }
}
