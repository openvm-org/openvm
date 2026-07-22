use std::{mem::size_of, ops::RangeInclusive};

use openvm_instructions::SysPhantom;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_state::{
    PreflightFieldBlock, PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent,
    PREFLIGHT_WRITE_BIT,
};

use crate::{
    arch::{
        AddressSpaceHostLayout, ExecutionCtxTrait, MemoryCellType, PreflightHistory,
        PreflightMemoryLog, BLOCK_FE_WIDTH,
    },
    system::memory::online::{GuestMemory, PagedVec, PAGE_SIZE},
};

#[derive(Debug)]
struct PendingWrite {
    timestamp: u32,
    address_space: u32,
    pointer: u32,
    initial_value: Option<[u16; BLOCK_FE_WIDTH]>,
}

/// Execution context for interpreter preflight.
///
/// The interpreter continues to execute against ordinary mutable guest
/// memory. This context turns each proof-visible access into an append-only,
/// read-only history for postflight and trace generation.
pub struct PreflightCtx {
    history: PreflightHistory,
    timestamp: u32,
    block_byte_shifts: Vec<u32>,
    seen_blocks: Vec<PagedVec<bool, PAGE_SIZE>>,
    pending_writes: Vec<PendingWrite>,
    read_canonical_field_block: unsafe fn(&GuestMemory, u32, u32) -> [u32; BLOCK_FE_WIDTH],
    pub instret_left: u64,
}

impl PreflightCtx {
    pub(crate) fn new<F: PrimeField32>(memory: &GuestMemory, instret_left: Option<u64>) -> Self {
        assert!(
            memory
                .memory
                .config
                .iter()
                .all(|config| config.layout != MemoryCellType::field32())
                || size_of::<F>() == size_of::<u32>(),
            "field32 memory requires a four-byte proof field"
        );
        let block_byte_shifts = memory
            .memory
            .config
            .iter()
            .map(|config| {
                let block_bytes = BLOCK_FE_WIDTH * config.layout.size();
                assert!(
                    block_bytes.is_power_of_two(),
                    "memory-bus block byte width must be a power of two"
                );
                block_bytes.ilog2()
            })
            .collect();
        let seen_blocks = memory
            .memory
            .config
            .iter()
            .map(|config| PagedVec::new(config.num_cells.div_ceil(BLOCK_FE_WIDTH)))
            .collect();
        let mut history = PreflightHistory::default();
        // A bounded segment knows its exact program-log length. Reservation is
        // best-effort so allocation failure remains on the normal push path.
        if let Some(program_capacity) = instret_left
            .and_then(|instret| usize::try_from(instret).ok())
            .and_then(|instret| instret.checked_add(1))
        {
            let _ = history.program.try_reserve_exact(program_capacity);
        }
        Self {
            history,
            timestamp: 1,
            block_byte_shifts,
            seen_blocks,
            pending_writes: Vec::with_capacity(2),
            read_canonical_field_block: read_canonical_field_block::<F>,
            instret_left: instret_left.unwrap_or(u64::MAX),
        }
    }

    pub fn finish(mut self, pc: u32) -> PreflightHistory {
        debug_assert!(self.pending_writes.is_empty());
        self.history.program.push(PreflightProgramEvent {
            pc,
            timestamp: self.timestamp,
        });
        self.history
    }

    #[inline(always)]
    fn block_range(block_byte_shift: u32, byte_ptr: u32, byte_len: u32) -> RangeInclusive<u32> {
        let first = byte_ptr >> block_byte_shift;
        let last_byte = byte_ptr
            .checked_add(byte_len - 1)
            .expect("preflight memory access range overflow");
        let last = last_byte >> block_byte_shift;
        first..=last
    }

    #[inline(always)]
    fn block_value(
        memory: &GuestMemory,
        address_space: u32,
        block_index: u32,
        read_canonical_field_block: unsafe fn(&GuestMemory, u32, u32) -> [u32; BLOCK_FE_WIDTH],
        log: &mut PreflightMemoryLog,
        initial: bool,
    ) -> [u16; BLOCK_FE_WIDTH] {
        let pointer = block_index * BLOCK_FE_WIDTH as u32;
        match memory.memory.config[address_space as usize].layout {
            MemoryCellType::U16 => unsafe {
                memory.read::<u16, BLOCK_FE_WIDTH>(address_space, pointer)
            },
            MemoryCellType::FIELD32 => {
                let values = unsafe { read_canonical_field_block(memory, address_space, pointer) };
                let reference = if initial {
                    let index = log.field_initial_values.len();
                    log.field_initial_values
                        .push(PreflightFieldBlock { values });
                    index
                } else {
                    let index = log.field_values.len();
                    log.field_values.push(PreflightFieldBlock { values });
                    index
                };
                let index =
                    u32::try_from(reference).expect("field preflight log exceeds u32::MAX blocks");
                [index as u16, (index >> 16) as u16, 0, 0]
            }
            _ => panic!("preflight memory log requires u16 or 32-bit field cells"),
        }
    }

    #[inline(always)]
    fn next_access(&mut self, address_space: u32, block_index: u32) -> (u32, bool) {
        let timestamp = self.timestamp;
        self.timestamp += 1;
        let seen = &mut self.seen_blocks[address_space as usize];
        let was_seen = seen.replace(block_index as usize, true);
        (timestamp, was_seen)
    }

    #[inline(always)]
    fn log_read(&mut self, memory: &GuestMemory, address_space: u32, byte_ptr: u32, byte_len: u32) {
        if byte_len == 0 {
            return;
        }
        let block_byte_shift = self.block_byte_shifts[address_space as usize];
        for block_index in Self::block_range(block_byte_shift, byte_ptr, byte_len) {
            let (timestamp, _) = self.next_access(address_space, block_index);
            let value = Self::block_value(
                memory,
                address_space,
                block_index,
                self.read_canonical_field_block,
                &mut self.history.memory,
                false,
            );
            self.history.memory.accesses.push(PreflightMemoryEvent {
                timestamp,
                address_space_and_kind: address_space,
                pointer: block_index * BLOCK_FE_WIDTH as u32,
                value,
            });
        }
    }

    #[inline(always)]
    fn begin_write(
        &mut self,
        memory: &GuestMemory,
        address_space: u32,
        byte_ptr: u32,
        byte_len: u32,
    ) {
        if byte_len == 0 {
            return;
        }
        debug_assert!(self.pending_writes.is_empty());
        let block_byte_shift = self.block_byte_shifts[address_space as usize];
        for block_index in Self::block_range(block_byte_shift, byte_ptr, byte_len) {
            let (timestamp, was_seen) = self.next_access(address_space, block_index);
            let pointer = block_index * BLOCK_FE_WIDTH as u32;
            let initial_value = (!was_seen).then(|| {
                Self::block_value(
                    memory,
                    address_space,
                    block_index,
                    self.read_canonical_field_block,
                    &mut self.history.memory,
                    true,
                )
            });
            self.pending_writes.push(PendingWrite {
                timestamp,
                address_space,
                pointer,
                initial_value,
            });
        }
    }

    #[inline(always)]
    fn finish_write(&mut self, memory: &GuestMemory) {
        for pending in self.pending_writes.drain(..) {
            let block_index = pending.pointer / BLOCK_FE_WIDTH as u32;
            let value = Self::block_value(
                memory,
                pending.address_space,
                block_index,
                self.read_canonical_field_block,
                &mut self.history.memory,
                false,
            );
            self.history.memory.accesses.push(PreflightMemoryEvent {
                timestamp: pending.timestamp,
                address_space_and_kind: pending.address_space | PREFLIGHT_WRITE_BIT,
                pointer: pending.pointer,
                value,
            });
            if let Some(initial_value) = pending.initial_value {
                self.history
                    .memory
                    .initial_writes
                    .push(PreflightInitialWrite {
                        address_space: pending.address_space,
                        pointer: pending.pointer,
                        initial_value,
                    });
            }
        }
    }
}

unsafe fn read_canonical_field_block<F: PrimeField32>(
    memory: &GuestMemory,
    address_space: u32,
    pointer: u32,
) -> [u32; BLOCK_FE_WIDTH] {
    // SAFETY: PreflightCtx::new checks that F matches the configured four-byte
    // field cell width, and block pointers come from validated memory accesses.
    unsafe { memory.read::<F, BLOCK_FE_WIDTH>(address_space, pointer) }
        .map(|value| value.as_canonical_u32())
}

impl ExecutionCtxTrait for PreflightCtx {
    #[inline(always)]
    fn on_memory_operation(&mut self, _address_space: u32, _ptr: u32, _size: u32, _is_write: bool) {
    }

    #[inline(always)]
    fn on_memory_read(
        &mut self,
        memory: &GuestMemory,
        address_space: u32,
        byte_ptr: u32,
        byte_len: u32,
    ) {
        self.log_read(memory, address_space, byte_ptr, byte_len);
    }

    #[inline(always)]
    fn on_memory_write_start(
        &mut self,
        memory: &GuestMemory,
        address_space: u32,
        byte_ptr: u32,
        byte_len: u32,
    ) {
        self.begin_write(memory, address_space, byte_ptr, byte_len);
    }

    #[inline(always)]
    fn on_memory_write_end(&mut self, memory: &GuestMemory) {
        self.finish_write(memory);
    }

    #[inline(always)]
    fn on_instruction_start(exec_state: &mut crate::arch::VmExecState<GuestMemory, Self>, pc: u32) {
        exec_state.ctx.history.program.push(PreflightProgramEvent {
            pc,
            timestamp: exec_state.ctx.timestamp,
        });

        #[cfg(all(feature = "metrics", any(debug_assertions, feature = "perf-metrics")))]
        exec_state.vm_state.metrics.update_backtrace(pc);
    }

    #[inline(always)]
    fn advance_timestamp(&mut self, slots: u32) {
        self.timestamp += slots;
    }

    // State and PC are used only in builds that collect guest backtraces.
    #[inline(always)]
    fn on_system_phantom(
        _exec_state: &mut crate::arch::VmExecState<GuestMemory, Self>,
        _pc: u32,
        phantom: SysPhantom,
    ) {
        if phantom == SysPhantom::DebugPanic {
            #[cfg(all(feature = "metrics", any(debug_assertions, feature = "perf-metrics")))]
            {
                let metrics = &mut _exec_state.vm_state.metrics;
                metrics.update_backtrace(_pc);
                if let Some(mut backtrace) = metrics.prev_backtrace.take() {
                    backtrace.resolve();
                    eprintln!("openvm program failure; backtrace:\n{backtrace:?}");
                } else {
                    eprintln!("openvm program failure; no backtrace");
                }
            }
        }
    }

    #[inline(always)]
    fn should_suspend(exec_state: &mut crate::arch::VmExecState<GuestMemory, Self>) -> bool {
        if exec_state.ctx.instret_left == 0 {
            true
        } else {
            exec_state.ctx.instret_left -= 1;
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use openvm_instructions::{riscv::RV64_MEMORY_AS, DEFERRAL_AS};
    use openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField32};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::PreflightCtx;
    use crate::{
        arch::{MemoryConfig, BLOCK_FE_WIDTH},
        system::memory::online::{AddressMap, GuestMemory},
    };

    #[test]
    fn field_history_uses_canonical_words() {
        let mut memory = GuestMemory::new(AddressMap::from_mem_config(&MemoryConfig::default()));
        let values = [
            BabyBear::ONE,
            BabyBear::TWO,
            BabyBear::from_u32(123_456),
            BabyBear::from_u32(BabyBear::ORDER_U32 - 1),
        ];
        unsafe {
            memory.write(DEFERRAL_AS, 0, values);
        }

        let mut ctx = PreflightCtx::new::<BabyBear>(&memory, None);
        ctx.log_read(
            &memory,
            DEFERRAL_AS,
            0,
            (BLOCK_FE_WIDTH * size_of::<BabyBear>()) as u32,
        );
        let history = ctx.finish(0);

        assert_eq!(
            history.memory.field_values[0].values,
            values.map(|value| value.as_canonical_u32())
        );
    }

    #[test]
    fn repeated_writes_seed_a_block_once() {
        let mut memory = GuestMemory::new(AddressMap::from_mem_config(&MemoryConfig::default()));
        let mut ctx = PreflightCtx::new::<BabyBear>(&memory, None);

        ctx.begin_write(&memory, RV64_MEMORY_AS, 0, size_of::<u16>() as u32);
        unsafe {
            memory.write(RV64_MEMORY_AS, 0, [1u16]);
        }
        ctx.finish_write(&memory);

        ctx.begin_write(&memory, RV64_MEMORY_AS, 0, size_of::<u16>() as u32);
        unsafe {
            memory.write(RV64_MEMORY_AS, 0, [2u16]);
        }
        ctx.finish_write(&memory);

        let history = ctx.finish(0);
        assert_eq!(history.memory.accesses.len(), 2);
        assert_eq!(history.memory.initial_writes.len(), 1);
        assert_eq!(
            history.memory.initial_writes[0].initial_value,
            [0; BLOCK_FE_WIDTH]
        );
    }

    #[test]
    fn read_then_write_uses_the_read_as_predecessor() {
        let mut memory = GuestMemory::new(AddressMap::from_mem_config(&MemoryConfig::default()));
        unsafe {
            memory.write(RV64_MEMORY_AS, 0, [7u16]);
        }
        let mut ctx = PreflightCtx::new::<BabyBear>(&memory, None);

        ctx.log_read(&memory, RV64_MEMORY_AS, 0, size_of::<u16>() as u32);
        ctx.begin_write(&memory, RV64_MEMORY_AS, 0, size_of::<u16>() as u32);
        unsafe {
            memory.write(RV64_MEMORY_AS, 0, [9u16]);
        }
        ctx.finish_write(&memory);

        let history = ctx.finish(0);
        assert_eq!(history.memory.accesses.len(), 2);
        assert!(!history.memory.accesses[0].is_write());
        assert!(history.memory.accesses[1].is_write());
        assert_eq!(history.memory.accesses[0].value, [7, 0, 0, 0]);
        assert_eq!(history.memory.accesses[1].value, [9, 0, 0, 0]);
        assert!(history.memory.initial_writes.is_empty());
    }
}
