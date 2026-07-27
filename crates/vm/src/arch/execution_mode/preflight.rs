use rvr_state::{
    PreflightFieldBlock, PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent,
    PREFLIGHT_WRITE_BIT,
};

use crate::{
    arch::{
        AddressSpaceHostLayout, Arena, ExecutionCtxTrait, MemoryCellType, PreflightHistory,
        PreflightMemoryLog, BLOCK_FE_WIDTH,
    },
    system::memory::online::{GuestMemory, PagedVec, PAGE_SIZE},
};

/// Temporary context for the record-producing interpreter. This disappears
/// with the legacy trace-generation path.
pub struct RecordCtx<RA> {
    pub arenas: Vec<RA>,
    pub program: Vec<PreflightProgramEvent>,
    pub instret_left: u64,
}

impl<RA: Arena> RecordCtx<RA> {
    /// `capacities` is list of `(height, width)` dimensions for each arena, indexed by AIR index.
    /// The length of `capacities` must equal the number of AIRs.
    /// Here `height` will always mean an overestimate of the trace height for that AIR, while
    /// `width` may have different meanings depending on the `RA` type.
    pub(crate) fn new_with_capacity(
        capacities: &[(usize, usize)],
        instret_left: Option<u64>,
    ) -> Self {
        let arenas = capacities
            .iter()
            .map(|&(height, main_width)| RA::with_capacity(height, main_width))
            .collect();

        Self {
            arenas,
            program: Vec::new(),
            instret_left: instret_left.unwrap_or(u64::MAX),
        }
    }
}

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
    last_access: Vec<PagedVec<u32, PAGE_SIZE>>,
    pending_writes: Vec<PendingWrite>,
    pub instret_left: u64,
}

impl PreflightCtx {
    pub(crate) fn new(memory: &GuestMemory, instret_left: Option<u64>) -> Self {
        let last_access = memory
            .memory
            .config
            .iter()
            .map(|config| PagedVec::new(config.num_cells.div_ceil(BLOCK_FE_WIDTH)))
            .collect();
        Self {
            history: PreflightHistory::default(),
            timestamp: 1,
            last_access,
            pending_writes: Vec::new(),
            instret_left: instret_left.unwrap_or(u64::MAX),
        }
    }

    pub fn timestamp(&self) -> u32 {
        self.timestamp
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
    fn block_bytes(memory: &GuestMemory, address_space: u32) -> usize {
        BLOCK_FE_WIDTH * memory.memory.config[address_space as usize].layout.size()
    }

    #[inline(always)]
    fn block_range(
        memory: &GuestMemory,
        address_space: u32,
        byte_ptr: u32,
        byte_len: u32,
    ) -> std::ops::RangeInclusive<u32> {
        let block_bytes = Self::block_bytes(memory, address_space) as u32;
        let first = byte_ptr / block_bytes;
        let last_byte = byte_ptr
            .checked_add(byte_len - 1)
            .expect("preflight memory access range overflow");
        let last = last_byte / block_bytes;
        first..=last
    }

    #[inline(always)]
    fn block_value(
        memory: &GuestMemory,
        address_space: u32,
        block_index: u32,
        log: &mut PreflightMemoryLog,
        initial: bool,
    ) -> [u16; BLOCK_FE_WIDTH] {
        let pointer = block_index * BLOCK_FE_WIDTH as u32;
        match memory.memory.config[address_space as usize].layout {
            MemoryCellType::U16 => unsafe {
                memory.read::<u16, BLOCK_FE_WIDTH>(address_space, pointer)
            },
            MemoryCellType::F { size: 4 } => {
                let values = unsafe { memory.read::<u32, BLOCK_FE_WIDTH>(address_space, pointer) };
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
    fn next_access(&mut self, address_space: u32, block_index: u32) -> (u32, u32) {
        let timestamp = self.timestamp;
        self.timestamp += 1;
        let last_access = &mut self.last_access[address_space as usize];
        let previous = last_access.get(block_index as usize);
        last_access.set(block_index as usize, timestamp);
        (timestamp, previous)
    }

    #[inline(always)]
    fn log_read(&mut self, memory: &GuestMemory, address_space: u32, byte_ptr: u32, byte_len: u32) {
        if byte_len == 0 {
            return;
        }
        for block_index in Self::block_range(memory, address_space, byte_ptr, byte_len) {
            let (timestamp, _) = self.next_access(address_space, block_index);
            let value = Self::block_value(
                memory,
                address_space,
                block_index,
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
        for block_index in Self::block_range(memory, address_space, byte_ptr, byte_len) {
            let (timestamp, previous) = self.next_access(address_space, block_index);
            let pointer = block_index * BLOCK_FE_WIDTH as u32;
            let initial_value = (previous == 0).then(|| {
                Self::block_value(
                    memory,
                    address_space,
                    block_index,
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

impl ExecutionCtxTrait for PreflightCtx {
    #[inline(always)]
    fn on_memory_operation(&mut self, _address_space: u32, _ptr: u32, _size: u32) {}

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
    fn on_instruction_start(&mut self, pc: u32) {
        self.history.program.push(PreflightProgramEvent {
            pc,
            timestamp: self.timestamp,
        });
    }

    #[inline(always)]
    fn advance_timestamp(&mut self, slots: u32) {
        self.timestamp += slots;
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
