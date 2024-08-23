use std::{cell::RefCell, collections::HashMap, rc::Rc, sync::Arc};

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use afs_primitives::range_gate::RangeCheckerGateChip;

use crate::{
    memory::{
        decompose,
        manager::{operation::MemoryOperation, trace_builder::MemoryTraceBuilder},
        offline_checker::bridge::MemoryOfflineChecker,
        OpType,
    },
    vm::config::MemoryConfig,
};

use super::{
    audit::{air::MemoryAuditAir, MemoryAuditChip},
    offline_checker::columns::MemoryAccess,
};

use self::{access_cell::AccessCell, interface::MemoryInterface};

pub mod access_cell;
pub mod dimensions;
pub mod interface;
pub mod operation;
pub mod trace_builder;

const WORD_SIZE: usize = 1;
const NUM_WORDS: usize = 16;

#[derive(Debug)]
pub struct MemoryManager<F: PrimeField32> {
    pub interface_chip: MemoryInterface<NUM_WORDS, WORD_SIZE, F>,
    mem_config: MemoryConfig,
    range_checker: Arc<RangeCheckerGateChip>,
    timestamp: F,
    /// Maps (addr_space, pointer) to (data, timestamp)
    pub memory: HashMap<(F, F), AccessCell<WORD_SIZE, F>>,
}

impl<F: PrimeField32> MemoryManager<F> {
    // pub fn with_persistent_memory(
    //     memory_dimensions: MemoryDimensions,
    //     memory: HashMap<(F, F), AccessCell<WORD_SIZE, F>>,
    // ) -> Self {
    //     Self {
    //         interface_chip: MemoryInterface::Persistent(MemoryExpandInterfaceChip::new(
    //             memory_dimensions,
    //         )),
    //         clk: F::one(),
    //         memory,
    //     }
    // }

    pub fn with_volatile_memory(
        mem_config: MemoryConfig,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            mem_config,
            interface_chip: MemoryInterface::Volatile(MemoryAuditChip::new(
                mem_config.addr_space_max_bits,
                mem_config.pointer_max_bits,
                mem_config.decomp,
                range_checker.clone(),
            )),
            timestamp: F::one(),
            memory: HashMap::new(),
            range_checker,
        }
    }

    pub fn make_offline_checker(manager: Rc<RefCell<Self>>) -> MemoryOfflineChecker {
        MemoryOfflineChecker::new(
            manager.borrow().mem_config.clk_max_bits,
            manager.borrow().mem_config.decomp,
        )
    }

    pub fn make_trace_builder(manager: Rc<RefCell<Self>>) -> MemoryTraceBuilder<F> {
        MemoryTraceBuilder::new(
            manager.clone(),
            manager.borrow().range_checker.clone(),
            Self::make_offline_checker(manager.clone()),
        )
    }

    pub fn read_word(&mut self, addr_space: F, pointer: F) -> MemoryAccess<WORD_SIZE, F> {
        let cur_clk = self.timestamp;
        self.timestamp += F::one();

        if addr_space == F::zero() {
            let data = decompose(pointer);
            return MemoryAccess::<WORD_SIZE, F>::new(
                MemoryOperation::new(
                    addr_space,
                    pointer,
                    F::from_canonical_u8(OpType::Read as u8),
                    AccessCell::new(data, cur_clk),
                    F::one(),
                ),
                AccessCell::new(data, F::zero()),
            );
        }
        debug_assert!((pointer.as_canonical_u32() as usize) % WORD_SIZE == 0);

        let cell = self.memory.get_mut(&(addr_space, pointer)).unwrap();
        let (old_clk, old_data) = (cell.clk, cell.data);
        debug_assert!(old_clk < cur_clk);

        // Updating AccessCell
        cell.clk = cur_clk;

        self.interface_chip
            .touch_address(addr_space, pointer, old_data, cur_clk);

        MemoryAccess::<WORD_SIZE, F>::new(
            MemoryOperation::new(
                addr_space,
                pointer,
                F::from_canonical_u8(OpType::Read as u8),
                *cell,
                F::one(),
            ),
            AccessCell::new(old_data, old_clk),
        )
    }

    /// Reads a word directly from memory without updating internal state.
    ///
    /// Any value returned is unconstrained.
    pub fn unsafe_read_word(&self, addr_space: F, pointer: F) -> [F; WORD_SIZE] {
        self.memory.get(&(addr_space, pointer)).unwrap().data
    }

    pub fn write_word(
        &mut self,
        addr_space: F,
        pointer: F,
        data: [F; WORD_SIZE],
    ) -> MemoryAccess<WORD_SIZE, F> {
        assert!(addr_space != F::zero());
        debug_assert!((pointer.as_canonical_u32() as usize) % WORD_SIZE == 0);

        let cur_clk = self.timestamp;
        self.timestamp += F::one();

        let cell = self
            .memory
            .entry((addr_space, pointer))
            .or_insert(AccessCell {
                data: [F::zero(); WORD_SIZE],
                clk: F::zero(),
            });
        let (old_clk, old_data) = (cell.clk, cell.data);
        assert!(old_clk < cur_clk);

        // Updating AccessCell
        cell.clk = cur_clk;
        cell.data = data;

        self.interface_chip
            .touch_address(addr_space, pointer, old_data, old_clk);

        MemoryAccess::<WORD_SIZE, F>::new(
            MemoryOperation::new(
                addr_space,
                pointer,
                F::from_canonical_u8(OpType::Write as u8),
                *cell,
                F::one(),
            ),
            AccessCell::new(old_data, old_clk),
        )
    }

    pub fn unsafe_write_word(&mut self, addr_space: F, pointer: F, data: [F; WORD_SIZE]) {
        assert!(addr_space != F::zero());
        debug_assert!((pointer.as_canonical_u32() as usize) % WORD_SIZE == 0);

        self.memory
            .entry((addr_space, pointer))
            .and_modify(|cell| cell.data = data)
            .or_insert(AccessCell {
                data,
                clk: F::zero(),
            });
    }

    pub fn generate_memory_interface_trace(&self) -> RowMajorMatrix<F> {
        let all_addresses = self.interface_chip.all_addresses();
        let mut final_memory = HashMap::new();
        for (addr_space, pointer) in all_addresses {
            final_memory.insert(
                (addr_space, pointer),
                *self.memory.get(&(addr_space, pointer)).unwrap(),
            );
        }

        self.interface_chip.generate_trace(final_memory)
    }

    /// Trace generation for dummy values when a memory operation should be selectively disabled.
    ///
    /// Warning: `self.clk` must be > 0 for less than constraints to pass.
    pub fn disabled_op(&mut self, addr_space: F, op_type: OpType) -> MemoryAccess<WORD_SIZE, F> {
        let timestamp = self.timestamp;
        // Below, we set timestamp_prev = 0
        MemoryAccess::<WORD_SIZE, F>::new(
            MemoryOperation::new(
                addr_space,
                F::zero(),
                F::from_canonical_u8(op_type as u8),
                AccessCell::new([F::zero(); WORD_SIZE], timestamp),
                F::zero(),
            ),
            AccessCell::new([F::zero(); WORD_SIZE], F::zero()),
        )
    }

    pub fn increment_timestamp(&mut self) {
        self.timestamp += F::one();
    }

    pub fn timestamp(&self) -> F {
        self.timestamp
    }

    pub fn get_audit_air(&self) -> MemoryAuditAir<WORD_SIZE> {
        match &self.interface_chip {
            MemoryInterface::Volatile(chip) => chip.air.clone(),
        }
    }

    // /// Reads an element directly from memory without updating internal state.
    // ///
    // /// Any value returned is unconstrained.
    // pub fn unsafe_read_elem(&self, address_space: F, address: F) -> F {
    //     compose(self.unsafe_read_word(address_space, address))
    // }

    // pub fn write_elem(&mut self, timestamp: usize, address_space: F, address: F, data: F) {
    //     self.write_word(timestamp, address_space, address, decompose(data));
    // }
}
