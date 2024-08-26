use std::{collections::HashMap, sync::Arc};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::rap::AnyRap;
use derive_new::new;
use p3_commit::PolynomialSpace;
use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use self::{access_cell::AccessCell, interface::MemoryInterface};
use super::audit::{air::MemoryAuditAir, MemoryAuditChip};
use crate::{
    arch::chips::MachineChip,
    memory::{
        decompose,
        manager::operation::MemoryOperation,
        offline_checker::{bridge::MemoryOfflineChecker, bus::MemoryBus},
        OpType,
    },
    vm::config::MemoryConfig,
};

pub mod access_cell;
pub mod dimensions;
pub mod interface;
pub mod operation;
pub mod trace_builder;

const WORD_SIZE: usize = 1;
const NUM_WORDS: usize = 16;

#[derive(Clone, Debug)]
pub struct MemoryManager<F: PrimeField32> {
    pub memory_bus: MemoryBus,
    pub interface_chip: MemoryInterface<NUM_WORDS, WORD_SIZE, F>,
    mem_config: MemoryConfig,
    pub(crate) range_checker: Arc<RangeCheckerGateChip>,
    pub timestamp: F,
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
        memory_bus: MemoryBus,
        mem_config: MemoryConfig,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            memory_bus,
            mem_config,
            interface_chip: MemoryInterface::Volatile(MemoryAuditChip::new(
                memory_bus,
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

    pub fn make_offline_checker(&self) -> MemoryOfflineChecker {
        MemoryOfflineChecker::new(
            self.memory_bus,
            self.mem_config.clk_max_bits,
            self.mem_config.decomp,
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

        let cell = self.memory.get_mut(&(addr_space, pointer)).unwrap();
        let (old_clk, old_data) = (cell.clk, cell.data);
        debug_assert!(old_clk < cur_clk);

        // Updating AccessCell
        cell.clk = cur_clk;

        self.interface_chip
            .touch_address(addr_space, pointer, old_data);

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
            .touch_address(addr_space, pointer, old_data);

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

    // annoying function, need a proper memory testing implementation so this isn't necessary
    pub fn generate_memory_interface_trace_with_height(
        &self,
        trace_height: usize,
    ) -> RowMajorMatrix<F> {
        let all_addresses = self.interface_chip.all_addresses();
        let mut final_memory = HashMap::new();
        for (addr_space, pointer) in all_addresses {
            final_memory.insert(
                (addr_space, pointer),
                *self.memory.get(&(addr_space, pointer)).unwrap(),
            );
        }

        self.interface_chip
            .generate_trace_with_height(final_memory, trace_height)
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

#[derive(new, Clone, Debug, Default)]
pub struct MemoryAccess<const WORD_SIZE: usize, T> {
    pub op: MemoryOperation<WORD_SIZE, T>,
    pub old_cell: AccessCell<WORD_SIZE, T>,
}

impl<const WORD_SIZE: usize, T: Field> MemoryAccess<WORD_SIZE, T> {
    // TODO[jpw]: we can default to addr_space = 1 after is_immediate checks are moved out of default memory access
    pub fn disabled_read(timestamp: T, addr_space: T) -> MemoryAccess<WORD_SIZE, T> {
        Self::disabled_op(timestamp, addr_space, OpType::Read)
    }

    // TODO[jpw]: we can default to addr_space = 1 after is_immediate checks are moved out of default memory access
    pub fn disabled_write(timestamp: T, addr_space: T) -> MemoryAccess<WORD_SIZE, T> {
        Self::disabled_op(timestamp, addr_space, OpType::Write)
    }

    fn disabled_op(timestamp: T, addr_space: T, op_type: OpType) -> MemoryAccess<WORD_SIZE, T> {
        debug_assert_ne!(
            addr_space,
            T::zero(),
            "Disabled memory operation cannot be immediate"
        );
        MemoryAccess::<WORD_SIZE, T>::new(
            MemoryOperation {
                addr_space,
                pointer: T::zero(),
                op_type: T::from_canonical_u8(op_type as u8),
                cell: AccessCell::new([T::zero(); WORD_SIZE], timestamp),
                enabled: T::zero(),
            },
            AccessCell::new([T::zero(); WORD_SIZE], T::zero()),
        )
    }
}

// TODO[jpw]: MemoryManager is taking the role of MemoryInterface here, which is weird.
// Necessary right now because MemoryInterface doesn't own the final memory state.
impl<F: PrimeField32> MachineChip<F> for MemoryManager<F> {
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        self.generate_memory_interface_trace()
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.get_audit_air())
    }

    fn current_trace_height(&self) -> usize {
        self.interface_chip.current_height()
    }

    fn trace_width(&self) -> usize {
        self.get_audit_air().air_width()
    }
}
