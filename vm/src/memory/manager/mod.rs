use std::{array, cell::RefCell, collections::HashMap, rc::Rc, sync::Arc};

use afs_primitives::var_range::VariableRangeCheckerChip;
use afs_stark_backend::rap::AnyRap;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use self::interface::MemoryInterface;
use super::audit::{air::MemoryAuditAir, MemoryAuditChip};
use crate::{
    arch::chips::MachineChip,
    memory::offline_checker::{
        bridge::MemoryOfflineChecker,
        bus::MemoryBus,
        columns::{MemoryReadAuxCols, MemoryWriteAuxCols},
    },
    vm::config::MemoryConfig,
};

pub mod dimensions;
pub mod interface;

const NUM_WORDS: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct TimestampedValue<T> {
    pub timestamp: T,
    pub value: T,
}

/// Represents a single or batch memory read operation.
///
/// Also used for "reads" from address space 0 (immediates).
/// Can be used to generate [MemoryReadAuxCols] or [MemoryReadOrImmediateAuxCols].
#[derive(Clone, Debug)]
pub struct MemoryReadRecord<const N: usize, T> {
    /// The address space in which the read operation occurs.
    pub address_space: T,
    /// The pointer indicating the memory location being read.
    pub pointer: T,
    /// The timestamp of the current read operation.
    pub timestamp: T,
    /// The timestamp of the previous batch access to this location.
    // TODO[zach]: Should be just prev_timestamp: T.
    pub(crate) prev_timestamps: [T; N],
    /// The data read from memory.
    pub data: [T; N],
}

impl<T: Copy> MemoryReadRecord<1, T> {
    pub fn value(&self) -> T {
        self.data[0]
    }
}

/// Represents a single or batch memory write operation.
/// Can be used to generate [MemoryWriteAuxCols].
#[derive(Clone, Debug)]
pub struct MemoryWriteRecord<const N: usize, T> {
    /// The address space in which the write operation occurs.
    pub address_space: T,
    /// The pointer indicating the memory location being written to.
    pub pointer: T,
    /// The timestamp of the current write operation.
    pub timestamp: T,
    /// The timestamp of the previous batch access to this location.
    pub(crate) prev_timestamps: [T; N],
    /// The data to be written to memory.
    pub data: [T; N],
    /// The data that existed at the memory location during the previous batch access.
    pub(crate) prev_data: [T; N],
}

impl<T: Copy> MemoryWriteRecord<1, T> {
    pub fn value(&self) -> T {
        self.data[0]
    }
}

pub type MemoryChipRef<F> = Rc<RefCell<MemoryChip<F>>>;

#[derive(Clone, Debug)]
pub struct MemoryChip<F: PrimeField32> {
    pub memory_bus: MemoryBus,
    pub interface_chip: MemoryInterface<NUM_WORDS, F>,
    pub(crate) mem_config: MemoryConfig,
    pub(crate) range_checker: Arc<VariableRangeCheckerChip>,
    timestamp: F,
    /// Maps (addr_space, pointer) to (data, timestamp)
    memory: HashMap<(F, F), TimestampedValue<F>>,
}

impl<F: PrimeField32> MemoryChip<F> {
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
        range_checker: Arc<VariableRangeCheckerChip>,
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

    pub fn read_cell(&mut self, address_space: F, pointer: F) -> MemoryReadRecord<1, F> {
        self.read(address_space, pointer)
    }

    pub fn read<const N: usize>(&mut self, address_space: F, pointer: F) -> MemoryReadRecord<N, F> {
        let timestamp = self.timestamp;
        self.timestamp += F::one();

        if address_space == F::zero() {
            assert_eq!(N, 1, "cannot batch read from address space 0");

            return MemoryReadRecord {
                address_space,
                pointer,
                timestamp,
                prev_timestamps: [F::zero(); N],
                data: array::from_fn(|_| pointer),
            };
        }

        let prev_entries = array::from_fn(|i| {
            let cur_ptr = pointer + F::from_canonical_usize(i);

            let entry = self.memory.get_mut(&(address_space, cur_ptr)).unwrap();
            debug_assert!(entry.timestamp < timestamp);

            let prev_entry = *entry;
            entry.timestamp = timestamp;

            self.interface_chip
                .touch_address(address_space, cur_ptr, entry.value);

            prev_entry
        });

        MemoryReadRecord {
            address_space,
            pointer,
            timestamp,
            prev_timestamps: prev_entries.map(|entry| entry.timestamp),
            data: prev_entries.map(|entry| entry.value),
        }
    }

    /// Reads a word directly from memory without updating internal state.
    ///
    /// Any value returned is unconstrained.
    pub fn unsafe_read_cell(&self, addr_space: F, pointer: F) -> F {
        self.memory.get(&(addr_space, pointer)).unwrap().value
    }

    pub fn write_cell(&mut self, address_space: F, pointer: F, data: F) -> MemoryWriteRecord<1, F> {
        self.write(address_space, pointer, [data])
    }

    pub fn write<const N: usize>(
        &mut self,
        address_space: F,
        pointer: F,
        data: [F; N],
    ) -> MemoryWriteRecord<N, F> {
        assert_ne!(address_space, F::zero());

        let timestamp = self.timestamp;
        self.timestamp += F::one();

        let prev_entries = array::from_fn(|i| {
            let cur_ptr = pointer + F::from_canonical_usize(i);

            let entry = self
                .memory
                .entry((address_space, cur_ptr))
                .or_insert(TimestampedValue {
                    value: F::zero(),
                    timestamp: F::zero(),
                });
            debug_assert!(entry.timestamp < timestamp);

            let prev_entry = *entry;

            entry.timestamp = timestamp;
            entry.value = data[i];

            self.interface_chip
                .touch_address(address_space, cur_ptr, prev_entry.value);

            prev_entry
        });

        MemoryWriteRecord {
            address_space,
            pointer,
            timestamp,
            prev_timestamps: prev_entries.map(|entry| entry.timestamp),
            data,
            prev_data: prev_entries.map(|entry| entry.value),
        }
    }

    pub fn unsafe_write_cell(&mut self, addr_space: F, pointer: F, data: F) {
        assert_ne!(addr_space, F::zero());

        self.memory
            .entry((addr_space, pointer))
            .and_modify(|cell| cell.value = data)
            .or_insert(TimestampedValue {
                value: data,
                timestamp: F::zero(),
            });
    }

    pub fn make_read_aux_cols<const N: usize>(
        &self,
        read: MemoryReadRecord<N, F>,
    ) -> MemoryReadAuxCols<N, F> {
        self.make_offline_checker()
            .make_read_aux_cols(self.range_checker.clone(), read)
    }

    pub fn make_disabled_read_aux_cols<const N: usize>(&self) -> MemoryReadAuxCols<N, F> {
        MemoryReadAuxCols::disabled(self.make_offline_checker())
    }

    pub fn make_write_aux_cols<const N: usize>(
        &self,
        write: MemoryWriteRecord<N, F>,
    ) -> MemoryWriteAuxCols<N, F> {
        self.make_offline_checker()
            .make_write_aux_cols(self.range_checker.clone(), write)
    }

    pub fn make_disabled_write_aux_cols<const N: usize>(&self) -> MemoryWriteAuxCols<N, F> {
        MemoryWriteAuxCols::disabled(self.make_offline_checker())
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

    pub fn increment_timestamp_by(&mut self, change: F) {
        self.timestamp += change;
    }

    pub fn timestamp(&self) -> F {
        self.timestamp
    }

    /// Advance the timestamp forward to `to_timestamp`. This should be used when the memory
    /// timestamp needs to sync up with the execution state because instruction execution
    /// uses an upper bound on timestamp change.
    pub fn jump_timestamp(&mut self, to_timestamp: F) {
        debug_assert!(
            self.timestamp <= to_timestamp,
            "Should never jump back in time"
        );
        self.timestamp = to_timestamp;
    }

    pub fn get_audit_air(&self) -> MemoryAuditAir {
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

// TODO[jpw]: MemoryManager is taking the role of MemoryInterface here, which is weird.
// Necessary right now because MemoryInterface doesn't own the final memory state.
impl<F: PrimeField32> MachineChip<F> for MemoryChip<F> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
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
