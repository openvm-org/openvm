use std::{
    array, cell::RefCell, collections::HashMap, iter, marker::PhantomData, rc::Rc, sync::Arc,
};

use afs_primitives::{
    assert_less_than::{columns::AssertLessThanAuxCols, AssertLessThanAir},
    is_zero::IsZeroAir,
    sub_chip::LocalTraceInstructions,
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};
use afs_stark_backend::rap::AnyRap;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use self::interface::MemoryInterface;
use super::{
    audit::{air::MemoryAuditAir, MemoryAuditChip},
    offline_checker::{MemoryHeapReadAuxCols, MemoryHeapWriteAuxCols},
};
use crate::{
    arch::chips::MachineChip,
    cpu::RANGE_CHECKER_BUS,
    memory::offline_checker::{
        MemoryBridge, MemoryBus, MemoryReadAuxCols, MemoryReadOrImmediateAuxCols,
        MemoryWriteAuxCols, AUX_LEN,
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
pub struct MemoryReadRecord<T, const N: usize> {
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

impl<T: Copy> MemoryReadRecord<T, 1> {
    pub fn value(&self) -> T {
        self.data[0]
    }
}

/// Represents first reads a pointer, and then a batch read at the pointer.
#[derive(Clone, Debug)]
pub struct MemoryHeapReadRecord<T, const N: usize> {
    pub address_read: MemoryReadRecord<T, 1>,
    pub data_read: MemoryReadRecord<T, N>,
}

/// Represents first reads a pointer, and then a batch write at the pointer.
#[derive(Clone, Debug)]
pub struct MemoryHeapWriteRecord<T, const N: usize> {
    pub address_read: MemoryReadRecord<T, 1>,
    pub data_write: MemoryWriteRecord<T, N>,
}

/// Represents a single or batch memory write operation.
/// Can be used to generate [MemoryWriteAuxCols].
#[derive(Clone, Debug)]
pub struct MemoryWriteRecord<T, const N: usize> {
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

impl<T: Copy> MemoryWriteRecord<T, 1> {
    pub fn value(&self) -> T {
        self.data[0]
    }
}

/// Holds the data and the information about its address.
#[derive(Clone, Debug)]
pub struct MemoryDataIoCols<T, const N: usize> {
    pub data: [T; N],
    pub address_space: T,
    pub address: T,
}

impl<T: Clone, const N: usize> MemoryDataIoCols<T, N> {
    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        Self {
            data: std::array::from_fn(|_| iter.next().unwrap()),
            address_space: iter.next().unwrap(),
            address: iter.next().unwrap(),
        }
    }

    pub fn flatten(&self) -> impl Iterator<Item = &T> {
        self.data
            .iter()
            .chain(iter::once(&self.address_space))
            .chain(iter::once(&self.address))
    }
}

/// Holds the heap data and the information about its address.
#[derive(Clone, Debug)]
pub struct MemoryHeapDataIoCols<T: Clone, const N: usize> {
    pub address: MemoryDataIoCols<T, 1>,
    pub data: MemoryDataIoCols<T, N>,
}

impl<T: Clone, const N: usize> MemoryHeapDataIoCols<T, N> {
    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        Self {
            address: MemoryDataIoCols::from_iterator(iter.by_ref()),
            data: MemoryDataIoCols::from_iterator(iter.by_ref()),
        }
    }

    pub fn flatten(&self) -> impl Iterator<Item = &T> {
        self.address.flatten().chain(self.data.flatten())
    }
}

impl<T: Clone, const N: usize> From<MemoryHeapReadRecord<T, N>> for MemoryHeapDataIoCols<T, N> {
    fn from(record: MemoryHeapReadRecord<T, N>) -> Self {
        Self {
            address: MemoryDataIoCols {
                data: record.address_read.data,
                address_space: record.address_read.address_space,
                address: record.address_read.pointer,
            },
            data: MemoryDataIoCols {
                data: record.data_read.data,
                address_space: record.data_read.address_space,
                address: record.data_read.pointer,
            },
        }
    }
}

impl<T: Clone, const N: usize> From<MemoryHeapWriteRecord<T, N>> for MemoryHeapDataIoCols<T, N> {
    fn from(record: MemoryHeapWriteRecord<T, N>) -> Self {
        Self {
            address: MemoryDataIoCols {
                data: record.address_read.data,
                address_space: record.address_read.address_space,
                address: record.address_read.pointer,
            },
            data: MemoryDataIoCols {
                data: record.data_write.data,
                address_space: record.data_write.address_space,
                address: record.data_write.pointer,
            },
        }
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

pub const MEMORY_TOP: u32 = (1 << 29) - 1;

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

    pub fn memory_bridge(&self) -> MemoryBridge {
        MemoryBridge::new(
            self.memory_bus,
            self.mem_config.clk_max_bits,
            self.mem_config.decomp,
        )
    }

    pub fn read_cell(&mut self, address_space: F, pointer: F) -> MemoryReadRecord<F, 1> {
        self.read(address_space, pointer)
    }

    pub fn read<const N: usize>(&mut self, address_space: F, pointer: F) -> MemoryReadRecord<F, N> {
        assert!(
            address_space == F::zero() || pointer.as_canonical_u32() <= MEMORY_TOP,
            "memory out of bounds: {:?}",
            pointer.as_canonical_u32()
        );

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

            let entry = self
                .memory
                .get_mut(&(address_space, cur_ptr))
                .unwrap_or_else(|| {
                    panic!("read of uninitialized memory ({address_space:?}, {cur_ptr:?})")
                });
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

    /// First lookup the heap pointer, and then read the data at the pointer.
    pub fn read_heap<const N: usize>(
        &mut self,
        ptr_address_space: F,
        data_address_space: F,
        ptr_pointer: F,
    ) -> MemoryHeapReadRecord<F, N> {
        let address_read = self.read_cell(ptr_address_space, ptr_pointer);
        let data_read = self.read(data_address_space, address_read.value());

        MemoryHeapReadRecord {
            address_read,
            data_read,
        }
    }

    /// Reads a word directly from memory without updating internal state.
    ///
    /// Any value returned is unconstrained.
    pub fn unsafe_read_cell(&self, addr_space: F, pointer: F) -> F {
        self.memory.get(&(addr_space, pointer)).unwrap().value
    }

    pub fn write_cell(&mut self, address_space: F, pointer: F, data: F) -> MemoryWriteRecord<F, 1> {
        self.write(address_space, pointer, [data])
    }

    pub fn write<const N: usize>(
        &mut self,
        address_space: F,
        pointer: F,
        data: [F; N],
    ) -> MemoryWriteRecord<F, N> {
        assert_ne!(address_space, F::zero());
        assert!(
            address_space == F::zero() || pointer.as_canonical_u32() <= MEMORY_TOP,
            "memory out of bounds: {:?}",
            pointer.as_canonical_u32()
        );

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

    /// First lookup the heap pointer, and then write the data at the pointer.
    pub fn write_heap<const N: usize>(
        &mut self,
        ptr_address_space: F,
        data_address_space: F,
        ptr_pointer: F,
        data: [F; N],
    ) -> MemoryHeapWriteRecord<F, N> {
        let address_read = self.read_cell(ptr_address_space, ptr_pointer);
        let data_write = self.write(data_address_space, address_read.value(), data);

        MemoryHeapWriteRecord {
            address_read,
            data_write,
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

    pub fn aux_cols_factory(&self) -> MemoryAuxColsFactory<F> {
        let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, self.mem_config.decomp);
        MemoryAuxColsFactory {
            range_checker: self.range_checker.clone(),
            timestamp_lt_air: AssertLessThanAir::<AUX_LEN>::new(
                range_bus,
                self.mem_config.clk_max_bits,
            ),
            _marker: Default::default(),
        }
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

pub struct MemoryAuxColsFactory<T> {
    range_checker: Arc<VariableRangeCheckerChip>,
    timestamp_lt_air: AssertLessThanAir<AUX_LEN>,
    _marker: PhantomData<T>,
}

// NOTE[jpw]: The `make_*_aux_cols` functions should be thread-safe so they can be used in parallelized trace generation.
impl<F: PrimeField32> MemoryAuxColsFactory<F> {
    pub fn make_read_aux_cols<const N: usize>(
        &self,
        read: MemoryReadRecord<F, N>,
    ) -> MemoryReadAuxCols<F, N> {
        assert!(
            !read.address_space.is_zero(),
            "cannot make `MemoryReadAuxCols` for address space 0"
        );
        MemoryReadAuxCols::new(
            read.prev_timestamps,
            self.generate_timestamp_lt_cols(&read.prev_timestamps, read.timestamp),
        )
    }

    pub fn make_heap_read_aux_cols<const N: usize>(
        &self,
        read: MemoryHeapReadRecord<F, N>,
    ) -> MemoryHeapReadAuxCols<F, N> {
        MemoryHeapReadAuxCols {
            address: self.make_read_aux_cols(read.address_read),
            data: self.make_read_aux_cols(read.data_read),
        }
    }

    pub fn make_heap_write_aux_cols<const N: usize>(
        &self,
        write: MemoryHeapWriteRecord<F, N>,
    ) -> MemoryHeapWriteAuxCols<F, N> {
        MemoryHeapWriteAuxCols {
            address: self.make_read_aux_cols(write.address_read),
            data: self.make_write_aux_cols(write.data_write),
        }
    }

    pub fn make_read_or_immediate_aux_cols(
        &self,
        read: MemoryReadRecord<F, 1>,
    ) -> MemoryReadOrImmediateAuxCols<F> {
        let [prev_timestamp] = read.prev_timestamps;

        let addr_space_is_zero_cols = IsZeroAir.generate_trace_row(read.address_space);
        let [timestamp_lt_cols] =
            self.generate_timestamp_lt_cols(&[prev_timestamp], read.timestamp);

        MemoryReadOrImmediateAuxCols::new(
            prev_timestamp,
            addr_space_is_zero_cols.io.is_zero,
            addr_space_is_zero_cols.inv,
            timestamp_lt_cols,
        )
    }

    pub fn make_write_aux_cols<const N: usize>(
        &self,
        write: MemoryWriteRecord<F, N>,
    ) -> MemoryWriteAuxCols<F, N> {
        MemoryWriteAuxCols::new(
            write.prev_data,
            write.prev_timestamps,
            self.generate_timestamp_lt_cols(&write.prev_timestamps, write.timestamp),
        )
    }

    fn generate_timestamp_lt_cols<const N: usize>(
        &self,
        prev_timestamps: &[F; N],
        timestamp: F,
    ) -> [AssertLessThanAuxCols<F, AUX_LEN>; N] {
        prev_timestamps.map(|prev_timestamp| {
            debug_assert!(prev_timestamp.as_canonical_u32() < timestamp.as_canonical_u32());
            let mut aux: AssertLessThanAuxCols<F, AUX_LEN> =
                AssertLessThanAuxCols::<F, AUX_LEN>::new([F::zero(); AUX_LEN]);
            self.timestamp_lt_air.generate_trace_row_aux(
                prev_timestamp.as_canonical_u32(),
                timestamp.as_canonical_u32(),
                &self.range_checker,
                &mut aux,
            );
            aux
        })
    }
}
