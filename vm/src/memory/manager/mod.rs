use std::{cell::RefCell, collections::HashMap, rc::Rc, sync::Arc};

use afs_primitives::{range_gate::RangeCheckerGateChip, sub_chip::LocalTraceInstructions};
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
        manager::operation::MemoryOperation,
        offline_checker::{
            bridge::MemoryOfflineChecker,
            bus::MemoryBus,
            columns::{MemoryOfflineCheckerAuxCols, MemoryReadAuxCols, MemoryWriteAuxCols},
        },
        OpType,
    },
    vm::config::MemoryConfig,
};

pub mod access_cell;
pub mod dimensions;
pub mod interface;
pub mod operation;
pub mod trace_builder;

const NUM_WORDS: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct TimestampedValue<T> {
    pub timestamp: T,
    pub value: T,
}

/// Represents a single or batch memory read operation.
/// Can be used to generate [MemoryReadAuxCols].
#[derive(Clone, Debug)]
pub struct MemoryRead<const N: usize, T> {
    /// The address space in which the read operation occurs.
    pub address_space: T,
    /// The pointer indicating the memory location being read.
    pub pointer: T,
    /// The timestamp of the current read operation.
    pub timestamp: T,
    /// The timestamp of the previous batch access to this location.
    pub prev_timestamp: T,
    /// The data read from memory.
    pub data: [T; N],
}

impl<T: Copy> MemoryRead<1, T> {
    pub fn value(&self) -> T {
        self.data[0]
    }
}

impl<const N: usize, F: Field> MemoryRead<N, F> {
    /// Will be deprecated.
    pub fn disabled(timestamp: F, address_space: F) -> Self {
        Self {
            address_space,
            pointer: F::zero(),
            timestamp,
            prev_timestamp: F::zero(),
            data: [F::zero(); N],
        }
    }
}

/// Represents a single or batch memory write operation.
/// Can be used to generate [MemoryWriteAuxCols].
#[derive(Clone, Debug)]
pub struct MemoryWrite<const N: usize, T> {
    /// The address space in which the write operation occurs.
    pub address_space: T,
    /// The pointer indicating the memory location being written to.
    pub pointer: T,
    /// The timestamp of the current write operation.
    pub timestamp: T,
    /// The timestamp of the previous batch access to this location.
    pub prev_timestamp: T,
    /// The data to be written to memory.
    pub data: [T; N],
    /// The data that existed at the memory location during the previous batch access.
    pub prev_data: [T; N],
}

impl<const N: usize, F: Field> MemoryWrite<N, F> {
    /// Will be deprecated.
    pub fn disabled(timestamp: F, address_space: F) -> Self {
        Self {
            address_space,
            pointer: F::zero(),
            timestamp,
            prev_timestamp: F::zero(),
            data: [F::zero(); N],
            prev_data: [F::zero(); N],
        }
    }
}

impl<T: Copy> MemoryWrite<1, T> {
    pub fn value(&self) -> T {
        self.data[0]
    }
}

pub type MemoryChipRef<F> = Rc<RefCell<MemoryChip<F>>>;

#[derive(Clone, Debug)]
pub struct MemoryChip<F: PrimeField32> {
    pub memory_bus: MemoryBus,
    pub interface_chip: MemoryInterface<NUM_WORDS, F>,
    mem_config: MemoryConfig,
    pub(crate) range_checker: Arc<RangeCheckerGateChip>,
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

    pub fn read(&mut self, address_space: F, pointer: F) -> MemoryRead<1, F> {
        let timestamp = self.timestamp;
        self.timestamp += F::one();

        if address_space == F::zero() {
            return MemoryRead {
                address_space,
                pointer,
                timestamp,
                prev_timestamp: F::zero(),
                data: [pointer],
            };
        }

        let timestamped_value = self.memory.get_mut(&(address_space, pointer)).unwrap();
        debug_assert!(timestamped_value.timestamp < timestamp);

        let prev_timestamp = timestamped_value.timestamp;
        timestamped_value.timestamp = timestamp;

        self.interface_chip
            .touch_address(address_space, pointer, timestamped_value.value);

        MemoryRead {
            address_space,
            pointer,
            timestamp,
            prev_timestamp,
            data: [timestamped_value.value],
        }
    }

    /// Reads a word directly from memory without updating internal state.
    ///
    /// Any value returned is unconstrained.
    pub fn unsafe_read_cell(&self, addr_space: F, pointer: F) -> F {
        self.memory.get(&(addr_space, pointer)).unwrap().value
    }

    pub fn write(&mut self, address_space: F, pointer: F, data: F) -> MemoryWrite<1, F> {
        assert_ne!(address_space, F::zero());

        let timestamp = self.timestamp;
        self.timestamp += F::one();

        let cell = self
            .memory
            .entry((address_space, pointer))
            .or_insert(TimestampedValue {
                value: F::zero(),
                timestamp: F::zero(),
            });
        let (prev_timestamp, old_data) = (cell.timestamp, cell.value);
        assert!(prev_timestamp < timestamp);

        // Updating AccessCell
        cell.timestamp = timestamp;
        cell.value = data;

        self.interface_chip
            .touch_address(address_space, pointer, old_data);

        MemoryWrite {
            address_space,
            pointer,
            timestamp,
            prev_timestamp,
            data: [data],
            prev_data: [old_data],
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
        read: MemoryRead<N, F>,
    ) -> MemoryReadAuxCols<N, F> {
        let access = MemoryAccess::from_read(read);
        self.make_access_cols(access)
    }

    pub fn make_write_aux_cols<const N: usize>(
        &self,
        write: MemoryWrite<N, F>,
    ) -> MemoryWriteAuxCols<N, F> {
        let access = MemoryAccess::from_write(write);
        self.make_access_cols(access)
    }

    // Deprecated.
    pub fn make_access_cols<const N: usize>(
        &self,
        memory_access: MemoryAccess<N, F>,
    ) -> MemoryOfflineCheckerAuxCols<N, F> {
        let timestamp_prev = memory_access.old_cell.clk.as_canonical_u32();
        let timestamp = memory_access.op.cell.clk.as_canonical_u32();

        debug_assert!(timestamp_prev < timestamp);
        let offline_checker = self.make_offline_checker();
        let clk_lt_cols = LocalTraceInstructions::generate_trace_row(
            &offline_checker.timestamp_lt_air,
            (timestamp_prev, timestamp, self.range_checker.clone()),
        );

        let addr_space_is_zero_cols = offline_checker
            .is_zero_air
            .generate_trace_row(memory_access.op.addr_space);

        MemoryOfflineCheckerAuxCols::new(
            memory_access.old_cell,
            addr_space_is_zero_cols.io.is_zero,
            addr_space_is_zero_cols.inv,
            clk_lt_cols.io.less_than,
            clk_lt_cols.aux,
        )
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

#[derive(new, Clone, Debug, Default)]
pub struct MemoryAccess<const WORD_SIZE: usize, T> {
    pub op: MemoryOperation<WORD_SIZE, T>,
    pub old_cell: AccessCell<WORD_SIZE, T>,
}

impl<const WORD_SIZE: usize, T: Field> MemoryAccess<WORD_SIZE, T> {
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

    pub fn from_read(read: MemoryRead<WORD_SIZE, T>) -> Self {
        Self {
            op: MemoryOperation {
                addr_space: read.address_space,
                pointer: read.pointer,
                op_type: T::zero(),
                cell: AccessCell::new(read.data, read.timestamp),
                enabled: T::one(),
            },
            old_cell: AccessCell::new(read.data, read.prev_timestamp),
        }
    }

    pub fn from_write(write: MemoryWrite<WORD_SIZE, T>) -> Self {
        Self {
            op: MemoryOperation {
                addr_space: write.address_space,
                pointer: write.pointer,
                op_type: T::one(),
                cell: AccessCell::new(write.data, write.timestamp),
                enabled: T::one(),
            },
            old_cell: AccessCell::new(write.prev_data, write.prev_timestamp),
        }
    }
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
