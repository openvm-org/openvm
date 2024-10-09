use std::{
    array, cell::RefCell, collections::HashMap, iter, marker::PhantomData, rc::Rc, sync::Arc,
};

use afs_derive::AlignedBorrow;
use afs_primitives::{
    assert_less_than::{columns::AssertLessThanAuxCols, AssertLessThanAir},
    is_less_than::IsLessThanAir,
    is_zero::IsZeroAir,
    sub_chip::LocalTraceInstructions,
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};
use afs_stark_backend::rap::AnyRap;
pub use memory::{AddressSpace, MemoryReadRecord, MemoryWriteRecord};
use p3_air::BaseAir;
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
    arch::{compose, MachineChip, RV32_REGISTER_NUM_LANES},
    core::RANGE_CHECKER_BUS,
    memory::{
        adapter::AccessAdapterAir,
        manager::memory::{AccessAdapterRecord, Memory},
        offline_checker::{
            MemoryBridge, MemoryBus, MemoryReadAuxCols, MemoryReadOrImmediateAuxCols,
            MemoryWriteAuxCols, AUX_LEN,
        },
        tree::Hasher,
    },
    vm::config::{MemoryConfig, PersistenceType},
};

pub mod dimensions;
mod interface;
mod memory;
mod trace;

const NUM_WORDS: usize = 16;
pub const CHUNK: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct TimestampedValue<T> {
    pub timestamp: T,
    pub value: T,
}

/// Represents first reads a pointer, and then a batch read at the pointer.
#[derive(Clone, Debug)]
pub struct MemoryHeapReadRecord<T, const N: usize> {
    pub address_read: MemoryReadRecord<T, 1>,
    pub data_read: MemoryReadRecord<T, N>,
}

/// Represents first reads a RV register, and then a batch read at the pointer.
#[derive(Clone, Debug)]
pub struct RegisterHeapReadRecord<T, const N: usize> {
    pub address_read: MemoryReadRecord<T, RV32_REGISTER_NUM_LANES>,
    pub data_read: MemoryReadRecord<T, N>,
}

/// Represents first reads a pointer, and then a batch write at the pointer.
#[derive(Clone, Debug)]
pub struct MemoryHeapWriteRecord<T, const N: usize> {
    pub address_read: MemoryReadRecord<T, 1>,
    pub data_write: MemoryWriteRecord<T, N>,
}

/// Represents first reads a RV register, and then a batch write at the pointer.
#[derive(Clone, Debug)]
pub struct RegisterHeapWriteRecord<T, const N: usize> {
    pub address_read: MemoryReadRecord<T, RV32_REGISTER_NUM_LANES>,
    pub data_write: MemoryWriteRecord<T, N>,
}

/// Holds the data and the information about its address.
#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
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
#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
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
    // addr_space -> Memory data structure
    memory: Memory<F>,
    /// Maps a length to a list of access adapters with that block length as th larger size.
    adapter_records: HashMap<usize, Vec<AccessAdapterRecord<F>>>,
}

impl<F: PrimeField32> MemoryChip<F> {
    pub fn new(
        memory_bus: MemoryBus,
        mem_config: MemoryConfig,
        range_checker: Arc<VariableRangeCheckerChip>,
    ) -> Self {
        Self {
            memory_bus,
            mem_config: mem_config.clone(),
            interface_chip: MemoryInterface::Volatile(MemoryAuditChip::new(
                memory_bus,
                mem_config.addr_space_max_bits,
                mem_config.pointer_max_bits,
                mem_config.decomp,
                range_checker.clone(),
            )),
            memory: Memory::new(1 << mem_config.pointer_max_bits, 1),
            adapter_records: HashMap::new(),
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
            address_space == F::zero()
                || pointer.as_canonical_u32() < (1 << self.mem_config.pointer_max_bits),
            "memory out of bounds: {:?}",
            pointer.as_canonical_u32()
        );

        if address_space == F::zero() {
            assert_eq!(N, 1, "cannot batch read from address space 0");

            let timestamp = self.timestamp();
            self.memory.increment_timestamp();

            return MemoryReadRecord {
                address_space,
                pointer,
                timestamp,
                prev_timestamp: F::zero(),
                data: array::from_fn(|_| pointer),
            };
        }

        let (record, adapter_records) = self.memory.read::<N>(
            AddressSpace(address_space.as_canonical_u32()),
            pointer.as_canonical_u32() as usize,
        );
        for record in adapter_records {
            self.adapter_records
                .entry(record.data.len())
                .or_default()
                .push(record);
        }

        for (i, value) in record.data.iter().enumerate() {
            let ptr = pointer + F::from_canonical_usize(i);
            self.interface_chip
                .touch_address(address_space, ptr, *value);
        }

        record
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

    pub fn read_register(
        &mut self,
        address_space: F,
        pointer: F,
    ) -> (MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>, u32) {
        debug_assert_eq!(address_space, F::one());
        let record = self.read::<RV32_REGISTER_NUM_LANES>(address_space, pointer);
        let val = compose(record.data);
        (record, val)
    }

    /// First lookup the heap pointer from register, and then read the data at the pointer.
    pub fn read_heap_from_register<const N: usize>(
        &mut self,
        ptr_address_space: F,
        data_address_space: F,
        ptr_pointer: F,
    ) -> RegisterHeapReadRecord<F, N> {
        let (address_read, val) = self.read_register(ptr_address_space, ptr_pointer);
        let data_read = self.read(data_address_space, F::from_canonical_u32(val));

        RegisterHeapReadRecord {
            address_read,
            data_read,
        }
    }

    /// Reads a word directly from memory without updating internal state.
    ///
    /// Any value returned is unconstrained.
    pub fn unsafe_read_cell(&self, addr_space: F, pointer: F) -> F {
        let (_, &value) = self
            .memory
            .get(
                AddressSpace(addr_space.as_canonical_u32()),
                pointer.as_canonical_u32() as usize,
            )
            .unwrap();
        value
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
            pointer.as_canonical_u32() < (1 << self.mem_config.pointer_max_bits),
            "memory out of bounds: {:?}",
            pointer.as_canonical_u32()
        );

        let (record, adapter_records) = self.memory.write(
            AddressSpace(address_space.as_canonical_u32()),
            pointer.as_canonical_u32() as usize,
            data,
        );
        for record in adapter_records {
            self.adapter_records
                .entry(record.data.len())
                .or_default()
                .push(record);
        }

        for (i, value) in record.prev_data.iter().enumerate() {
            let ptr = pointer + F::from_canonical_usize(i);
            self.interface_chip
                .touch_address(address_space, ptr, *value);
        }

        record
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

    /// First lookup the heap pointer from register, and then write the data at the pointer.
    pub fn write_heap_from_register<const N: usize>(
        &mut self,
        ptr_address_space: F,
        data_address_space: F,
        ptr_pointer: F,
        data: [F; N],
    ) -> RegisterHeapWriteRecord<F, N> {
        let (address_read, val) = self.read_register(ptr_address_space, ptr_pointer);
        let data_write = self.write(data_address_space, F::from_canonical_u32(val), data);

        RegisterHeapWriteRecord {
            address_read,
            data_write,
        }
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

    pub fn increment_timestamp(&mut self) {
        self.memory.increment_timestamp();
    }

    pub fn increment_timestamp_by(&mut self, change: F) {
        self.memory
            .increment_timestamp_by(change.as_canonical_u32());
    }

    pub fn increase_timestamp_to(&mut self, timestamp: F) {
        let timestamp = timestamp.as_canonical_u32();
        self.memory
            .increment_timestamp_by(timestamp - self.memory.timestamp());
    }

    pub fn timestamp(&self) -> F {
        F::from_canonical_u32(self.memory.timestamp())
    }

    pub fn get_audit_air(&self) -> MemoryAuditAir {
        match &self.interface_chip {
            MemoryInterface::Volatile(chip) => chip.air.clone(),
        }
    }

    pub fn access_adapter_air<const N: usize>(&self) -> AccessAdapterAir<N> {
        let lt_air = IsLessThanAir::new(self.range_checker.bus(), self.mem_config.clk_max_bits);
        AccessAdapterAir::<N> {
            memory_bus: self.memory_bus,
            lt_air,
        }
    }

    pub fn finalize(&mut self, hasher: Option<&mut impl Hasher<CHUNK, F>>) {
        if let Some(_hasher) = hasher {
            assert_eq!(
                self.mem_config.persistence_type,
                PersistenceType::Persistent
            );
            todo!("finalize persistent memory");
        } else {
            assert_eq!(self.mem_config.persistence_type, PersistenceType::Volatile);

            let all_addresses = self.interface_chip.all_addresses();
            for (address_space, pointer) in all_addresses {
                let records = self.memory.access(
                    AddressSpace(address_space.as_canonical_u32()),
                    pointer.as_canonical_u32() as usize,
                    1,
                );
                for record in records {
                    self.adapter_records
                        .entry(record.data.len())
                        .or_default()
                        .push(record);
                }
            }
        }
    }
}

impl<F: PrimeField32> MachineChip<F> for MemoryChip<F> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        panic!("cannot call generate_trace on MemoryChip, which has more than one trace");
    }
    fn generate_traces(self) -> Vec<RowMajorMatrix<F>> {
        vec![
            self.generate_memory_interface_trace(),
            self.generate_access_adapter_trace::<2>(),
            self.generate_access_adapter_trace::<4>(),
            self.generate_access_adapter_trace::<8>(),
            self.generate_access_adapter_trace::<16>(),
            self.generate_access_adapter_trace::<32>(),
            self.generate_access_adapter_trace::<64>(),
        ]
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        panic!("cannot call air on MemoryChip, which has more than one air");
    }
    fn airs<SC: StarkGenericConfig>(&self) -> Vec<Box<dyn AnyRap<SC>>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        vec![
            Box::new(self.get_audit_air()),
            Box::new(self.access_adapter_air::<2>()),
            Box::new(self.access_adapter_air::<4>()),
            Box::new(self.access_adapter_air::<8>()),
            Box::new(self.access_adapter_air::<16>()),
            Box::new(self.access_adapter_air::<32>()),
            Box::new(self.access_adapter_air::<64>()),
        ]
    }

    fn air_name(&self) -> String {
        panic!("cannot call air_name on MemoryChip, which has more than one trace");
    }
    fn air_names(&self) -> Vec<String> {
        vec![
            "Audit".to_string(),
            "AccessAdapter<2>".to_string(),
            "AccessAdapter<4>".to_string(),
            "AccessAdapter<8>".to_string(),
            "AccessAdapter<16>".to_string(),
            "AccessAdapter<32>".to_string(),
            "AccessAdapter<64>".to_string(),
        ]
    }

    fn current_trace_height(&self) -> usize {
        panic!("cannot call current_trace_height on MemoryChip, which has more than one trace");
    }
    fn current_trace_heights(&self) -> Vec<usize> {
        vec![
            self.interface_chip.current_height(),
            self.adapter_records
                .get(&2)
                .map_or(0, |records| records.len()),
            self.adapter_records
                .get(&4)
                .map_or(0, |records| records.len()),
            self.adapter_records
                .get(&8)
                .map_or(0, |records| records.len()),
            self.adapter_records
                .get(&16)
                .map_or(0, |records| records.len()),
            self.adapter_records
                .get(&32)
                .map_or(0, |records| records.len()),
            self.adapter_records
                .get(&64)
                .map_or(0, |records| records.len()),
        ]
    }

    fn trace_width(&self) -> usize {
        panic!("cannot call trace_width on MemoryChip, which has more than one trace");
    }
    fn trace_widths(&self) -> Vec<usize> {
        vec![
            self.get_audit_air().air_width(),
            BaseAir::<F>::width(&self.access_adapter_air::<2>()),
            BaseAir::<F>::width(&self.access_adapter_air::<4>()),
            BaseAir::<F>::width(&self.access_adapter_air::<8>()),
            BaseAir::<F>::width(&self.access_adapter_air::<16>()),
            BaseAir::<F>::width(&self.access_adapter_air::<32>()),
            BaseAir::<F>::width(&self.access_adapter_air::<64>()),
        ]
    }

    fn generate_public_values(&mut self) -> Vec<F> {
        panic!("cannot call generate_public_values on MemoryChip, which has more than one trace");
    }
    fn generate_public_values_per_air(&mut self) -> Vec<Vec<F>> {
        vec![vec![]; 7]
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
            read.prev_timestamp,
            self.generate_timestamp_lt_cols(read.prev_timestamp, read.timestamp),
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
        let addr_space_is_zero_cols = IsZeroAir.generate_trace_row(read.address_space);
        let timestamp_lt_cols =
            self.generate_timestamp_lt_cols(read.prev_timestamp, read.timestamp);

        MemoryReadOrImmediateAuxCols::new(
            read.prev_timestamp,
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
            write.prev_timestamp,
            self.generate_timestamp_lt_cols(write.prev_timestamp, write.timestamp),
        )
    }

    fn generate_timestamp_lt_cols(
        &self,
        prev_timestamp: F,
        timestamp: F,
    ) -> AssertLessThanAuxCols<F, AUX_LEN> {
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
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use afs_primitives::var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip};
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use rand::{prelude::SliceRandom, thread_rng, Rng};

    use super::MemoryChip;
    use crate::{
        core::RANGE_CHECKER_BUS, memory::offline_checker::MemoryBus, vm::config::MemoryConfig,
    };

    #[test]
    fn test_no_adapter_records_for_singleton_accesses() {
        type F = BabyBear;

        let memory_bus = MemoryBus(1);
        let memory_config = MemoryConfig::default();
        let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, memory_config.decomp);
        let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let mut memory_chip =
            MemoryChip::new(memory_bus, memory_config.clone(), range_checker.clone());

        let mut rng = thread_rng();
        for _ in 0..1000 {
            let address_space = F::from_canonical_u32(*[1, 2].choose(&mut rng).unwrap());
            let pointer =
                F::from_canonical_u32(rng.gen_range(0..1 << memory_config.pointer_max_bits));

            if rng.gen_bool(0.5) {
                let data = F::from_canonical_u32(rng.gen_range(0..1 << 30));
                memory_chip.write(address_space, pointer, [data]);
            } else {
                memory_chip.read::<1>(address_space, pointer);
            }
        }
        assert_eq!(memory_chip.adapter_records.len(), 0);
    }
}
