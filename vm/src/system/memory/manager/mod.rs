use std::{
    array,
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    iter,
    marker::PhantomData,
    rc::Rc,
    sync::Arc,
};

use afs_derive::AlignedBorrow;
use afs_primitives::{
    assert_less_than::{columns::AssertLessThanAuxCols, AssertLessThanAir},
    is_less_than::IsLessThanAir,
    is_zero::IsZeroAir,
    sub_chip::LocalTraceInstructions,
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};
use afs_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    rap::AnyRap,
};
use itertools::zip_eq;
pub use memory::{MemoryReadRecord, MemoryWriteRecord};
use p3_air::BaseAir;
use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;

use self::interface::MemoryInterface;
use super::{
    offline_checker::{MemoryHeapReadAuxCols, MemoryHeapWriteAuxCols},
    volatile::VolatileBoundaryChip,
};
use crate::{
    kernels::core::RANGE_CHECKER_BUS,
    system::{
        memory::{
            adapter::AccessAdapterAir,
            manager::memory::{AccessAdapterRecord, Memory},
            offline_checker::{
                MemoryBridge, MemoryBus, MemoryReadAuxCols, MemoryReadOrImmediateAuxCols,
                MemoryWriteAuxCols, AUX_LEN,
            },
        },
        vm::config::MemoryConfig,
    },
};

pub mod dimensions;
mod interface;
pub(super) mod memory;
mod trace;

use crate::system::memory::{
    dimensions::MemoryDimensions,
    expand::{MemoryMerkleBus, MemoryMerkleChip},
    persistent::PersistentBoundaryChip,
    tree::{HasherChip, MemoryNode},
};

pub const CHUNK: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimestampedValues<T, const N: usize> {
    pub timestamp: u32,
    pub values: [T; N],
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
            data: array::from_fn(|_| iter.next().unwrap()),
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
pub struct MemoryHeapDataIoCols<T, const N: usize> {
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

#[derive(Clone, Debug)]
pub struct MemoryChipResult<F> {
    traces: Vec<RowMajorMatrix<F>>,
    public_values: Vec<Vec<F>>,
}

pub type MemoryChipRef<F> = Rc<RefCell<MemoryChip<F>>>;

/// A equipartition of memory, with timestamps and values.
///
/// The key is a pair `(address_space, label)`, where `label` is the index of the block in the
/// partition. I.e., the start of the block is `(address_space, label * N)`.
///
/// If a key is not present in the map, then the block is uninitialized (and therefore zero).
pub type MemoryEquipartition<F, const N: usize> = BTreeMap<(F, usize), TimestampedValues<F, N>>;

// TODO[zach]: We shouldn't need this; other functions should take a MemoryEquipartition instead of a HashMap.
fn partition_to_cell_map<F: Field, const N: usize>(
    partition: &MemoryEquipartition<F, N>,
) -> HashMap<(F, F), F> {
    let mut map = HashMap::new();
    for ((address_space, label), ts_values) in partition.iter() {
        for (i, value) in ts_values.values.iter().enumerate() {
            map.insert(
                (*address_space, F::from_canonical_usize(label * N + i)),
                *value,
            );
        }
    }
    map
}

#[derive(Clone, Debug)]
pub struct MemoryChip<F: Field> {
    pub memory_bus: MemoryBus,
    pub interface_chip: MemoryInterface<F>,
    pub(crate) mem_config: MemoryConfig,
    pub(crate) range_checker: Arc<VariableRangeCheckerChip>,

    initial_memory: Option<MemoryEquipartition<F, CHUNK>>,

    // addr_space -> Memory data structure
    memory: Memory<F>,
    /// Maps a length to a list of access adapters with that block length as th larger size.
    adapter_records: HashMap<usize, Vec<AccessAdapterRecord<F>>>,

    // Filled during finalization.
    result: Option<MemoryChipResult<F>>,
}

impl<F: PrimeField32> MemoryChip<F> {
    pub fn with_volatile_memory(
        memory_bus: MemoryBus,
        mem_config: MemoryConfig,
        range_checker: Arc<VariableRangeCheckerChip>,
    ) -> Self {
        Self {
            memory_bus,
            mem_config: mem_config.clone(),
            interface_chip: MemoryInterface::Volatile {
                boundary_chip: VolatileBoundaryChip::new(
                    memory_bus,
                    mem_config.addr_space_max_bits,
                    mem_config.pointer_max_bits,
                    mem_config.decomp,
                    range_checker.clone(),
                ),
            },
            memory: Memory::new(
                &MemoryEquipartition::<_, 1>::new(),
                mem_config.pointer_max_bits,
            ),
            adapter_records: HashMap::new(),
            range_checker,
            initial_memory: None,
            result: None,
        }
    }

    pub fn with_persistent_memory(
        memory_bus: MemoryBus,
        mem_config: MemoryConfig,
        range_checker: Arc<VariableRangeCheckerChip>,
        merkle_bus: MemoryMerkleBus,
        initial_memory: MemoryEquipartition<F, CHUNK>,
    ) -> Self {
        let memory_dims = MemoryDimensions {
            as_height: mem_config.addr_space_max_bits,
            address_height: mem_config.pointer_max_bits - log2_strict_usize(CHUNK),
            as_offset: 1,
        };
        let interface_chip = MemoryInterface::Persistent {
            boundary_chip: PersistentBoundaryChip::new(memory_dims, memory_bus, merkle_bus),
            merkle_chip: MemoryMerkleChip::new(memory_dims, merkle_bus),
        };
        Self {
            memory_bus,
            mem_config: mem_config.clone(),
            interface_chip,
            memory: Memory::new(&initial_memory, mem_config.pointer_max_bits),
            adapter_records: HashMap::new(),
            range_checker,
            initial_memory: Some(initial_memory),
            result: None,
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

        let (record, adapter_records) = self
            .memory
            .read::<N>(address_space, pointer.as_canonical_u32() as usize);
        for record in adapter_records {
            self.adapter_records
                .entry(record.data.len())
                .or_default()
                .push(record);
        }

        for i in 0..N as u32 {
            let ptr = F::from_canonical_u32(pointer.as_canonical_u32() + i);
            self.interface_chip.touch_address(address_space, ptr);
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

    /// Reads a word directly from memory without updating internal state.
    ///
    /// Any value returned is unconstrained.
    pub fn unsafe_read_cell(&self, addr_space: F, pointer: F) -> F {
        let (_, &value) = self
            .memory
            .get(addr_space, pointer.as_canonical_u32() as usize)
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

        let (record, adapter_records) =
            self.memory
                .write(address_space, pointer.as_canonical_u32() as usize, data);
        for record in adapter_records {
            self.adapter_records
                .entry(record.data.len())
                .or_default()
                .push(record);
        }

        for i in 0..N as u32 {
            let ptr = F::from_canonical_u32(pointer.as_canonical_u32() + i);
            self.interface_chip.touch_address(address_space, ptr);
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

    pub fn access_adapter_air<const N: usize>(&self) -> AccessAdapterAir<N> {
        let lt_air = IsLessThanAir::new(self.range_checker.bus(), self.mem_config.clk_max_bits);
        AccessAdapterAir::<N> {
            memory_bus: self.memory_bus,
            lt_air,
        }
    }

    // TODO[zach]: Make finalize return a list of `MachineChip`.
    pub fn finalize(&mut self, hasher: Option<&mut impl HasherChip<CHUNK, F>>) {
        let mut traces = vec![];
        let mut pvs = vec![];

        let records = match &mut self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                let (final_memory, records) = self.memory.finalize::<1>();
                traces.push(boundary_chip.generate_trace(&final_memory));
                pvs.push(vec![]);

                records
            }
            MemoryInterface::Persistent {
                merkle_chip,
                boundary_chip,
            } => {
                let (final_partition, records) = self.memory.finalize::<8>();
                traces.push(boundary_chip.generate_trace(&final_partition));
                pvs.push(vec![]);

                let hasher = hasher.unwrap();

                let initial_memory = partition_to_cell_map(self.initial_memory.as_ref().unwrap());
                let initial_node = MemoryNode::tree_from_memory(
                    merkle_chip.air.memory_dimensions,
                    &initial_memory,
                    hasher,
                );
                let final_memory = partition_to_cell_map(&final_partition);
                let (expand_trace, final_node) =
                    merkle_chip.generate_trace_and_final_tree(&initial_node, &final_memory, hasher);
                traces.push(expand_trace);
                let mut expand_pvs = vec![];
                expand_pvs.extend(initial_node.hash());
                expand_pvs.extend(final_node.hash());
                pvs.push(expand_pvs);
                records
            }
        };
        for record in records {
            self.adapter_records
                .entry(record.data.len())
                .or_default()
                .push(record);
        }

        traces.extend([
            self.generate_access_adapter_trace::<2>(),
            self.generate_access_adapter_trace::<4>(),
            self.generate_access_adapter_trace::<8>(),
            self.generate_access_adapter_trace::<16>(),
            self.generate_access_adapter_trace::<32>(),
            self.generate_access_adapter_trace::<64>(),
        ]);
        pvs.extend(vec![vec![]; 6]);

        self.result = Some(MemoryChipResult {
            traces,
            public_values: pvs,
        });
    }

    // TEMPORARY[jpw]: MemoryChip is not a VmChip: it is not a chip and instead owns multiple AIRs. To be renamed and refactored in the future.
    pub fn generate_traces(self) -> Vec<RowMajorMatrix<F>> {
        self.result.as_ref().unwrap().traces.clone()
    }

    pub fn airs<SC: StarkGenericConfig>(&self) -> Vec<Arc<dyn AnyRap<SC>>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let mut airs = Vec::<Arc<dyn AnyRap<SC>>>::new();

        match &self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                airs.push(Arc::new(boundary_chip.air.clone()))
            }
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
            } => {
                airs.push(Arc::new(boundary_chip.air.clone()));
                airs.push(Arc::new(merkle_chip.air.clone()));
            }
        }

        airs.push(Arc::new(self.access_adapter_air::<2>()));
        airs.push(Arc::new(self.access_adapter_air::<4>()));
        airs.push(Arc::new(self.access_adapter_air::<8>()));
        airs.push(Arc::new(self.access_adapter_air::<16>()));
        airs.push(Arc::new(self.access_adapter_air::<32>()));
        airs.push(Arc::new(self.access_adapter_air::<64>()));
        airs
    }

    pub fn air_names(&self) -> Vec<String> {
        let mut air_names = vec!["MemoryBoundary".to_string()];
        match &self.interface_chip {
            MemoryInterface::Volatile { .. } => {}
            MemoryInterface::Persistent { .. } => air_names.push("Merkle".to_string()),
        }
        air_names.extend([
            "AccessAdapter<2>".to_string(),
            "AccessAdapter<4>".to_string(),
            "AccessAdapter<8>".to_string(),
            "AccessAdapter<16>".to_string(),
            "AccessAdapter<32>".to_string(),
            "AccessAdapter<64>".to_string(),
        ]);
        air_names
    }

    pub fn current_trace_heights(&self) -> Vec<usize> {
        let mut heights = vec![];
        match &self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                heights.push(boundary_chip.current_height());
            }
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
            } => {
                heights.push(boundary_chip.current_height());
                heights.push(merkle_chip.current_height());
            }
        };
        heights.extend([
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
        ]);
        heights
    }

    fn trace_widths(&self) -> Vec<usize> {
        let mut widths = vec![];
        match &self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                widths.push(BaseAir::<F>::width(&boundary_chip.air));
            }
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
            } => {
                widths.push(BaseAir::<F>::width(&boundary_chip.air));
                widths.push(BaseAir::<F>::width(&merkle_chip.air));
            }
        };
        widths.extend([
            BaseAir::<F>::width(&self.access_adapter_air::<2>()),
            BaseAir::<F>::width(&self.access_adapter_air::<4>()),
            BaseAir::<F>::width(&self.access_adapter_air::<8>()),
            BaseAir::<F>::width(&self.access_adapter_air::<16>()),
            BaseAir::<F>::width(&self.access_adapter_air::<32>()),
            BaseAir::<F>::width(&self.access_adapter_air::<64>()),
        ]);
        widths
    }

    pub fn current_trace_cells(&self) -> Vec<usize> {
        zip_eq(self.current_trace_heights(), self.trace_widths())
            .map(|(h, w)| h * w)
            .collect()
    }

    pub fn generate_public_values_per_air(&self) -> Vec<Vec<F>> {
        self.result.as_ref().unwrap().public_values.clone()
    }
}

#[derive(Clone, Debug)]
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
        kernels::core::RANGE_CHECKER_BUS,
        system::{memory::offline_checker::MemoryBus, vm::config::MemoryConfig},
    };

    #[test]
    fn test_no_adapter_records_for_singleton_accesses() {
        type F = BabyBear;

        let memory_bus = MemoryBus(1);
        let memory_config = MemoryConfig::default();
        let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, memory_config.decomp);
        let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let mut memory_chip = MemoryChip::with_volatile_memory(
            memory_bus,
            memory_config.clone(),
            range_checker.clone(),
        );

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
