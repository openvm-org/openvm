use std::{
    array,
    collections::BTreeMap,
    iter,
    marker::PhantomData,
    mem,
    sync::{Arc, Mutex},
};

use getset::Getters;
use openvm_circuit_primitives::{
    assert_less_than::{AssertLtSubAir, LessThanAuxCols},
    is_zero::IsZeroSubAir,
    utils::next_power_of_two_or_zero,
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    TraceSubRowGenerator,
};
use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    p3_field::PrimeField32,
    p3_maybe_rayon::prelude::{IntoParallelIterator, ParallelIterator},
    p3_util::log2_strict_usize,
    prover::types::AirProofInput,
    rap::AnyRap,
    Chip, ChipUsageGetter,
};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use self::interface::MemoryInterface;
use super::{merkle::DirectCompressionBus, volatile::VolatileBoundaryChip};
use crate::{
    arch::{hasher::HasherChip, MemoryConfig},
    system::memory::{
        adapter::AccessAdapterInventory,
        dimensions::MemoryDimensions,
        merkle::{MemoryMerkleBus, MemoryMerkleChip},
        offline::{MemoryRecord, OfflineMemory, INITIAL_TIMESTAMP},
        offline_checker::{
            MemoryBridge, MemoryBus, MemoryReadAuxCols, MemoryReadOrImmediateAuxCols,
            MemoryWriteAuxCols, AUX_LEN,
        },
        online::{Address, Memory, MemoryLogEntry},
        persistent::PersistentBoundaryChip,
        tree::MemoryNode,
    },
};

pub mod dimensions;
mod interface;

pub const CHUNK: usize = 8;
/// The offset of the Merkle AIR in AIRs of MemoryController.
pub const MERKLE_AIR_OFFSET: usize = 1;
/// The offset of the boundary AIR in AIRs of MemoryController.
pub const BOUNDARY_AIR_OFFSET: usize = 0;

#[derive(Debug, Clone, Copy)]
pub struct RecordId(pub usize);

pub type MemoryImage<F> = FxHashMap<Address, F>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimestampedValues<T, const N: usize> {
    pub timestamp: u32,
    pub values: [T; N],
}

/// An equipartition of memory, with timestamps and values.
///
/// The key is a pair `(address_space, label)`, where `label` is the index of the block in the
/// partition. I.e., the starting address of the block is `(address_space, label * N)`.
///
/// If a key is not present in the map, then the block is uninitialized (and therefore zero).
pub type TimestampedEquipartition<F, const N: usize> =
    BTreeMap<(u32, u32), TimestampedValues<F, N>>;

/// An equipartition of memory values.
///
/// The key is a pair `(address_space, label)`, where `label` is the index of the block in the
/// partition. I.e., the starting address of the block is `(address_space, label * N)`.
///
/// If a key is not present in the map, then the block is uninitialized (and therefore zero).
pub type Equipartition<F, const N: usize> = BTreeMap<(u32, u32), [F; N]>;

#[derive(Debug, Getters)]
pub struct MemoryController<F> {
    pub memory_bus: MemoryBus,
    pub interface_chip: MemoryInterface<F>,

    #[getset(get = "pub")]
    pub(crate) mem_config: MemoryConfig,
    pub range_checker: Arc<VariableRangeCheckerChip>,
    // Store separately to avoid smart pointer reference each time
    range_checker_bus: VariableRangeCheckerBus,

    // addr_space -> Memory data structure
    memory: Memory<F>,

    /// A reference to the `OfflineMemory`. Will be populated after `finalize()`.
    ///
    /// TODO: Remove this as state on `MemoryController` and compute and return this during `finalize()`.
    offline_memory: Arc<Mutex<OfflineMemory<F>>>,

    access_adapters: AccessAdapterInventory<F>,

    // Filled during finalization.
    final_state: Option<FinalState<F>>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum FinalState<F> {
    Volatile(VolatileFinalState<F>),
    #[allow(dead_code)]
    Persistent(PersistentFinalState<F>),
}
#[derive(Debug, Default)]
struct VolatileFinalState<F> {
    _marker: PhantomData<F>,
}
#[allow(dead_code)]
#[derive(Debug)]
struct PersistentFinalState<F> {
    final_memory: Equipartition<F, CHUNK>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryTraceHeights {
    Volatile(VolatileMemoryTraceHeights),
    Persistent(PersistentMemoryTraceHeights),
}

impl MemoryTraceHeights {
    fn flatten(&self) -> Vec<usize> {
        match self {
            MemoryTraceHeights::Volatile(oh) => oh.flatten(),
            MemoryTraceHeights::Persistent(oh) => oh.flatten(),
        }
    }

    /// Round all trace heights to the next power of two. This will round trace heights of 0 to 1.
    pub fn round_to_next_power_of_two(&mut self) {
        match self {
            MemoryTraceHeights::Volatile(oh) => oh.round_to_next_power_of_two(),
            MemoryTraceHeights::Persistent(oh) => oh.round_to_next_power_of_two(),
        }
    }

    /// Round all trace heights to the next power of two, except 0 stays 0.
    pub fn round_to_next_power_of_two_or_zero(&mut self) {
        match self {
            MemoryTraceHeights::Volatile(oh) => oh.round_to_next_power_of_two_or_zero(),
            MemoryTraceHeights::Persistent(oh) => oh.round_to_next_power_of_two_or_zero(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VolatileMemoryTraceHeights {
    pub boundary: usize,
    pub access_adapters: Vec<usize>,
}

impl VolatileMemoryTraceHeights {
    pub fn flatten(&self) -> Vec<usize> {
        iter::once(self.boundary)
            .chain(self.access_adapters.iter().copied())
            .collect()
    }

    fn round_to_next_power_of_two(&mut self) {
        self.boundary = self.boundary.next_power_of_two();
        self.access_adapters
            .iter_mut()
            .for_each(|v| *v = v.next_power_of_two());
    }

    fn round_to_next_power_of_two_or_zero(&mut self) {
        self.boundary = next_power_of_two_or_zero(self.boundary);
        self.access_adapters
            .iter_mut()
            .for_each(|v| *v = next_power_of_two_or_zero(*v));
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersistentMemoryTraceHeights {
    boundary: usize,
    merkle: usize,
    access_adapters: Vec<usize>,
}
impl PersistentMemoryTraceHeights {
    pub fn flatten(&self) -> Vec<usize> {
        vec![self.boundary, self.merkle]
            .into_iter()
            .chain(self.access_adapters.iter().copied())
            .collect()
    }

    fn round_to_next_power_of_two(&mut self) {
        self.boundary = self.boundary.next_power_of_two();
        self.merkle = self.merkle.next_power_of_two();
        self.access_adapters
            .iter_mut()
            .for_each(|v| *v = v.next_power_of_two());
    }

    fn round_to_next_power_of_two_or_zero(&mut self) {
        self.boundary = next_power_of_two_or_zero(self.boundary);
        self.merkle = next_power_of_two_or_zero(self.merkle);
        self.access_adapters
            .iter_mut()
            .for_each(|v| *v = next_power_of_two_or_zero(*v));
    }
}

impl<F: PrimeField32> MemoryController<F> {
    pub fn continuation_enabled(&self) -> bool {
        match &self.interface_chip {
            MemoryInterface::Volatile { .. } => false,
            MemoryInterface::Persistent { .. } => true,
        }
    }
    pub fn with_volatile_memory(
        memory_bus: MemoryBus,
        mem_config: MemoryConfig,
        range_checker: Arc<VariableRangeCheckerChip>,
    ) -> Self {
        let range_checker_bus = range_checker.bus();
        let initial_memory = MemoryImage::default();
        Self {
            memory_bus,
            mem_config,
            interface_chip: MemoryInterface::Volatile {
                boundary_chip: VolatileBoundaryChip::new(
                    memory_bus,
                    mem_config.as_height,
                    mem_config.pointer_max_bits,
                    range_checker.clone(),
                ),
            },
            memory: Memory::new(),
            offline_memory: Arc::new(Mutex::new(OfflineMemory::new(
                initial_memory,
                1,
                memory_bus,
                range_checker.clone(),
                mem_config.clk_max_bits,
            ))),
            access_adapters: AccessAdapterInventory::new(
                range_checker.clone(),
                memory_bus,
                mem_config.clk_max_bits,
                mem_config.max_access_adapter_n,
            ),
            range_checker,
            range_checker_bus,
            final_state: None,
        }
    }

    pub fn with_persistent_memory(
        memory_bus: MemoryBus,
        mem_config: MemoryConfig,
        range_checker: Arc<VariableRangeCheckerChip>,
        merkle_bus: MemoryMerkleBus,
        compression_bus: DirectCompressionBus,
        initial_memory: MemoryImage<F>,
    ) -> Self {
        let memory_dims = MemoryDimensions {
            as_height: mem_config.as_height,
            address_height: mem_config.pointer_max_bits - log2_strict_usize(CHUNK),
            as_offset: 1,
        };
        let memory = Memory::from_image(initial_memory.clone());
        let range_checker_bus = range_checker.bus();
        let interface_chip = MemoryInterface::Persistent {
            boundary_chip: PersistentBoundaryChip::new(
                memory_dims,
                memory_bus,
                merkle_bus,
                compression_bus,
            ),
            merkle_chip: MemoryMerkleChip::new(memory_dims, merkle_bus, compression_bus),
            initial_memory: initial_memory.clone(),
        };
        Self {
            memory_bus,
            mem_config,
            interface_chip,
            memory,
            offline_memory: Arc::new(Mutex::new(OfflineMemory::new(
                initial_memory,
                CHUNK,
                memory_bus,
                range_checker.clone(),
                mem_config.clk_max_bits,
            ))),
            access_adapters: AccessAdapterInventory::new(
                range_checker.clone(),
                memory_bus,
                mem_config.clk_max_bits,
                mem_config.max_access_adapter_n,
            ),
            range_checker,
            range_checker_bus,
            final_state: None,
        }
    }

    pub fn memory_image(&self) -> &MemoryImage<F> {
        &self.memory.data
    }

    pub fn set_override_trace_heights(&mut self, overridden_heights: MemoryTraceHeights) {
        match &mut self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => match overridden_heights {
                MemoryTraceHeights::Volatile(oh) => {
                    boundary_chip.set_overridden_height(oh.boundary);
                    self.access_adapters
                        .set_override_trace_heights(oh.access_adapters);
                }
                _ => panic!("Expect overridden_heights to be MemoryTraceHeights::Volatile"),
            },
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
                ..
            } => match overridden_heights {
                MemoryTraceHeights::Persistent(oh) => {
                    boundary_chip.set_overridden_height(oh.boundary);
                    merkle_chip.set_overridden_height(oh.merkle);
                    self.access_adapters
                        .set_override_trace_heights(oh.access_adapters);
                }
                _ => panic!("Expect overridden_heights to be MemoryTraceHeights::Persistent"),
            },
        }
    }

    pub fn set_initial_memory(&mut self, memory: MemoryImage<F>) {
        if self.timestamp() > INITIAL_TIMESTAMP + 1 {
            panic!("Cannot set initial memory after first timestamp");
        }
        let mut offline_memory = self.offline_memory.lock().unwrap();
        offline_memory.set_initial_memory(memory.clone());

        match &mut self.interface_chip {
            MemoryInterface::Volatile { .. } => {
                if !memory.is_empty() {
                    panic!("Cannot set initial memory for volatile memory");
                }
            }
            MemoryInterface::Persistent { initial_memory, .. } => {
                *initial_memory = memory;
                self.memory = Memory::from_image(initial_memory.clone());
            }
        }
    }

    pub fn memory_bridge(&self) -> MemoryBridge {
        MemoryBridge::new(
            self.memory_bus,
            self.mem_config.clk_max_bits,
            self.range_checker_bus,
        )
    }

    pub fn read_cell(&mut self, address_space: F, pointer: F) -> (RecordId, F) {
        let (record_id, [data]) = self.read(address_space, pointer);
        (record_id, data)
    }

    pub fn read<const N: usize>(&mut self, address_space: F, pointer: F) -> (RecordId, [F; N]) {
        let address_space_u32 = address_space.as_canonical_u32();
        let ptr_u32 = pointer.as_canonical_u32();
        assert!(
            address_space == F::ZERO || ptr_u32 < (1 << self.mem_config.pointer_max_bits),
            "memory out of bounds: {ptr_u32:?}",
        );

        let (record_id, values) = self.memory.read::<N>(address_space_u32, ptr_u32);

        (record_id, values)
    }

    /// Reads a word directly from memory without updating internal state.
    ///
    /// Any value returned is unconstrained.
    pub fn unsafe_read_cell(&self, addr_space: F, ptr: F) -> F {
        self.unsafe_read::<1>(addr_space, ptr)[0]
    }

    /// Reads a word directly from memory without updating internal state.
    ///
    /// Any value returned is unconstrained.
    pub fn unsafe_read<const N: usize>(&self, addr_space: F, ptr: F) -> [F; N] {
        let addr_space = addr_space.as_canonical_u32();
        let ptr = ptr.as_canonical_u32();
        array::from_fn(|i| self.memory.get(addr_space, ptr + i as u32))
    }

    /// Writes `data` to the given cell.
    ///
    /// Returns the `RecordId` and previous data.
    pub fn write_cell(&mut self, address_space: F, pointer: F, data: F) -> (RecordId, F) {
        let (record_id, [data]) = self.write(address_space, pointer, [data]);
        (record_id, data)
    }

    pub fn write<const N: usize>(
        &mut self,
        address_space: F,
        pointer: F,
        data: [F; N],
    ) -> (RecordId, [F; N]) {
        assert_ne!(address_space, F::ZERO);
        let address_space_u32 = address_space.as_canonical_u32();
        let ptr_u32 = pointer.as_canonical_u32();
        assert!(
            ptr_u32 < (1 << self.mem_config.pointer_max_bits),
            "memory out of bounds: {ptr_u32:?}",
        );

        self.memory.write(address_space_u32, ptr_u32, data)
    }

    pub fn aux_cols_factory(&self) -> MemoryAuxColsFactory<F> {
        let range_bus = self.range_checker.bus();
        MemoryAuxColsFactory {
            range_checker: self.range_checker.clone(),
            timestamp_lt_air: AssertLtSubAir::new(range_bus, self.mem_config.clk_max_bits),
            _marker: Default::default(),
        }
    }

    pub fn increment_timestamp(&mut self) {
        self.memory.increment_timestamp_by(1);
    }

    pub fn increment_timestamp_by(&mut self, change: u32) {
        self.memory.increment_timestamp_by(change);
    }

    pub fn timestamp(&self) -> u32 {
        self.memory.timestamp()
    }

    fn replay_access_log(&mut self) {
        let mut offline_memory = self.offline_memory.lock().unwrap();
        let log = mem::take(&mut self.memory.log);

        for entry in log {
            Self::replay_access(
                entry,
                &mut offline_memory,
                &mut self.interface_chip,
                &mut self.access_adapters,
            );
        }
    }

    fn replay_access(
        entry: MemoryLogEntry<F>,
        offline_memory: &mut OfflineMemory<F>,
        interface_chip: &mut MemoryInterface<F>,
        access_adapters: &mut AccessAdapterInventory<F>,
    ) {
        let records = match entry {
            MemoryLogEntry::Read {
                address_space,
                pointer,
                len,
            } => {
                if address_space != 0 {
                    interface_chip.touch_range(address_space, pointer, len as u32);
                }
                offline_memory.read(address_space, pointer, len)
            }
            MemoryLogEntry::Write {
                address_space,
                pointer,
                data,
            } => {
                if address_space != 0 {
                    interface_chip.touch_range(address_space, pointer, data.len() as u32);
                }
                offline_memory.write(address_space, pointer, data)
            }
            MemoryLogEntry::IncrementTimestampBy(amount) => {
                offline_memory.increment_timestamp_by(amount);
                Vec::new()
            }
        };
        access_adapters.extend_records(records);
    }

    /// Returns the final memory state if persistent.
    pub fn finalize(&mut self, hasher: Option<&mut impl HasherChip<CHUNK, F>>) {
        if self.final_state.is_some() {
            return;
        }

        self.replay_access_log();
        let mut offline_memory = self.offline_memory.lock().unwrap();

        match &mut self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                let (final_memory, records) = offline_memory.finalize::<1>();
                self.access_adapters.extend_records(records);

                boundary_chip.finalize(final_memory);
                self.final_state = Some(FinalState::Volatile(VolatileFinalState::default()));
            }
            MemoryInterface::Persistent {
                merkle_chip,
                boundary_chip,
                initial_memory,
            } => {
                let hasher = hasher.unwrap();

                let (final_partition, records) = offline_memory.finalize::<CHUNK>();
                self.access_adapters.extend_records(records);

                boundary_chip.finalize(initial_memory, &final_partition, hasher);
                let final_memory_values = final_partition
                    .into_par_iter()
                    .map(|(key, value)| (key, value.values))
                    .collect();
                let initial_node = MemoryNode::tree_from_memory(
                    merkle_chip.air.memory_dimensions,
                    initial_memory,
                    hasher,
                );
                merkle_chip.finalize(&initial_node, &final_memory_values, hasher);
                self.final_state = Some(FinalState::Persistent(PersistentFinalState {
                    final_memory: final_memory_values.clone(),
                }));
            }
        };
    }

    pub fn generate_air_proof_inputs<SC: StarkGenericConfig>(self) -> Vec<AirProofInput<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let mut ret = Vec::new();

        let Self {
            interface_chip,
            access_adapters,
            ..
        } = self;
        match interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                ret.push(boundary_chip.generate_air_proof_input());
            }
            MemoryInterface::Persistent {
                merkle_chip,
                boundary_chip,
                ..
            } => {
                debug_assert_eq!(ret.len(), BOUNDARY_AIR_OFFSET);
                ret.push(boundary_chip.generate_air_proof_input());
                debug_assert_eq!(ret.len(), MERKLE_AIR_OFFSET);
                ret.push(merkle_chip.generate_air_proof_input());
            }
        }
        ret.extend(access_adapters.generate_air_proof_inputs());
        ret
    }

    pub fn airs<SC: StarkGenericConfig>(&self) -> Vec<Arc<dyn AnyRap<SC>>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let mut airs = Vec::<Arc<dyn AnyRap<SC>>>::new();

        match &self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                debug_assert_eq!(airs.len(), BOUNDARY_AIR_OFFSET);
                airs.push(boundary_chip.air())
            }
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
                ..
            } => {
                debug_assert_eq!(airs.len(), BOUNDARY_AIR_OFFSET);
                airs.push(boundary_chip.air());
                debug_assert_eq!(airs.len(), MERKLE_AIR_OFFSET);
                airs.push(merkle_chip.air());
            }
        }
        airs.extend(self.access_adapters.airs());

        airs
    }

    /// Return the number of AIRs in the memory controller.
    pub fn num_airs(&self) -> usize {
        let mut num_airs = 1;
        if self.continuation_enabled() {
            num_airs += 1;
        }
        num_airs += self.access_adapters.num_access_adapters();
        num_airs
    }

    pub fn air_names(&self) -> Vec<String> {
        let mut air_names = vec!["Boundary".to_string()];
        if self.continuation_enabled() {
            air_names.push("Merkle".to_string());
        }
        air_names.extend(self.access_adapters.air_names());
        air_names
    }

    pub fn current_trace_heights(&self) -> Vec<usize> {
        self.get_memory_trace_heights().flatten()
    }

    pub fn get_memory_trace_heights(&self) -> MemoryTraceHeights {
        let access_adapters = self.access_adapters.get_heights();
        match &self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                MemoryTraceHeights::Volatile(VolatileMemoryTraceHeights {
                    boundary: boundary_chip.current_trace_height(),
                    access_adapters,
                })
            }
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
                ..
            } => MemoryTraceHeights::Persistent(PersistentMemoryTraceHeights {
                boundary: boundary_chip.current_trace_height(),
                merkle: merkle_chip.current_trace_height(),
                access_adapters,
            }),
        }
    }
    pub fn get_dummy_memory_trace_heights(&self) -> MemoryTraceHeights {
        let access_adapters = vec![1; self.access_adapters.num_access_adapters()];
        match &self.interface_chip {
            MemoryInterface::Volatile { .. } => {
                MemoryTraceHeights::Volatile(VolatileMemoryTraceHeights {
                    boundary: 1,
                    access_adapters,
                })
            }
            MemoryInterface::Persistent { .. } => {
                MemoryTraceHeights::Persistent(PersistentMemoryTraceHeights {
                    boundary: 1,
                    merkle: 1,
                    access_adapters,
                })
            }
        }
    }

    pub fn current_trace_cells(&self) -> Vec<usize> {
        let mut ret = Vec::new();
        match &self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                ret.push(boundary_chip.current_trace_cells())
            }
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
                ..
            } => {
                ret.push(boundary_chip.current_trace_cells());
                ret.push(merkle_chip.current_trace_cells());
            }
        }
        ret.extend(self.access_adapters.get_cells());
        ret
    }

    /// Returns a reference to the offline memory.
    ///
    /// Until `finalize` is called, the `OfflineMemory` does not contain useful state, and should
    /// therefore not be used by any chip during execution. However, to obtain a reference to the
    /// offline memory that will be useful in trace generation, a chip can call `offline_memory()`
    /// and store the returned reference for later use.
    pub fn offline_memory(&self) -> Arc<Mutex<OfflineMemory<F>>> {
        self.offline_memory.clone()
    }
}

#[derive(Clone, Debug)]
pub struct MemoryAuxColsFactory<T> {
    pub(crate) range_checker: Arc<VariableRangeCheckerChip>,
    pub(crate) timestamp_lt_air: AssertLtSubAir,
    pub(crate) _marker: PhantomData<T>,
}

// NOTE[jpw]: The `make_*_aux_cols` functions should be thread-safe so they can be used in parallelized trace generation.
impl<F: PrimeField32> MemoryAuxColsFactory<F> {
    pub fn make_read_aux_cols<const N: usize>(
        &self,
        read: &MemoryRecord<F>,
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

    pub fn make_read_or_immediate_aux_cols(
        &self,
        read: &MemoryRecord<F>,
    ) -> MemoryReadOrImmediateAuxCols<F> {
        let mut inv = F::ZERO;
        let mut is_zero = F::ZERO;
        IsZeroSubAir.generate_subrow(read.address_space, (&mut inv, &mut is_zero));
        let timestamp_lt_cols =
            self.generate_timestamp_lt_cols(read.prev_timestamp, read.timestamp);

        MemoryReadOrImmediateAuxCols::new(
            F::from_canonical_u32(read.prev_timestamp),
            is_zero,
            inv,
            timestamp_lt_cols,
        )
    }

    pub fn make_write_aux_cols<const N: usize>(
        &self,
        write: &MemoryRecord<F>,
    ) -> MemoryWriteAuxCols<F, N> {
        let prev_data = write.prev_data.clone().unwrap();
        MemoryWriteAuxCols::new(
            prev_data.try_into().unwrap(),
            F::from_canonical_u32(write.prev_timestamp),
            self.generate_timestamp_lt_cols(write.prev_timestamp, write.timestamp),
        )
    }

    fn generate_timestamp_lt_cols(
        &self,
        prev_timestamp: u32,
        timestamp: u32,
    ) -> LessThanAuxCols<F, AUX_LEN> {
        debug_assert!(prev_timestamp < timestamp);
        let mut decomp = [F::ZERO; AUX_LEN];
        self.timestamp_lt_air.generate_subrow(
            (&self.range_checker, prev_timestamp, timestamp),
            &mut decomp,
        );
        LessThanAuxCols::new(decomp)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openvm_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use rand::{prelude::SliceRandom, thread_rng, Rng};

    use super::MemoryController;
    use crate::{
        arch::{MemoryConfig, MEMORY_BUS},
        system::memory::offline_checker::MemoryBus,
    };

    const RANGE_CHECKER_BUS: usize = 3;

    #[test]
    fn test_no_adapter_records_for_singleton_accesses() {
        type F = BabyBear;

        let memory_bus = MemoryBus(MEMORY_BUS);
        let memory_config = MemoryConfig::default();
        let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, memory_config.decomp);
        let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let mut memory_controller = MemoryController::with_volatile_memory(
            memory_bus,
            memory_config,
            range_checker.clone(),
        );

        let mut rng = thread_rng();
        for _ in 0..1000 {
            let address_space = F::from_canonical_u32(*[1, 2].choose(&mut rng).unwrap());
            let pointer =
                F::from_canonical_u32(rng.gen_range(0..1 << memory_config.pointer_max_bits));

            if rng.gen_bool(0.5) {
                let data = F::from_canonical_u32(rng.gen_range(0..1 << 30));
                memory_controller.write(address_space, pointer, [data]);
            } else {
                memory_controller.read::<1>(address_space, pointer);
            }
        }
        assert!(memory_controller
            .access_adapters
            .get_heights()
            .iter()
            .all(|&h| h == 0));
    }
}