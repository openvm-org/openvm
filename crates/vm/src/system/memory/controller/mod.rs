use std::{array::from_fn, collections::BTreeMap, fmt::Debug, iter, marker::PhantomData};

use getset::{Getters, MutGetters};
use openvm_circuit_primitives::{
    assert_less_than::{AssertLtSubAir, LessThanAuxCols},
    utils::next_power_of_two_or_zero,
    var_range::{
        SharedVariableRangeCheckerChip, VariableRangeCheckerBus, VariableRangeCheckerChip,
    },
    TraceSubRowGenerator,
};
use openvm_instructions::NATIVE_AS;
use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig},
    interaction::PermutationCheckBus,
    p3_commit::PolynomialSpace,
    p3_field::PrimeField32,
    p3_maybe_rayon::prelude::{IntoParallelIterator, ParallelIterator},
    p3_util::{log2_ceil_usize, log2_strict_usize},
    prover::types::AirProofInput,
    AirRef, Chip, ChipUsageGetter,
};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use self::interface::MemoryInterface;
use super::{online::INITIAL_TIMESTAMP, volatile::VolatileBoundaryChip, AddressMap, MemoryAddress};
use crate::{
    arch::{hasher::HasherChip, MemoryConfig, ADDR_SPACE_OFFSET},
    system::memory::{
        adapter::records::AccessRecordHeader,
        dimensions::MemoryDimensions,
        merkle::{MemoryMerkleChip, SerialReceiver},
        offline_checker::{MemoryBaseAuxCols, MemoryBridge, MemoryBus, AUX_LEN},
        online::{AccessMetadata, TracingMemory},
        persistent::PersistentBoundaryChip,
    },
};

pub mod dimensions;
pub mod interface;

pub const CHUNK: usize = 8;

/// The offset of the Merkle AIR in AIRs of MemoryController.
pub const MERKLE_AIR_OFFSET: usize = 1;
/// The offset of the boundary AIR in AIRs of MemoryController.
pub const BOUNDARY_AIR_OFFSET: usize = 0;

pub type MemoryImage = AddressMap;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimestampedValues<T, const N: usize> {
    pub timestamp: u32,
    pub values: [T; N],
}

/// A sorted equipartition of memory, with timestamps and values.
///
/// The "key" is a pair `(address_space, label)`, where `label` is the index of the block in the
/// partition. I.e., the starting address of the block is `(address_space, label * N)`.
pub type TimestampedEquipartition<F, const N: usize> = Vec<((u32, u32), TimestampedValues<F, N>)>;

/// An equipartition of memory values.
///
/// The key is a pair `(address_space, label)`, where `label` is the index of the block in the
/// partition. I.e., the starting address of the block is `(address_space, label * N)`.
///
/// If a key is not present in the map, then the block is uninitialized (and therefore zero).
pub type Equipartition<F, const N: usize> = BTreeMap<(u32, u32), [F; N]>;

#[derive(Getters, MutGetters)]
pub struct MemoryController<F> {
    pub memory_bus: MemoryBus,
    pub interface_chip: MemoryInterface<F>,
    #[getset(get = "pub")]
    pub(crate) mem_config: MemoryConfig,
    pub range_checker: SharedVariableRangeCheckerChip,
    // Store separately to avoid smart pointer reference each time
    range_checker_bus: VariableRangeCheckerBus,
    // addr_space -> Memory data structure
    pub memory: TracingMemory<F>,
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
        range_checker: SharedVariableRangeCheckerChip,
    ) -> Self {
        let range_checker_bus = range_checker.bus();
        assert!(mem_config.pointer_max_bits <= F::bits() - 2);
        assert!(mem_config
            .addr_space_sizes
            .iter()
            .all(|&x| x <= (1 << mem_config.pointer_max_bits)));
        assert!(mem_config.addr_space_height < F::bits() - 2);
        let addr_space_max_bits = log2_ceil_usize(
            (ADDR_SPACE_OFFSET + 2u32.pow(mem_config.addr_space_height as u32)) as usize,
        );
        Self {
            memory_bus,
            mem_config: mem_config.clone(),
            interface_chip: MemoryInterface::Volatile {
                boundary_chip: VolatileBoundaryChip::new(
                    memory_bus,
                    addr_space_max_bits,
                    mem_config.pointer_max_bits,
                    range_checker.clone(),
                ),
            },
            memory: TracingMemory::new(&mem_config, range_checker.clone(), memory_bus, 1),
            range_checker,
            range_checker_bus,
        }
    }

    /// Creates a new memory controller for persistent memory.
    ///
    /// Call `set_initial_memory` to set the initial memory state after construction.
    pub fn with_persistent_memory(
        memory_bus: MemoryBus,
        mem_config: MemoryConfig,
        range_checker: SharedVariableRangeCheckerChip,
        merkle_bus: PermutationCheckBus,
        compression_bus: PermutationCheckBus,
    ) -> Self {
        let memory_dims = MemoryDimensions {
            addr_space_height: mem_config.addr_space_height,
            address_height: mem_config.pointer_max_bits - log2_strict_usize(CHUNK),
        };
        let range_checker_bus = range_checker.bus();
        let interface_chip = MemoryInterface::Persistent {
            boundary_chip: PersistentBoundaryChip::new(
                memory_dims,
                memory_bus,
                merkle_bus,
                compression_bus,
            ),
            merkle_chip: MemoryMerkleChip::new(memory_dims, merkle_bus, compression_bus),
            initial_memory: AddressMap::from_mem_config(&mem_config),
        };
        Self {
            memory_bus,
            mem_config: mem_config.clone(),
            interface_chip,
            memory: TracingMemory::new(&mem_config, range_checker.clone(), memory_bus, CHUNK), /* it is expected that the memory will be
                                                                                                * set later */
            range_checker,
            range_checker_bus,
        }
    }

    pub fn memory_image(&self) -> &MemoryImage {
        &self.memory.data.memory
    }

    pub fn set_override_trace_heights(&mut self, overridden_heights: MemoryTraceHeights) {
        match &mut self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => match overridden_heights {
                MemoryTraceHeights::Volatile(oh) => {
                    boundary_chip.set_overridden_height(oh.boundary);
                    self.memory
                        .access_adapter_inventory
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
                    self.memory
                        .access_adapter_inventory
                        .set_override_trace_heights(oh.access_adapters);
                }
                _ => panic!("Expect overridden_heights to be MemoryTraceHeights::Persistent"),
            },
        }
    }

    // TODO[jpw]: change MemoryImage interface here
    pub fn set_initial_memory(&mut self, memory: MemoryImage) {
        if self.timestamp() > INITIAL_TIMESTAMP + 1 {
            panic!("Cannot set initial memory after first timestamp");
        }

        match &mut self.interface_chip {
            MemoryInterface::Volatile { .. } => {
                // Skip initialization for volatile memory
                return;
            }
            MemoryInterface::Persistent { initial_memory, .. } => {
                *initial_memory = memory.clone();
            }
        }

        self.memory = TracingMemory::new(
            &self.mem_config,
            self.range_checker.clone(),
            self.memory_bus,
            CHUNK,
        )
        .with_image(memory);
    }

    pub fn memory_bridge(&self) -> MemoryBridge {
        MemoryBridge::new(
            self.memory_bus,
            self.mem_config.clk_max_bits,
            self.range_checker_bus,
        )
    }

    pub fn helper(&self) -> SharedMemoryHelper<F> {
        let range_bus = self.range_checker.bus();
        SharedMemoryHelper {
            range_checker: self.range_checker.clone(),
            timestamp_lt_air: AssertLtSubAir::new(range_bus, self.mem_config.clk_max_bits),
            _marker: Default::default(),
        }
    }

    pub fn aux_cols_factory(&self) -> MemoryAuxColsFactory<F> {
        let range_bus = self.range_checker.bus();
        MemoryAuxColsFactory {
            range_checker: self.range_checker.as_ref(),
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

    /// Returns the equipartition of the touched blocks.
    /// Modifies records and adds new to account for the initial/final segments.
    fn touched_blocks_to_equipartition<const CHUNK: usize>(
        &mut self,
        touched_blocks: Vec<((u32, u32), AccessMetadata)>,
    ) -> TimestampedEquipartition<F, CHUNK> {
        // [perf] We can `.with_capacity()` if we keep track of the number of segments we initialize
        let mut final_memory = Vec::new();

        debug_assert!(touched_blocks.is_sorted_by_key(|(addr, _)| addr));
        let (bytes, fs): (Vec<_>, Vec<_>) = touched_blocks
            .into_iter()
            .partition(|((addr_sp, _), _)| *addr_sp < NATIVE_AS); // TODO: normal way

        self.handle_touched_blocks::<u8, CHUNK>(&mut final_memory, bytes, 4, |x| {
            F::from_canonical_u8(x)
        });
        self.handle_touched_blocks::<F, CHUNK>(&mut final_memory, fs, 1, |x| x);

        debug_assert!(final_memory.is_sorted_by_key(|(key, _)| *key));
        final_memory
    }

    fn handle_touched_blocks<T: Copy + Debug + Default, const CHUNK: usize>(
        &mut self,
        final_memory: &mut Vec<((u32, u32), TimestampedValues<F, CHUNK>)>,
        touched_blocks: Vec<((u32, u32), AccessMetadata)>,
        min_block_size: usize,
        convert: impl Fn(T) -> F,
    ) {
        let mut current_values = [T::default(); CHUNK];
        let mut current_cnt = 0;
        let mut current_address = MemoryAddress::new(0, 0);
        let mut current_timestamps = vec![0; CHUNK];
        for ((addr_space, ptr), metadata) in touched_blocks {
            let AccessMetadata {
                start_ptr,
                timestamp,
                block_size,
            } = metadata;
            assert!(
                current_cnt == 0
                    || (current_address.address_space == addr_space
                        && current_address.pointer + current_cnt as u32 == ptr),
                "The union of all touched blocks must consist of blocks with sizes divisible by `CHUNK`"
            );
            debug_assert!(block_size >= min_block_size as u32);
            debug_assert!(ptr % min_block_size as u32 == 0);

            if current_cnt == 0 {
                assert_eq!(
                    ptr & (CHUNK as u32 - 1),
                    0,
                    "The union of all touched blocks must consist of `CHUNK`-aligned blocks"
                );
                current_address = MemoryAddress::new(addr_space, ptr);
            }

            if block_size > min_block_size as u32 {
                self.memory.add_split_record(AccessRecordHeader {
                    timestamp_and_mask: timestamp,
                    address_space: addr_space,
                    pointer: start_ptr,
                    block_size,
                    lowest_block_size: min_block_size as u32,
                    type_size: size_of::<T>() as u32,
                });
            }
            if min_block_size > CHUNK {
                assert_eq!(current_cnt, 0);
                for i in (0..block_size).step_by(min_block_size) {
                    self.memory.add_split_record(AccessRecordHeader {
                        timestamp_and_mask: timestamp,
                        address_space: addr_space,
                        pointer: start_ptr + i,
                        block_size: min_block_size as u32,
                        lowest_block_size: CHUNK as u32,
                        type_size: size_of::<T>() as u32,
                    });
                }
                let values = unsafe {
                    self.memory
                        .data
                        .memory
                        .get_slice::<T>((addr_space, ptr), block_size as usize)
                };
                for i in (0..block_size).step_by(CHUNK) {
                    final_memory.push((
                        (addr_space, ptr + i),
                        TimestampedValues {
                            timestamp,
                            values: from_fn(|j| convert(values[i as usize + j])),
                        },
                    ));
                }
            } else {
                for i in 0..block_size {
                    current_values[current_cnt] =
                        unsafe { self.memory.data.memory.get((addr_space, ptr + i)) };
                    if current_cnt & (min_block_size - 1) == 0 {
                        current_timestamps[current_cnt / min_block_size] = timestamp;
                    }
                    current_cnt += 1;
                    if current_cnt == CHUNK {
                        let timestamp = *current_timestamps[..CHUNK / min_block_size]
                            .iter()
                            .max()
                            .unwrap();
                        self.memory.add_merge_record(
                            AccessRecordHeader {
                                timestamp_and_mask: timestamp,
                                address_space: addr_space,
                                pointer: current_address.pointer,
                                block_size: CHUNK as u32,
                                lowest_block_size: min_block_size as u32,
                                type_size: size_of::<T>() as u32,
                            },
                            &current_values,
                            &current_timestamps[..CHUNK / min_block_size],
                        );
                        final_memory.push((
                            (current_address.address_space, current_address.pointer),
                            TimestampedValues {
                                timestamp,
                                values: from_fn(|i| convert(current_values[i])),
                            },
                        ));
                        current_address.pointer += current_cnt as u32;
                        current_cnt = 0;
                    }
                }
            }
        }
        assert_eq!(current_cnt, 0, "The union of all touched blocks must consist of blocks with sizes divisible by `CHUNK`");
    }

    /// Finalize the boundary and merkle chips.
    #[instrument(name = "memory_finalize", skip_all)]
    pub fn finalize<H>(&mut self, hasher: Option<&mut H>)
    where
        H: HasherChip<CHUNK, F> + Sync + for<'a> SerialReceiver<&'a [F]>,
    {
        let touched_blocks = self.memory.touched_blocks();

        // Compute trace heights for access adapter chips and update their stored heights
        self.memory.access_adapter_inventory.compute_trace_heights();

        let mut final_memory_volatile = None;
        let mut final_memory_persistent = None;

        match &self.interface_chip {
            MemoryInterface::Volatile { .. } => {
                final_memory_volatile =
                    Some(self.touched_blocks_to_equipartition::<1>(touched_blocks));
            }
            MemoryInterface::Persistent { .. } => {
                final_memory_persistent =
                    Some(self.touched_blocks_to_equipartition::<CHUNK>(touched_blocks));
            }
        }

        match &mut self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                let final_memory = final_memory_volatile.unwrap();
                boundary_chip.finalize(final_memory);
            }
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
                initial_memory,
            } => {
                let final_memory = final_memory_persistent.unwrap();

                let hasher = hasher.unwrap();
                boundary_chip.finalize(initial_memory, &final_memory, hasher);
                let final_memory_values = final_memory
                    .into_par_iter()
                    .map(|(key, value)| (key, value.values))
                    .collect();
                merkle_chip.finalize(initial_memory, &final_memory_values, hasher);
            }
        }
    }

    pub fn generate_air_proof_inputs<SC: StarkGenericConfig>(self) -> Vec<AirProofInput<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let mut ret = Vec::new();

        let access_adapters = self.memory.access_adapter_inventory;
        match self.interface_chip {
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

    pub fn airs<SC: StarkGenericConfig>(&self) -> Vec<AirRef<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let mut airs = Vec::<AirRef<SC>>::new();

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
        airs.extend(self.memory.access_adapter_inventory.airs());

        airs
    }

    /// Return the number of AIRs in the memory controller.
    pub fn num_airs(&self) -> usize {
        let mut num_airs = 1;
        if self.continuation_enabled() {
            num_airs += 1;
        }
        num_airs += self.memory.access_adapter_inventory.num_access_adapters();
        num_airs
    }

    pub fn air_names(&self) -> Vec<String> {
        let mut air_names = vec!["Boundary".to_string()];
        if self.continuation_enabled() {
            air_names.push("Merkle".to_string());
        }
        air_names.extend(self.memory.access_adapter_inventory.air_names());
        air_names
    }

    pub fn current_trace_heights(&self) -> Vec<usize> {
        self.get_memory_trace_heights().flatten()
    }

    pub fn get_memory_trace_heights(&self) -> MemoryTraceHeights {
        let access_adapters = self.memory.access_adapter_inventory.get_heights();
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
        let access_adapters = vec![1; self.memory.access_adapter_inventory.num_access_adapters()];
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

    pub fn get_memory_trace_widths(&self) -> Vec<usize> {
        let access_adapter_widths = self.memory.access_adapter_inventory.get_widths();
        match &self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                vec![boundary_chip.trace_width()]
            }
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
                ..
            } => [boundary_chip.trace_width(), merkle_chip.trace_width()]
                .into_iter()
                .chain(access_adapter_widths)
                .collect(),
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
        ret.extend(self.memory.access_adapter_inventory.get_cells());
        ret
    }
}

/// Owned version of [MemoryAuxColsFactory].
pub struct SharedMemoryHelper<T> {
    pub(crate) range_checker: SharedVariableRangeCheckerChip,
    pub(crate) timestamp_lt_air: AssertLtSubAir,
    pub(crate) _marker: PhantomData<T>,
}

/// A helper for generating trace values in auxiliary memory columns related to the offline memory
/// argument.
pub struct MemoryAuxColsFactory<'a, T> {
    pub(crate) range_checker: &'a VariableRangeCheckerChip,
    pub(crate) timestamp_lt_air: AssertLtSubAir,
    pub(crate) _marker: PhantomData<T>,
}

// NOTE[jpw]: The `make_*_aux_cols` functions should be thread-safe so they can be used in
// parallelized trace generation.
impl<F: PrimeField32> MemoryAuxColsFactory<'_, F> {
    /// Fill the trace assuming `prev_timestamp` is already provided in `buffer`.
    pub fn fill(&self, prev_timestamp: u32, timestamp: u32, buffer: &mut MemoryBaseAuxCols<F>) {
        self.generate_timestamp_lt(prev_timestamp, timestamp, &mut buffer.timestamp_lt_aux);
        // Safety: even if prev_timestamp were obtained by transmute_ref from
        // `buffer.prev_timestamp`, this should still work because it is a direct assignment
        buffer.prev_timestamp = F::from_canonical_u32(prev_timestamp);
    }

    /// # Safety
    /// We assume that `F::ZERO` has underlying memory equivalent to `mem::zeroed()`.
    pub fn fill_zero(&self, buffer: &mut MemoryBaseAuxCols<F>) {
        *buffer = unsafe { std::mem::zeroed() };
    }

    fn generate_timestamp_lt(
        &self,
        prev_timestamp: u32,
        timestamp: u32,
        buffer: &mut LessThanAuxCols<F, AUX_LEN>,
    ) {
        debug_assert!(
            prev_timestamp < timestamp,
            "prev_timestamp {prev_timestamp} >= timestamp {timestamp}"
        );
        self.timestamp_lt_air.generate_subrow(
            (self.range_checker, prev_timestamp, timestamp),
            &mut buffer.lower_decomp,
        );
    }
}

impl<T> SharedMemoryHelper<T> {
    pub fn as_borrowed(&self) -> MemoryAuxColsFactory<'_, T> {
        MemoryAuxColsFactory {
            range_checker: self.range_checker.as_ref(),
            timestamp_lt_air: self.timestamp_lt_air,
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use openvm_circuit_primitives::var_range::{
        SharedVariableRangeCheckerChip, VariableRangeCheckerBus,
    };
    use openvm_stark_backend::{interaction::BusIndex, p3_field::FieldAlgebra};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use rand::{thread_rng, Rng};

    use super::MemoryController;
    use crate::{
        arch::{testing::MEMORY_BUS, MemoryConfig},
        system::memory::offline_checker::MemoryBus,
    };

    const RANGE_CHECKER_BUS: BusIndex = 3;

    #[test]
    fn test_no_adapter_records_for_singleton_accesses() {
        type F = BabyBear;

        let memory_bus = MemoryBus::new(MEMORY_BUS);
        let memory_config = MemoryConfig::default();
        let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, memory_config.decomp);
        let range_checker = SharedVariableRangeCheckerChip::new(range_bus);

        let mut memory_controller = MemoryController::<F>::with_volatile_memory(
            memory_bus,
            memory_config.clone(),
            range_checker.clone(),
        );

        let mut rng = thread_rng();
        for _ in 0..1000 {
            // TODO[jpw]: test other address spaces?
            let address_space = 4u32;
            let pointer = rng.gen_range(0..1 << memory_config.pointer_max_bits);

            if rng.gen_bool(0.5) {
                let data = F::from_canonical_u32(rng.gen_range(0..1 << 30));
                // address space is 4 so cell type is `F`
                unsafe {
                    memory_controller
                        .memory
                        .write::<F, 1, 1>(address_space, pointer, [data]);
                }
            } else {
                unsafe {
                    memory_controller
                        .memory
                        .read::<F, 1, 1>(address_space, pointer);
                }
            }
        }
        assert!(memory_controller
            .memory
            .access_adapter_inventory
            .get_heights()
            .iter()
            .all(|&h| h == 0));
    }
}
