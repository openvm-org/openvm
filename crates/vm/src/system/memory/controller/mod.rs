//! [MemoryController] can be considered as the Memory Chip Complex for the CPU Backend.
use std::{collections::BTreeMap, fmt::Debug, iter, marker::PhantomData, sync::Arc};

use getset::{Getters, MutGetters};
use openvm_circuit_primitives::{
    assert_less_than::{AssertLtSubAir, LessThanAuxCols},
    utils::next_power_of_two_or_zero,
    var_range::{
        SharedVariableRangeCheckerChip, VariableRangeCheckerBus, VariableRangeCheckerChip,
    },
    TraceSubRowGenerator,
};
use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig},
    interaction::PermutationCheckBus,
    p3_commit::PolynomialSpace,
    p3_field::{Field, PrimeField32},
    p3_maybe_rayon::prelude::{IntoParallelIterator, ParallelIterator},
    p3_util::{log2_ceil_usize, log2_strict_usize},
    prover::{cpu::CpuBackend, types::AirProvingContext},
    Chip, ChipUsageGetter,
};
use serde::{Deserialize, Serialize};

use self::interface::MemoryInterface;
use super::{volatile::VolatileBoundaryChip, AddressMap};
use crate::{
    arch::{DenseRecordArena, MemoryConfig, ADDR_SPACE_OFFSET},
    system::{
        memory::{
            adapter::AccessAdapterInventory,
            dimensions::MemoryDimensions,
            merkle::MemoryMerkleChip,
            offline_checker::{MemoryBaseAuxCols, MemoryBridge, MemoryBus, AUX_LEN},
            persistent::PersistentBoundaryChip,
        },
        poseidon2::Poseidon2PeripheryChip,
        TouchedMemory,
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
pub struct MemoryController<F: Field> {
    pub memory_bus: MemoryBus,
    pub interface_chip: MemoryInterface<F>,
    #[getset(get = "pub")]
    pub(crate) mem_config: MemoryConfig,
    pub range_checker: SharedVariableRangeCheckerChip,
    // Store separately to avoid smart pointer reference each time
    range_checker_bus: VariableRangeCheckerBus,
    access_adapter_inventory: AccessAdapterInventory<F>,
    pub(crate) hasher_chip: Option<Arc<Poseidon2PeripheryChip<F>>>,
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
            access_adapter_inventory: AccessAdapterInventory::new(
                range_checker.clone(),
                memory_bus,
                mem_config.clk_max_bits,
                mem_config.max_access_adapter_n,
            ),
            range_checker,
            range_checker_bus,
            hasher_chip: None,
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
        hasher_chip: Arc<Poseidon2PeripheryChip<F>>,
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
            access_adapter_inventory: AccessAdapterInventory::new(
                range_checker.clone(),
                memory_bus,
                mem_config.clk_max_bits,
                mem_config.max_access_adapter_n,
            ),
            range_checker,
            range_checker_bus,
            hasher_chip: Some(hasher_chip),
        }
    }

    pub fn set_override_trace_heights(&mut self, overridden_heights: MemoryTraceHeights) {
        match &mut self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => match overridden_heights {
                MemoryTraceHeights::Volatile(oh) => {
                    boundary_chip.set_overridden_height(oh.boundary);
                    self.access_adapter_inventory
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
                    self.access_adapter_inventory
                        .set_override_trace_heights(oh.access_adapters);
                }
                _ => panic!("Expect overridden_heights to be MemoryTraceHeights::Persistent"),
            },
        }
    }

    /// This only sets the initial memory image for the persistent boundary and merkle tree chips.
    /// Tracing memory should be set separately.
    pub(crate) fn set_initial_memory(&mut self, memory: AddressMap) {
        match &mut self.interface_chip {
            MemoryInterface::Volatile { .. } => {
                // Skip initialization for volatile memory
            }
            MemoryInterface::Persistent { initial_memory, .. } => {
                *initial_memory = memory;
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

    // @dev: Memory is complicated and allowed to break all the rules (e.g., 1 arena per chip) and
    // there's no need for any memory chip to implement the Chip trait. We do it when convenient,
    // but all that matters is that you can tracegen all the trace matrices for the memory AIRs
    // _somehow_.
    pub fn generate_proving_ctx<SC: StarkGenericConfig>(
        &mut self,
        access_adapter_records: DenseRecordArena,
        touched_memory: TouchedMemory<F>,
    ) -> Vec<AirProvingContext<CpuBackend<SC>>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        match (&mut self.interface_chip, touched_memory) {
            (
                MemoryInterface::Volatile { boundary_chip },
                TouchedMemory::Volatile(final_memory),
            ) => {
                boundary_chip.finalize(final_memory);
            }
            (
                MemoryInterface::Persistent {
                    boundary_chip,
                    merkle_chip,
                    initial_memory,
                },
                TouchedMemory::Persistent(final_memory),
            ) => {
                let hasher = self.hasher_chip.as_ref().unwrap();
                boundary_chip.finalize(initial_memory, &final_memory, hasher.as_ref());
                let final_memory_values = final_memory
                    .into_par_iter()
                    .map(|(key, value)| (key, value.values))
                    .collect();
                merkle_chip.finalize(initial_memory, &final_memory_values, hasher.as_ref());
            }
            _ => panic!("TouchedMemory incorrect type"),
        }

        let mut ret = Vec::new();

        let access_adapters = &mut self.access_adapter_inventory;
        access_adapters.set_arena(access_adapter_records);
        match &mut self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                ret.push(boundary_chip.generate_proving_ctx(()));
            }
            MemoryInterface::Persistent {
                merkle_chip,
                boundary_chip,
                ..
            } => {
                debug_assert_eq!(ret.len(), BOUNDARY_AIR_OFFSET);
                ret.push(boundary_chip.generate_proving_ctx(()));
                debug_assert_eq!(ret.len(), MERKLE_AIR_OFFSET);
                ret.push(merkle_chip.generate_proving_ctx());
            }
        }
        ret.extend(access_adapters.generate_proving_ctx());
        ret
    }

    /// Return the number of AIRs in the memory controller.
    pub fn num_airs(&self) -> usize {
        let mut num_airs = 1;
        if self.continuation_enabled() {
            num_airs += 1;
        }
        num_airs += self.access_adapter_inventory.num_access_adapters();
        num_airs
    }

    // The following functions are for instrumentation but not necessarily required by any traits.
    // They may be deleted in the future.

    pub fn current_trace_heights(&self) -> Vec<usize> {
        self.get_memory_trace_heights().flatten()
    }

    pub fn get_memory_trace_heights(&self) -> MemoryTraceHeights {
        let access_adapters = self.access_adapter_inventory.get_heights();
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
        let access_adapters = vec![1; self.access_adapter_inventory.num_access_adapters()];
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
        ret.extend(self.access_adapter_inventory.get_cells());
        ret
    }
}

/// Owned version of [MemoryAuxColsFactory].
#[derive(Clone)]
pub struct SharedMemoryHelper<F> {
    pub(crate) range_checker: SharedVariableRangeCheckerChip,
    pub(crate) timestamp_lt_air: AssertLtSubAir,
    pub(crate) _marker: PhantomData<F>,
}

impl<F> SharedMemoryHelper<F> {
    pub fn new(range_checker: SharedVariableRangeCheckerChip, timestamp_max_bits: usize) -> Self {
        let timestamp_lt_air = AssertLtSubAir::new(range_checker.bus(), timestamp_max_bits);
        Self {
            range_checker,
            timestamp_lt_air,
            _marker: PhantomData,
        }
    }
}

/// A helper for generating trace values in auxiliary memory columns related to the offline memory
/// argument.
pub struct MemoryAuxColsFactory<'a, F> {
    pub(crate) range_checker: &'a VariableRangeCheckerChip,
    pub(crate) timestamp_lt_air: AssertLtSubAir,
    pub(crate) _marker: PhantomData<F>,
}

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

impl<F> SharedMemoryHelper<F> {
    pub fn as_borrowed(&self) -> MemoryAuxColsFactory<'_, F> {
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
