//! [MemoryController] can be considered as the Memory Chip Complex for the CPU Backend.
use std::{collections::BTreeMap, fmt::Debug, marker::PhantomData, sync::Arc};

use getset::{Getters, MutGetters};
use openvm_circuit_primitives::{
    assert_less_than::{AssertLtSubAir, LessThanAuxCols},
    var_range::{
        SharedVariableRangeCheckerChip, VariableRangeCheckerBus, VariableRangeCheckerChip,
    },
    Chip, TraceSubRowGenerator,
};
use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::{
    interaction::PermutationCheckBus, p3_field::PrimeField32, p3_util::log2_strict_usize,
    prover::AirProvingContext, StarkProtocolConfig,
};
use serde::{Deserialize, Serialize};

use self::interface::MemoryInterface;
use super::AddressMap;
use crate::{
    arch::{MemoryConfig, VmField, CONST_BLOCK_SIZE},
    system::{
        memory::{
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
pub struct MemoryController<F: VmField> {
    pub memory_bus: MemoryBus,
    pub interface_chip: MemoryInterface<F>,
    pub range_checker: SharedVariableRangeCheckerChip,
    pub(crate) memory_config: MemoryConfig,
    // Store separately to avoid smart pointer reference each time
    range_checker_bus: VariableRangeCheckerBus,
    pub(crate) hasher_chip: Option<Arc<Poseidon2PeripheryChip<F>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersistentMemoryTraceHeights {
    boundary: usize,
    merkle: usize,
}
impl PersistentMemoryTraceHeights {
    /// `heights` must consist of only memory trace heights, in order of AIR IDs.
    pub fn from_slice(heights: &[u32]) -> Self {
        Self {
            boundary: heights[0] as usize,
            merkle: heights[1] as usize,
        }
    }
}

impl<F: VmField> MemoryController<F> {
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
        let interface_chip = MemoryInterface {
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
            interface_chip,
            memory_config: mem_config,
            range_checker,
            range_checker_bus,
            hasher_chip: Some(hasher_chip),
        }
    }

    pub fn memory_config(&self) -> &MemoryConfig {
        &self.memory_config
    }

    pub(crate) fn set_override_trace_heights(&mut self, overridden_heights: &[u32]) {
        let oh = PersistentMemoryTraceHeights::from_slice(overridden_heights);
        self.interface_chip
            .boundary_chip
            .set_overridden_height(oh.boundary);
        self.interface_chip
            .merkle_chip
            .set_overridden_height(oh.merkle);
    }

    /// This only sets the initial memory image for the boundary and merkle tree chips.
    /// Tracing memory should be set separately.
    pub(crate) fn set_initial_memory(&mut self, memory: AddressMap) {
        self.interface_chip.initial_memory = memory;
    }

    pub fn memory_bridge(&self) -> MemoryBridge {
        MemoryBridge::new(
            self.memory_bus,
            self.memory_config().timestamp_max_bits,
            self.range_checker_bus,
        )
    }

    pub fn helper(&self) -> SharedMemoryHelper<F> {
        let range_bus = self.range_checker.bus();
        SharedMemoryHelper {
            range_checker: self.range_checker.clone(),
            timestamp_lt_air: AssertLtSubAir::new(
                range_bus,
                self.memory_config().timestamp_max_bits,
            ),
            _marker: Default::default(),
        }
    }

    // @dev: Memory is complicated and allowed to break all the rules (e.g., 1 arena per chip) and
    // there's no need for any memory chip to implement the Chip trait. We do it when convenient,
    // but all that matters is that you can tracegen all the trace matrices for the memory AIRs
    // _somehow_.
    pub fn generate_proving_ctx<SC: StarkProtocolConfig<F = F>>(
        &mut self,
        touched_memory: TouchedMemory<F>,
    ) -> Vec<AirProvingContext<CpuBackend<SC>>> {
        let final_memory = touched_memory;
        let MemoryInterface {
            boundary_chip,
            merkle_chip,
            initial_memory,
        } = &mut self.interface_chip;

        let hasher = self.hasher_chip.as_ref().unwrap();
        boundary_chip.finalize(initial_memory, &final_memory, hasher.as_ref());

        // Rechunk CONST_BLOCK_SIZE blocks into CHUNK-sized blocks for merkle_chip
        // Note: Equipartition key is (addr_space, ptr) where ptr is the starting pointer
        let final_memory_values: Equipartition<F, CHUNK> = {
            use std::collections::BTreeMap;
            let mut chunk_map: BTreeMap<(u32, u32), [F; CHUNK]> = BTreeMap::new();
            for ((addr_space, ptr), ts_values) in final_memory {
                // Align to CHUNK boundary to get the chunk's starting pointer
                let chunk_ptr = (ptr / CHUNK as u32) * CHUNK as u32;
                let block_idx_in_chunk = ((ptr % CHUNK as u32) / CONST_BLOCK_SIZE as u32) as usize;
                let entry = chunk_map.entry((addr_space, chunk_ptr)).or_insert_with(|| {
                    // Initialize with values from initial memory
                    std::array::from_fn(|i| unsafe {
                        initial_memory.get_f::<F>(addr_space, chunk_ptr + i as u32)
                    })
                });
                // Copy values for this block
                for (i, val) in ts_values.values.into_iter().enumerate() {
                    entry[block_idx_in_chunk * CONST_BLOCK_SIZE + i] = val;
                }
            }
            chunk_map
        };
        merkle_chip.finalize(initial_memory, &final_memory_values, hasher.as_ref());

        vec![
            boundary_chip.generate_proving_ctx(()),
            merkle_chip.generate_proving_ctx(),
        ]
    }

    /// Return the number of AIRs in the memory controller.
    pub fn num_airs(&self) -> usize {
        2
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
        buffer.prev_timestamp = F::from_u32(prev_timestamp);
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
