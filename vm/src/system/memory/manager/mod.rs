use std::{
    array::{self, from_fn},
    cell::RefCell,
    collections::BTreeMap,
    iter,
    marker::PhantomData,
    rc::Rc,
    sync::Arc,
};

use ax_circuit_primitives::{
    assert_less_than::{AssertLtSubAir, LessThanAuxCols},
    is_less_than::IsLtSubAir,
    is_zero::IsZeroSubAir,
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    TraceSubRowGenerator,
};
use ax_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    prover::types::AirProofInput,
    rap::AnyRap,
};
use axvm_instructions::exe::MemoryImage;
use getset::Getters;
use itertools::{izip, zip_eq, Itertools};
pub use memory::{MemoryReadRecord, MemoryWriteRecord};
use p3_air::BaseAir;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use self::interface::MemoryInterface;
use super::{merkle::DirectCompressionBus, volatile::VolatileBoundaryChip};
use crate::{
    arch::{hasher::HasherChip, MemoryConfig},
    system::memory::{
        adapter::AccessAdapterAir,
        manager::memory::AccessAdapterRecord,
        offline_checker::{
            MemoryBridge, MemoryBus, MemoryReadAuxCols, MemoryReadOrImmediateAuxCols,
            MemoryWriteAuxCols, AUX_LEN,
        },
    },
};

pub mod dimensions;
mod interface;
pub(super) mod memory;
mod trace;

use crate::system::memory::{
    dimensions::MemoryDimensions,
    manager::memory::{Memory, INITIAL_TIMESTAMP},
    merkle::{MemoryMerkleBus, MemoryMerkleChip},
    persistent::PersistentBoundaryChip,
    tree::MemoryNode,
};

pub const CHUNK: usize = 8;
/// The offset of the Merkle AIR in AIRs of MemoryController.
pub const MERKLE_AIR_OFFSET: usize = 1;
/// The offset of the boundary AIR in AIRs of MemoryController.
pub const BOUNDARY_AIR_OFFSET: usize = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimestampedValues<T, const N: usize> {
    pub timestamp: u32,
    pub values: [T; N],
}

#[derive(Clone, Debug)]
pub struct MemoryControllerResult<F> {
    traces: Vec<RowMajorMatrix<F>>,
    public_values: Vec<Vec<F>>,
}

pub type MemoryControllerRef<F> = Rc<RefCell<MemoryController<F>>>;

/// A equipartition of memory, with timestamps and values.
///
/// The key is a pair `(address_space, label)`, where `label` is the index of the block in the
/// partition. I.e., the starting address of the block is `(address_space, label * N)`.
///
/// If a key is not present in the map, then the block is uninitialized (and therefore zero).
pub type TimestampedEquipartition<F, const N: usize> =
    BTreeMap<(F, usize), TimestampedValues<F, N>>;

/// A equipartition of memory values.
///
/// The key is a pair `(address_space, label)`, where `label` is the index of the block in the
/// partition. I.e., the starting address of the block is `(address_space, label * N)`.
///
/// If a key is not present in the map, then the block is uninitialized (and therefore zero).
pub type Equipartition<F, const N: usize> = BTreeMap<(F, usize), [F; N]>;

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
    /// Maps a length to a list of access adapters with that block length as th larger size.
    adapter_records: FxHashMap<usize, Vec<AccessAdapterRecord<F>>>,

    /// If set, the height of the traces will be overridden.
    overridden_heights: Option<MemoryTraceHeights>,

    // Filled during finalization.
    result: Option<MemoryControllerResult<F>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryTraceHeights {
    Volatile(VolatileMemoryTraceHeights),
    Persistent(PersistentMemoryTraceHeights),
}

impl MemoryTraceHeights {
    fn access_adapters_ref(&self) -> &FxHashMap<usize, usize> {
        match self {
            MemoryTraceHeights::Volatile(oh) => &oh.access_adapters,
            MemoryTraceHeights::Persistent(oh) => &oh.access_adapters,
        }
    }
    fn flatten(&self) -> Vec<usize> {
        match self {
            MemoryTraceHeights::Volatile(oh) => oh.flatten(),
            MemoryTraceHeights::Persistent(oh) => oh.flatten(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatileMemoryTraceHeights {
    boundary: usize,
    access_adapters: FxHashMap<usize, usize>,
}

impl VolatileMemoryTraceHeights {
    pub fn flatten(&self) -> Vec<usize> {
        iter::once(self.boundary)
            .chain(self.access_adapters.iter().sorted().map(|(_, &v)| v))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentMemoryTraceHeights {
    boundary: usize,
    merkle: usize,
    access_adapters: FxHashMap<usize, usize>,
}
impl PersistentMemoryTraceHeights {
    pub fn flatten(&self) -> Vec<usize> {
        vec![self.boundary, self.merkle]
            .into_iter()
            .chain(self.access_adapters.iter().sorted().map(|(_, v)| *v))
            .collect()
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
        mut overridden_heights: Option<MemoryTraceHeights>,
    ) -> Self {
        if let Some(overridden_heights) = overridden_heights.as_ref() {
            match overridden_heights {
                MemoryTraceHeights::Volatile { .. } => {}
                _ => panic!("Expect overridden_heights to be MemoryTraceHeights::Volatile"),
            }
            assert!(
                mem_config.boundary_air_height.is_none(),
                "Both mem_config.boundary_air_height and overridden_heights are set"
            );
        } else {
            // A temporary hack to support the old code.
            if let Some(boundary_air_height) = mem_config.boundary_air_height {
                overridden_heights =
                    Some(MemoryTraceHeights::Volatile(VolatileMemoryTraceHeights {
                        boundary: boundary_air_height,
                        access_adapters: FxHashMap::default(),
                    }));
            }
        }
        let range_checker_bus = range_checker.bus();
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
            memory: Memory::new(&Equipartition::<_, 1>::new()),
            adapter_records: FxHashMap::default(),
            range_checker,
            range_checker_bus,
            result: None,
            overridden_heights,
        }
    }

    pub fn with_persistent_memory(
        memory_bus: MemoryBus,
        mem_config: MemoryConfig,
        range_checker: Arc<VariableRangeCheckerChip>,
        merkle_bus: MemoryMerkleBus,
        compression_bus: DirectCompressionBus,
        initial_memory: Equipartition<F, CHUNK>,
        overridden_heights: Option<MemoryTraceHeights>,
    ) -> Self {
        if let Some(overridden_heights) = overridden_heights.as_ref() {
            match overridden_heights {
                MemoryTraceHeights::Persistent { .. } => {}
                _ => panic!("Expect overridden_heights to be MemoryTraceHeights::Persistent"),
            }
        }
        let memory_dims = MemoryDimensions {
            as_height: mem_config.as_height,
            address_height: mem_config.pointer_max_bits - log2_strict_usize(CHUNK),
            as_offset: 1,
        };
        let memory = Memory::new(&initial_memory);
        let range_checker_bus = range_checker.bus();
        let interface_chip = MemoryInterface::Persistent {
            boundary_chip: PersistentBoundaryChip::new(
                memory_dims,
                memory_bus,
                merkle_bus,
                compression_bus,
            ),
            merkle_chip: MemoryMerkleChip::new(memory_dims, merkle_bus, compression_bus),
            initial_memory,
        };
        Self {
            memory_bus,
            mem_config,
            interface_chip,
            memory,
            adapter_records: FxHashMap::default(),
            range_checker,
            range_checker_bus,
            result: None,
            overridden_heights,
        }
    }

    pub fn set_initial_memory(&mut self, memory: Equipartition<F, CHUNK>) {
        if self.timestamp() > INITIAL_TIMESTAMP + 1 {
            panic!("Cannot set initial memory after first timestamp");
        }
        match &mut self.interface_chip {
            MemoryInterface::Volatile { .. } => {
                if !memory.is_empty() {
                    panic!("Cannot set initial memory for volatile memory");
                }
            }
            MemoryInterface::Persistent { initial_memory, .. } => {
                *initial_memory = memory;
                self.memory = Memory::new(initial_memory);
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

    pub fn read_cell(&mut self, address_space: F, pointer: F) -> MemoryReadRecord<F, 1> {
        self.read(address_space, pointer)
    }

    pub fn read<const N: usize>(&mut self, address_space: F, pointer: F) -> MemoryReadRecord<F, N> {
        let ptr_u32 = pointer.as_canonical_u32();
        assert!(
            address_space == F::ZERO || ptr_u32 < (1 << self.mem_config.pointer_max_bits),
            "memory out of bounds: {ptr_u32:?}",
        );

        if address_space == F::ZERO {
            assert_eq!(N, 1, "cannot batch read from address space 0");

            let timestamp = self.timestamp();
            self.memory.increment_timestamp();

            return MemoryReadRecord {
                address_space,
                pointer,
                timestamp,
                prev_timestamp: 0,
                data: array::from_fn(|_| pointer),
            };
        }

        let (record, adapter_records) = self
            .memory
            .read::<N>(address_space.as_canonical_u32() as usize, ptr_u32 as usize);
        for record in adapter_records {
            self.adapter_records
                .entry(record.data.len())
                .or_default()
                .push(record);
        }

        for i in 0..N as u32 {
            let ptr = F::from_canonical_u32(ptr_u32 + i);
            self.interface_chip.touch_address(address_space, ptr);
        }

        record
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
        let addr_space = addr_space.as_canonical_u32() as usize;
        let ptr = ptr.as_canonical_u32() as usize;
        from_fn(|i| self.memory.get(addr_space, ptr + i))
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
        assert_ne!(address_space, F::ZERO);
        let ptr_u32 = pointer.as_canonical_u32();
        assert!(
            ptr_u32 < (1 << self.mem_config.pointer_max_bits),
            "memory out of bounds: {ptr_u32:?}",
        );

        let (record, adapter_records) = self.memory.write(
            address_space.as_canonical_u32() as usize,
            ptr_u32 as usize,
            data,
        );
        for record in adapter_records {
            self.adapter_records
                .entry(record.data.len())
                .or_default()
                .push(record);
        }

        for i in 0..N as u32 {
            let ptr = F::from_canonical_u32(ptr_u32 + i);
            self.interface_chip.touch_address(address_space, ptr);
        }

        record
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
        self.memory.increment_timestamp();
    }

    pub fn increment_timestamp_by(&mut self, change: u32) {
        self.memory.increment_timestamp_by(change);
    }

    pub fn increase_timestamp_to(&mut self, timestamp: u32) {
        self.memory
            .increment_timestamp_by(timestamp - self.memory.timestamp());
    }

    pub fn timestamp(&self) -> u32 {
        self.memory.timestamp()
    }

    fn access_adapter_air<const N: usize>(&self) -> AccessAdapterAir<N> {
        let lt_air = IsLtSubAir::new(self.range_checker.bus(), self.mem_config.clk_max_bits);
        AccessAdapterAir::<N> {
            memory_bus: self.memory_bus,
            lt_air,
        }
    }

    /// Returns the final memory state if persistent.
    pub fn finalize(
        &mut self,
        hasher: Option<&mut impl HasherChip<CHUNK, F>>,
    ) -> Option<Equipartition<F, CHUNK>> {
        if self.result.is_some() {
            panic!("Cannot finalize more than once");
        }
        let mut traces = vec![];
        let mut pvs = vec![];

        let (records, final_memory) = match &mut self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                let overridden_heights = self.overridden_heights.as_ref().map(|oh| match oh {
                    MemoryTraceHeights::Volatile(oh) => oh,
                    _ => unreachable!(),
                });
                let (final_memory, records) = self.memory.finalize::<1>();
                debug_assert_eq!(traces.len(), BOUNDARY_AIR_OFFSET);
                traces.push(
                    boundary_chip
                        .generate_trace(&final_memory, overridden_heights.map(|oh| oh.boundary)),
                );
                debug_assert_eq!(pvs.len(), BOUNDARY_AIR_OFFSET);
                pvs.push(vec![]);

                (records, None)
            }
            MemoryInterface::Persistent {
                merkle_chip,
                boundary_chip,
                initial_memory,
            } => {
                let overridden_heights = self.overridden_heights.as_ref().map(|oh| match oh {
                    MemoryTraceHeights::Persistent(oh) => oh,
                    _ => unreachable!(),
                });
                let hasher = hasher.unwrap();

                let (final_partition, records) = self.memory.finalize::<8>();
                traces.push(boundary_chip.generate_trace(
                    initial_memory,
                    &final_partition,
                    hasher,
                    overridden_heights.map(|oh| oh.boundary),
                ));
                pvs.push(vec![]);

                let final_memory_values = final_partition
                    .iter()
                    .map(|(key, value)| (*key, value.values))
                    .collect();

                let initial_node = MemoryNode::tree_from_memory(
                    merkle_chip.air.memory_dimensions,
                    initial_memory,
                    hasher,
                );
                let (expand_trace, final_node) = merkle_chip.generate_trace_and_final_tree(
                    &initial_node,
                    &final_memory_values,
                    hasher,
                    overridden_heights.map(|oh| oh.merkle),
                );

                debug_assert_eq!(traces.len(), MERKLE_AIR_OFFSET);
                traces.push(expand_trace);
                let mut expand_pvs = vec![];
                expand_pvs.extend(initial_node.hash());
                expand_pvs.extend(final_node.hash());
                debug_assert_eq!(pvs.len(), MERKLE_AIR_OFFSET);
                pvs.push(expand_pvs);
                (records, Some(final_memory_values))
            }
        };
        for record in records {
            self.adapter_records
                .entry(record.data.len())
                .or_default()
                .push(record);
        }

        self.add_access_adapter_trace::<2>(&mut traces, &mut pvs);
        self.add_access_adapter_trace::<4>(&mut traces, &mut pvs);
        self.add_access_adapter_trace::<8>(&mut traces, &mut pvs);
        self.add_access_adapter_trace::<16>(&mut traces, &mut pvs);
        self.add_access_adapter_trace::<32>(&mut traces, &mut pvs);
        self.add_access_adapter_trace::<64>(&mut traces, &mut pvs);

        self.result = Some(MemoryControllerResult {
            traces,
            public_values: pvs,
        });

        final_memory
    }
    fn add_access_adapter_trace<const N: usize>(
        &self,
        traces: &mut Vec<RowMajorMatrix<F>>,
        pvs: &mut Vec<Vec<F>>,
    ) {
        if self.mem_config.max_access_adapter_n >= N {
            traces.push(self.generate_access_adapter_trace::<N>());
            pvs.push(vec![]);
        }
    }

    pub fn generate_air_proof_inputs<SC: StarkGenericConfig>(self) -> Vec<AirProofInput<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let airs = self.airs();
        let MemoryControllerResult {
            traces,
            public_values,
        } = self.result.unwrap();
        izip!(airs, traces, public_values)
            .map(|(air, trace, pvs)| AirProofInput::simple(air, trace, pvs))
            .collect()
    }

    pub fn generate_traces(self) -> Vec<RowMajorMatrix<F>> {
        self.result.unwrap().traces
    }

    pub fn airs<SC: StarkGenericConfig>(&self) -> Vec<Arc<dyn AnyRap<SC>>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let mut airs = Vec::<Arc<dyn AnyRap<SC>>>::new();

        match &self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                debug_assert_eq!(airs.len(), BOUNDARY_AIR_OFFSET);
                airs.push(Arc::new(boundary_chip.air.clone()))
            }
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
                ..
            } => {
                debug_assert_eq!(airs.len(), BOUNDARY_AIR_OFFSET);
                airs.push(Arc::new(boundary_chip.air.clone()));
                debug_assert_eq!(airs.len(), MERKLE_AIR_OFFSET);
                airs.push(Arc::new(merkle_chip.air.clone()));
            }
        }

        self.add_access_adapter_air::<SC, 2>(&mut airs);
        self.add_access_adapter_air::<SC, 4>(&mut airs);
        self.add_access_adapter_air::<SC, 8>(&mut airs);
        self.add_access_adapter_air::<SC, 16>(&mut airs);
        self.add_access_adapter_air::<SC, 32>(&mut airs);
        self.add_access_adapter_air::<SC, 64>(&mut airs);

        airs
    }
    fn add_access_adapter_air<SC: StarkGenericConfig, const N: usize>(
        &self,
        airs: &mut Vec<Arc<dyn AnyRap<SC>>>,
    ) {
        if self.mem_config.max_access_adapter_n >= N {
            airs.push(Arc::new(self.access_adapter_air::<N>()));
        }
    }

    /// Return the number of AIRs in the memory controller.
    pub fn num_airs(&self) -> usize {
        let mut num_airs = 1;
        if self.continuation_enabled() {
            num_airs += 1;
        }
        for n in [2, 4, 8, 16, 32, 64] {
            if self.mem_config.max_access_adapter_n >= n {
                num_airs += 1;
            }
        }
        num_airs
    }

    pub fn air_names(&self) -> Vec<String> {
        let mut air_names = vec!["Boundary".to_string()];
        if self.continuation_enabled() {
            air_names.push("Merkle".to_string());
        }
        for n in [2, 4, 8, 16, 32, 64] {
            if self.mem_config.max_access_adapter_n >= n {
                air_names.push(format!("AccessAdapter<{}>", n));
            }
        }
        air_names
    }

    pub fn current_trace_heights(&self) -> Vec<usize> {
        self.get_memory_trace_heights().flatten()
    }

    pub fn get_memory_trace_heights(&self) -> MemoryTraceHeights {
        let access_adapters = [2, 4, 8, 16, 32, 64]
            .iter()
            .flat_map(|&n| {
                if self.mem_config.max_access_adapter_n >= n {
                    Some((
                        n,
                        self.adapter_records
                            .get(&n)
                            .map_or(0, |records| records.len()),
                    ))
                } else {
                    None
                }
            })
            .collect();
        match &self.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                MemoryTraceHeights::Volatile(VolatileMemoryTraceHeights {
                    boundary: boundary_chip.current_height(),
                    access_adapters,
                })
            }
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
                ..
            } => MemoryTraceHeights::Persistent(PersistentMemoryTraceHeights {
                boundary: boundary_chip.current_height(),
                merkle: merkle_chip.current_height(),
                access_adapters,
            }),
        }
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
                ..
            } => {
                widths.push(BaseAir::<F>::width(&boundary_chip.air));
                widths.push(BaseAir::<F>::width(&merkle_chip.air));
            }
        };
        self.add_access_adapter_width::<2>(&mut widths);
        self.add_access_adapter_width::<4>(&mut widths);
        self.add_access_adapter_width::<8>(&mut widths);
        self.add_access_adapter_width::<16>(&mut widths);
        self.add_access_adapter_width::<32>(&mut widths);
        self.add_access_adapter_width::<64>(&mut widths);
        widths
    }
    fn add_access_adapter_width<const N: usize>(&self, widths: &mut Vec<usize>) {
        if self.mem_config.max_access_adapter_n >= N {
            widths.push(BaseAir::<F>::width(&self.access_adapter_air::<N>()));
        }
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
    timestamp_lt_air: AssertLtSubAir,
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

    pub fn make_read_or_immediate_aux_cols(
        &self,
        read: MemoryReadRecord<F, 1>,
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
        write: MemoryWriteRecord<F, N>,
    ) -> MemoryWriteAuxCols<F, N> {
        MemoryWriteAuxCols::new(
            write.prev_data,
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

pub fn memory_image_to_equipartition<F: PrimeField32, const N: usize>(
    memory_image: MemoryImage<F>,
) -> Equipartition<F, { N }> {
    let mut result = Equipartition::new();
    for ((addr_space, addr), word) in memory_image {
        let addr_u32 = addr.as_canonical_u32();
        let shift = addr_u32 as usize % N;
        let key = (addr_space, (addr_u32 / N as u32) as usize);
        result.entry(key).or_insert([F::ZERO; N])[shift] = word;
    }
    result
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ax_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
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
            None,
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
        assert_eq!(memory_controller.adapter_records.len(), 0);
    }
}
