use std::{
    any::{type_name, Any},
    iter::{self, zip},
    sync::Arc,
};

use getset::{CopyGetters, Getters};
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerAir,
};
use openvm_instructions::{PhantomDiscriminant, VmOpcode};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::BusIndex,
    keygen::{types::MultiStarkProvingKey, MultiStarkKeygenBuilder},
    prover::{
        cpu::CpuBackend,
        hal::ProverBackend,
        types::{AirProvingContext, ProvingContext},
    },
    rap::AnyRap,
    AirRef, AnyChip, Chip,
};
use rustc_hash::FxHashMap;

use super::{GenerationError, PhantomSubExecutor, SystemConfig};
use crate::{
    arch::MatrixRecordArena,
    system::{
        memory::{BOUNDARY_AIR_OFFSET, MERKLE_AIR_OFFSET},
        phantom::PhantomExecutor,
        SystemAirInventory, SystemChipComplex, SystemRecords,
    },
};

/// Global AIR ID in the VM circuit verifying key.
pub const PROGRAM_AIR_ID: usize = 0;
/// ProgramAir is the first AIR so its cached trace should be the first main trace.
pub const PROGRAM_CACHED_TRACE_INDEX: usize = 0;
pub const CONNECTOR_AIR_ID: usize = 1;
/// If PublicValuesAir is **enabled**, its AIR ID is 2. PublicValuesAir is always disabled when
/// continuations is enabled.
pub const PUBLIC_VALUES_AIR_ID: usize = 2;
/// AIR ID of the Memory Boundary AIR.
pub const BOUNDARY_AIR_ID: usize = PUBLIC_VALUES_AIR_ID + 1 + BOUNDARY_AIR_OFFSET;
/// If VM has continuations enabled, all AIRs of MemoryController are added after ConnectorChip.
/// Merkle AIR commits start/final memory states.
pub const MERKLE_AIR_ID: usize = CONNECTOR_AIR_ID + 1 + MERKLE_AIR_OFFSET;

pub type ExecutorId = u32;

// ======================= VM Extension Traits =============================

/// A full VM extension consists of three components, represented by sub-traits:
/// - [VmExecutionExtension]
/// - [VmCircuitExtension]
/// - [VmProverExtension]
pub trait VmExtension<SC, RA = MatrixRecordArena<Val<SC>>, PB = CpuBackend<SC>>:
    VmExecutionExtension<Val<SC>> + VmCircuitExtension<SC> + VmProverExtension<SC, RA, PB>
where
    SC: StarkGenericConfig,
    PB: ProverBackend,
{
}

impl<SC, RA, PB, EXT> VmExtension<SC, RA, PB> for EXT
where
    SC: StarkGenericConfig,
    PB: ProverBackend,
    EXT: VmExecutionExtension<Val<SC>> + VmCircuitExtension<SC> + VmProverExtension<SC, RA, PB>,
{
}

/// Extension of VM execution. Allows registration of custom execution of new instructions by
/// opcode.
pub trait VmExecutionExtension<F> {
    /// Enum of executor variants
    type Executor: AnyEnum;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventory<Self::Executor>,
    ) -> Result<(), ExecutorInventoryError>;
}

/// Extension of the VM circuit. Allows _in-order_ addition of new AIRs with interactions.
pub trait VmCircuitExtension<SC: StarkGenericConfig> {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError>;
}

/// Extension of VM trace generation.
/// The returned vector should exactly match the order of AIRs in [`VmCircuitExtension`] for this
/// extension.
pub trait VmProverExtension<SC, RA, PB>
where
    SC: StarkGenericConfig,
    PB: ProverBackend,
{
    /// We do not provide access to the [ExecutorInventory] because the process to find an executor
    /// from the inventory seems more cumbersome than to simply re-construct any necessary executors
    /// directly within this function implementation.
    fn extend_prover(
        &self,
        inventory: &mut ChipInventory<SC, RA, PB>,
    ) -> Result<(), ChipInventoryError>;
}

// ======================= Different Inventory Struct Definitions =============================

pub struct ExecutorInventory<E> {
    /// Lookup table to executor ID.
    /// This is stored in a hashmap because it is _not_ expected to be used in the hot path.
    /// A direct opcode -> executor mapping should be generated before runtime execution.
    pub instruction_lookup: FxHashMap<VmOpcode, ExecutorId>,
    pub executors: Vec<E>,
    /// `ext_start[i]` will have the starting index in `executors` for extension `i`
    ext_start: Vec<usize>,
}

#[derive(Clone, Getters, CopyGetters)]
pub struct AirInventory<SC: StarkGenericConfig> {
    #[get = "pub"]
    config: SystemConfig,
    /// The system AIRs required by the circuit architecture.
    #[get = "pub"]
    system: SystemAirInventory<SC>,
    /// List of all non-system AIRs in the circuit, in insertion order, which is the **reverse** of
    /// the order they appear in the verifying key.
    ///
    /// Note that the system will ensure that the first AIR in the list is always the
    /// [VariableRangeCheckerAir].
    #[get = "pub"]
    ext_airs: Vec<AirRef<SC>>,
    /// `ext_start[i]` will have the starting index in `ext_airs` for extension `i`
    ext_start: Vec<usize>,

    bus_idx_mgr: BusIndexManager,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BusIndexManager {
    /// All existing buses use indices in [0, bus_idx_max)
    bus_idx_max: BusIndex,
}

// @dev: ChipInventory does not have the SystemChipComplex because that is custom depending on `PB`.
// The full struct with SystemChipComplex is VmChipComplex
#[derive(Getters)]
pub struct ChipInventory<SC, RA, PB>
where
    SC: StarkGenericConfig,
    PB: ProverBackend,
{
    /// Read-only view of AIRs, as constructed via the [VmCircuitExtension] trait.
    #[get = "pub"]
    airs: AirInventory<SC>,
    /// Chips that are being built.
    #[get = "pub"]
    chips: Vec<Box<dyn AnyChip<RA, PB>>>,

    /// Number of extensions that have chips added, including the current one that is still being
    /// built.
    cur_num_exts: usize,
    /// Mapping from executor index to chip insertion index. Chips must be added in order so the
    /// chip insertion index matches the AIR insertion index. Reminder: this is in **reverse**
    /// order of the verifying key AIR ordering.
    ///
    /// Note: if public values chip exists, then it will be the first entry and point to
    /// `usize::MAX`. This entry should never be used.
    pub(crate) executor_idx_to_insertion_idx: Vec<usize>,
}

/// The collection of all chips in the VM. The chips should correspond 1-to-1 with the associated
/// [AirInventory]. The [VmChipComplex] coordinates the trace generation for all chips in the VM
/// after construction.
#[derive(Getters)]
pub struct VmChipComplex<SC, RA, PB, SCC>
where
    SC: StarkGenericConfig,
    PB: ProverBackend,
{
    /// System chip complex responsible for trace generation of [SystemAirInventory]
    pub system: SCC,
    pub inventory: ChipInventory<SC, RA, PB>,
}

// ======================= Inventory Function Definitions =============================

impl<E> ExecutorInventory<E> {
    /// Empty inventory should be created at the start of the declaration of a new extension.
    pub fn new() -> Self {
        Self {
            instruction_lookup: Default::default(),
            executors: Default::default(),
            ext_start: vec![0],
        }
    }

    /// Inserts an executor with the collection of opcodes that it handles.
    /// If some executor already owns one of the opcodes, an error is returned with the existing
    /// executor.
    pub fn add_executor(
        &mut self,
        executor: impl Into<E>,
        opcodes: impl IntoIterator<Item = VmOpcode>,
    ) -> Result<(), ExecutorInventoryError> {
        let opcodes: Vec<_> = opcodes.into_iter().collect();
        for opcode in &opcodes {
            if let Some(id) = self.instruction_lookup.get(opcode) {
                return Err(ExecutorInventoryError::ExecutorExists {
                    opcode: *opcode,
                    id: *id,
                });
            }
        }
        let id = self.executors.len();
        self.executors.push(executor.into());
        for opcode in opcodes {
            self.instruction_lookup
                .insert(opcode, id.try_into().unwrap());
        }
        Ok(())
    }

    /// The generic `F` must match that of the `PhantomChip<F>`.
    pub fn add_phantom_sub_executor<F, PE>(
        &mut self,
        phantom_sub: PE,
        discriminant: PhantomDiscriminant,
    ) -> Result<(), ExecutorInventoryError>
    where
        E: AnyEnum,
        F: 'static,
        PE: PhantomSubExecutor<F> + 'static,
    {
        let phantom_chip: &mut PhantomExecutor<F> = self
            .find_executor_mut()
            .next()
            .expect("system always has phantom chip");
        let existing = phantom_chip.add_sub_executor(phantom_sub, discriminant);
        if existing.is_some() {
            return Err(ExecutorInventoryError::PhantomSubExecutorExists { discriminant });
        }
        Ok(())
    }

    /// Extend the inventory with a new extension.
    /// A new inventory with different type generics is returned with the combined inventory.
    pub fn extend<F, E3, EXT>(
        self,
        other: &EXT,
    ) -> Result<ExecutorInventory<E3>, ExecutorInventoryError>
    where
        EXT: VmExecutionExtension<F>,
        E: Into<E3> + AnyEnum,
        EXT::Executor: Into<E3>,
    {
        let mut other_inventory = ExecutorInventory::new();
        other.extend_execution(&mut other_inventory)?;
        let mut inventory_ext = self.transmute();
        inventory_ext.append(other_inventory.transmute())?;
        Ok(inventory_ext)
    }

    pub fn transmute<E2>(self) -> ExecutorInventory<E2>
    where
        E: Into<E2>,
    {
        ExecutorInventory {
            instruction_lookup: self.instruction_lookup,
            executors: self.executors.into_iter().map(|e| e.into()).collect(),
            ext_start: self.ext_start,
        }
    }

    /// Append `other` to current inventory. This means `self` comes earlier in the dependency
    /// chain.
    pub fn append(
        &mut self,
        mut other: ExecutorInventory<E>,
    ) -> Result<(), ExecutorInventoryError> {
        let num_executors = self.executors.len();
        for (opcode, mut id) in other.instruction_lookup.into_iter() {
            id = id.checked_add(num_executors.try_into().unwrap()).unwrap();
            if let Some(old_id) = self.instruction_lookup.insert(opcode, id) {
                return Err(ExecutorInventoryError::ExecutorExists { opcode, id: old_id });
            }
        }
        for id in &mut other.ext_start {
            *id = id.checked_add(num_executors).unwrap();
        }
        self.executors.append(&mut other.executors);
        self.ext_start.append(&mut other.ext_start);
        Ok(())
    }

    pub fn get_executor(&self, opcode: VmOpcode) -> Option<&E> {
        let id = self.instruction_lookup.get(&opcode)?;
        self.executors.get(*id as usize)
    }

    pub fn get_mut_executor(&mut self, opcode: &VmOpcode) -> Option<&mut E> {
        let id = self.instruction_lookup.get(opcode)?;
        self.executors.get_mut(*id as usize)
    }

    pub fn executors(&self) -> &[E] {
        &self.executors
    }

    pub fn find_executor<EX: 'static>(&self) -> impl Iterator<Item = &'_ EX>
    where
        E: AnyEnum,
    {
        self.executors
            .iter()
            .filter_map(|e| e.as_any_kind().downcast_ref())
    }

    pub fn find_executor_mut<EX: 'static>(&mut self) -> impl Iterator<Item = &'_ mut EX>
    where
        E: AnyEnum,
    {
        self.executors
            .iter_mut()
            .filter_map(|e| e.as_any_kind_mut().downcast_mut())
    }
}

impl<SC: StarkGenericConfig> AirInventory<SC> {
    /// Outside of this crate, [AirInventory] must be constructed via [SystemConfig].
    pub(crate) fn new(
        config: SystemConfig,
        system: SystemAirInventory<SC>,
        bus_idx_mgr: BusIndexManager,
    ) -> Self {
        Self {
            config,
            system,
            ext_start: Vec::new(),
            ext_airs: Vec::new(),
            bus_idx_mgr,
        }
    }

    /// This should be called **exactly once** at the start of the declaration of a new extension.
    pub fn start_new_extension(&mut self) {
        self.ext_start.push(self.ext_airs.len());
    }

    pub fn new_bus_idx(&mut self) -> BusIndex {
        self.bus_idx_mgr.new_bus_idx()
    }

    /// Looks through already-defined AIRs to see if there exists any of type `A` by downcasting.
    /// Returns all chips of type `A` in the circuit.
    ///
    /// This should not be used to look for system AIRs.
    pub fn find_air<A: 'static>(&self) -> impl Iterator<Item = &'_ A> {
        self.ext_airs
            .iter()
            .filter_map(|air| air.as_any().downcast_ref())
    }

    pub fn add_air<A: AnyRap<SC> + 'static>(&mut self, air: A) {
        self.add_air_ref(Arc::new(air));
    }

    pub fn add_air_ref(&mut self, air: AirRef<SC>) {
        self.ext_airs.push(air);
    }

    pub fn range_checker(&self) -> &VariableRangeCheckerAir {
        self.find_air()
            .next()
            .expect("system always has range checker AIR")
    }

    /// The AIRs in the order they appear in the verifying key.
    /// This is the system AIRs, followed by the other AIRs in the **reverse** of the order they
    /// were added in the VM extension definitions. In particular, the AIRs that have dependencies
    /// appear later. The system guarantees that the last AIR is the [VariableRangeCheckerAir].
    pub fn into_airs(self) -> impl Iterator<Item = AirRef<SC>> {
        self.system
            .into_airs()
            .into_iter()
            .chain(self.ext_airs.into_iter().rev())
    }

    /// This is O(1). Returns the total number of AIRs and equals the length of [`Self::into_airs`].
    pub fn num_airs(&self) -> usize {
        self.config.num_airs() + self.ext_airs.len()
    }

    /// Standalone function to generate proving key and verifying key for this circuit.
    pub fn keygen(self, stark_config: &SC) -> MultiStarkProvingKey<SC> {
        let mut builder = MultiStarkKeygenBuilder::new(stark_config);
        for air in self.into_airs() {
            builder.add_air(air);
        }
        builder.generate_pk()
    }
}

impl BusIndexManager {
    pub fn new() -> Self {
        Self { bus_idx_max: 0 }
    }

    pub fn new_bus_idx(&mut self) -> BusIndex {
        let idx = self.bus_idx_max;
        self.bus_idx_max = self.bus_idx_max.checked_add(1).unwrap();
        idx
    }
}

impl<SC, RA, PB> ChipInventory<SC, RA, PB>
where
    SC: StarkGenericConfig,
    PB: ProverBackend,
{
    pub(crate) fn new(airs: AirInventory<SC>) -> Self {
        Self {
            airs,
            chips: Vec::new(),
            cur_num_exts: 0,
            executor_idx_to_insertion_idx: Vec::new(),
        }
    }

    pub fn config(&self) -> &SystemConfig {
        &self.airs.config
    }

    pub fn start_new_extension(&mut self) -> Result<(), ChipInventoryError> {
        if self.cur_num_exts >= self.airs.ext_start.len() {
            return Err(ChipInventoryError::MissingCircuitExtension(
                self.airs.ext_start.len(),
            ));
        }
        if self.chips.len() != self.airs.ext_start[self.cur_num_exts] {
            return Err(ChipInventoryError::MissingChip {
                actual: self.chips.len(),
                expected: self.airs.ext_start[self.cur_num_exts],
            });
        }

        self.cur_num_exts += 1;
        Ok(())
    }

    /// Gets the next AIR from the pre-existing AIR inventory according to the index of the next
    /// chip to be built.
    pub fn next_air<A: 'static>(&self) -> Result<&A, ChipInventoryError> {
        let cur_idx = self.chips.len();
        self.airs
            .ext_airs
            .get(cur_idx)
            .and_then(|air| air.as_any().downcast_ref())
            .ok_or_else(|| ChipInventoryError::AirNotFound {
                name: type_name::<A>().to_string(),
            })
    }

    /// Looks through built chips to see if there exists any of type `C` by downcasting.
    /// Returns all chips of type `C` in the chipset.
    ///
    /// Note: the type `C` will usually be a smart pointer to a chip.
    pub fn find_chip<C: 'static>(&self) -> impl Iterator<Item = &'_ C> {
        self.chips.iter().filter_map(|c| c.as_any().downcast_ref())
    }

    /// Adds a chip that is not associated with any executor, as defined by the
    /// [VmExecutionExtension] trait.
    pub fn add_periphery_chip<C: Chip<RA, PB> + 'static>(&mut self, chip: C) {
        self.chips.push(Box::new(chip));
    }

    /// Adds a chip and associates it to the next executor.
    /// **Caution:** you must add chips in the order matching the order that executors were added in
    /// the [VmExecutionExtension] implementation.
    pub fn add_executor_chip<C: Chip<RA, PB> + 'static>(&mut self, chip: C) {
        self.executor_idx_to_insertion_idx.push(self.chips.len());
        self.chips.push(Box::new(chip));
    }

    /// Returns the mapping from executor index to the AIR index, where AIR index is the index of
    /// the AIR within the verifying key.
    ///
    /// This should only be called after the `ChipInventory` is fully built.
    pub fn executor_idx_to_air_idx(&self) -> Vec<usize> {
        let num_airs = self.airs.num_airs();
        assert_eq!(
            num_airs,
            self.config().num_airs() + self.chips.len(),
            "Number of chips does not match number of AIRs"
        );
        // system AIRs are at the front of vkey, and then insertion index is the reverse ordering of
        // AIR index
        self.executor_idx_to_insertion_idx
            .iter()
            .map(|insertion_idx| {
                num_airs
                    .checked_sub(insertion_idx.checked_add(1).unwrap())
                    .unwrap()
            })
            .collect()
    }
}

// SharedVariableRangeCheckerChip is only used by the CPU backend.
impl<SC, RA> ChipInventory<SC, RA, CpuBackend<SC>>
where
    SC: StarkGenericConfig,
{
    pub fn range_checker(&self) -> Result<&SharedVariableRangeCheckerChip, ChipInventoryError> {
        self.find_chip::<SharedVariableRangeCheckerChip>()
            .next()
            .ok_or_else(|| ChipInventoryError::ChipNotFound {
                name: "VariableRangeCheckerChip".to_string(),
            })
    }
}

// ================================== Error Types =====================================

#[derive(thiserror::Error, Debug)]
pub enum ExecutorInventoryError {
    #[error("Opcode {opcode} already owned by executor id {id}")]
    ExecutorExists { opcode: VmOpcode, id: ExecutorId },
    #[error("Phantom discriminant {} already has sub-executor", .discriminant.0)]
    PhantomSubExecutorExists { discriminant: PhantomDiscriminant },
}

#[derive(thiserror::Error, Debug)]
pub enum AirInventoryError {
    #[error("AIR {name} not found")]
    AirNotFound { name: String },
}

#[derive(thiserror::Error, Debug)]
pub enum ChipInventoryError {
    #[error("Air {name} not found")]
    AirNotFound { name: String },
    #[error("Chip {name} not found")]
    ChipNotFound { name: String },
    #[error("Adding prover extension without execution extension. Number of execution extensions is {0}")]
    MissingExecutionExtension(usize),
    #[error(
        "Adding prover extension without circuit extension. Number of circuit extensions is {0}"
    )]
    MissingCircuitExtension(usize),
    #[error("Missing chip. Number of chips is {actual}, expected number is {expected}")]
    MissingChip { actual: usize, expected: usize },
    #[error("Missing executor chip. Number of executors with associated chips is {actual}, expected number is {expected}")]
    MissingExecutor { actual: usize, expected: usize },
}

// ======================= VM Chip Complex Implementation =============================

impl<SC, RA, PB, SCC> VmChipComplex<SC, RA, PB, SCC>
where
    SC: StarkGenericConfig,
    PB: ProverBackend,
    SCC: SystemChipComplex<RA, PB>,
{
    pub fn system_config(&self) -> &SystemConfig {
        self.inventory.config()
    }
    // pub fn finalize_memory(&mut self)
    // where
    //     P: AnyEnum,
    // {
    //     if self.config.continuation_enabled {
    //         let chip = self
    //             .inventory
    //             .periphery
    //             .get_mut(Self::POSEIDON2_PERIPHERY_IDX)
    //             .expect("Poseidon2 chip required for persistent memory");
    //         let hasher: &mut Poseidon2PeripheryChip<F> = chip
    //             .as_any_kind_mut()
    //             .downcast_mut()
    //             .expect("Poseidon2 chip required for persistent memory");
    //         self.base.memory_controller.finalize(Some(hasher));
    //     } else {
    //         self.base
    //             .memory_controller
    //             .finalize(None::<&mut Poseidon2PeripheryChip<F>>);
    //     };
    // }

    // TODO: move these two into SystemChipComplex trait
    // pub(crate) fn set_program(&mut self, program: Program<F>) {
    //     self.base.program_chip.set_program(program);
    // }

    // pub(crate) fn set_initial_memory(&mut self, memory: MemoryImage) {
    //     self.base.memory_controller.set_initial_memory(memory);
    // }

    // // This is O(1).
    // pub fn num_airs(&self) -> usize {
    //     3 + self.memory_controller().num_airs() + self.inventory.num_airs()
    // }

    // Note[jpw]: do we still need this?
    /// Return trace heights of (SystemBase, Inventory). Usually this is for aggregation and not
    /// useful for regular users.
    ///
    /// **Warning**: the order of `get_trace_heights` is deterministic, but it is not the same as
    /// the order of `air_names`. In other words, the order here does not match the order of AIR
    /// IDs.
    // pub fn get_internal_trace_heights(&self) -> VmComplexTraceHeights
    // where
    //     E: ChipUsageGetter,
    //     P: ChipUsageGetter,
    // {
    //     VmComplexTraceHeights::new(
    //         self.base.get_system_trace_heights(),
    //         self.inventory.get_trace_heights(),
    //     )
    // }

    // /// Return dummy trace heights of (SystemBase, Inventory). Usually this is for aggregation to
    // /// generate a dummy proof and not useful for regular users.
    // ///
    // /// **Warning**: the order of `get_dummy_trace_heights` is deterministic, but it is not the
    // same /// as the order of `air_names`. In other words, the order here does not match the
    // order of /// AIR IDs.
    // pub fn get_dummy_internal_trace_heights(&self) -> VmComplexTraceHeights
    // where
    //     E: ChipUsageGetter,
    //     P: ChipUsageGetter,
    // {
    //     VmComplexTraceHeights::new(
    //         self.base.get_dummy_system_trace_heights(),
    //         self.inventory.get_dummy_trace_heights(),
    //     )
    // }

    /// `record_arenas` is expected to have length equal to the number of AIRs in the verifying key
    /// and in the same order as the AIRs appearing in the verifying key, even though some chips may
    /// not require a record arena.
    pub(crate) fn generate_proving_ctx(
        &mut self,
        system_records: SystemRecords<PB::Val>,
        record_arenas: Vec<RA>,
        // trace_height_constraints: &[LinearConstraint],
    ) -> Result<ProvingContext<PB>, GenerationError> {
        // ATTENTION: The order of AIR proving context generation MUST be consistent with
        // `AirInventory::into_airs`.

        // Execution has finished at this point.
        // ASSUMPTION WHICH MUST HOLD: non-system chips do not have a dependency on the system chips
        // during trace generation. Given this assumption, we can generate trace on the system chips
        // first.
        let num_sys_airs = self.system_config().num_airs();
        let num_airs = num_sys_airs + self.inventory.chips.len();
        if num_airs != record_arenas.len() {
            return Err(GenerationError::UnexpectedNumArenas {
                actual: record_arenas.len(),
                expected: num_airs,
            });
        }
        let mut _record_arenas = record_arenas;
        let record_arenas = _record_arenas.split_off(num_sys_airs);
        let sys_record_arenas = _record_arenas;

        // First go through all system chips
        // Then go through all other chips in inventory in **reverse** order they were added (to
        // resolve dependencies)
        //
        // Perf[jpw]: currently we call tracegen on each chip **serially** (although tracegen per
        // chip is parallelized). We could introduce more parallelism, while potentially increasing
        // the peak memory usage, by keeping a dependency tree and generating traces at the same
        // layer of the tree in parallel.
        let ctx_without_empties: Vec<(usize, AirProvingContext<_>)> = iter::empty()
            .chain(
                self.system
                    .generate_proving_ctx(system_records, sys_record_arenas),
            )
            .chain(
                zip(&mut self.inventory.chips, record_arenas)
                    .map(|(chip, records)| chip.generate_proving_ctx(records))
                    .rev(),
            )
            .enumerate()
            .filter(|(_air_id, ctx)| ctx.main_trace_height() > 0)
            .collect();

        // TODO: move out to VirtualMachine
        // // Defensive checks that the trace heights satisfy the linear constraints:
        // let idx_trace_heights = ctx_without_empties
        //     .iter()
        //     .map(|(air_idx, ctx)| (*air_idx, ctx.main_trace_height()))
        //     .collect_vec();
        // if let Some(&(air_idx, height)) = idx_trace_heights
        //     .iter()
        //     .find(|(_, height)| *height > self.max_trace_height)
        // {
        //     return Err(GenerationError::TraceHeightsLimitExceeded {
        //         air_idx,
        //         height,
        //         max_height: self.max_trace_height,
        //     });
        // }
        // if trace_height_constraints.is_empty() {
        //     tracing::warn!("generating proof input without trace height constraints");
        // }
        // for (i, constraint) in trace_height_constraints.iter().enumerate() {
        //     let value = idx_trace_heights
        //         .iter()
        //         .map(|&(air_idx, h)| constraint.coefficients[air_idx] as u64 * h as u64)
        //         .sum::<u64>();

        //     if value >= constraint.threshold as u64 {
        //         tracing::info!(
        //             "trace heights {:?} violate linear constraint {} ({} >= {})",
        //             idx_trace_heights,
        //             i,
        //             value,
        //             constraint.threshold
        //         );
        //         return Err(GenerationError::LinearTraceHeightConstraintExceeded {
        //             constraint_idx: i,
        //             value,
        //             threshold: constraint.threshold,
        //         });
        //     }
        // }

        Ok(ProvingContext {
            per_air: ctx_without_empties,
        })
    }

    // TODO[jpw]: This doesn't belong here!
    // #[cfg(feature = "bench-metrics")]
    // fn finalize_metrics(&self, metrics: &mut VmMetrics)
    // where
    //     E: ChipUsageGetter,
    //     P: ChipUsageGetter,
    // {
    //     tracing::info!(metrics.cycle_count);
    //     counter!("total_cycles").absolute(metrics.cycle_count as u64);
    //     counter!("main_cells_used")
    //         .absolute(self.current_trace_cells().into_iter().sum::<usize>() as u64);

    //     if self.config.profiling {
    //         metrics.chip_heights =
    //             itertools::izip!(self.air_names(), self.current_trace_heights()).collect();
    //         metrics.emit();
    //     }
    // }
}

// ============ Blanket implementation of VM extension traits for Option<E> ===========

impl<F, EXT: VmExecutionExtension<F>> VmExecutionExtension<F> for Option<EXT> {
    type Executor = EXT::Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventory<Self::Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        if let Some(extension) = self {
            extension.extend_execution(inventory)
        } else {
            Ok(())
        }
    }
}

impl<SC: StarkGenericConfig, EXT: VmCircuitExtension<SC>> VmCircuitExtension<SC> for Option<EXT> {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        if let Some(extension) = self {
            extension.extend_circuit(inventory)
        } else {
            Ok(())
        }
    }
}

impl<SC, RA, PB, EXT> VmProverExtension<SC, RA, PB> for Option<EXT>
where
    SC: StarkGenericConfig,
    PB: ProverBackend,
    EXT: VmProverExtension<SC, RA, PB>,
{
    fn extend_prover(
        &self,
        inventory: &mut ChipInventory<SC, RA, PB>,
    ) -> Result<(), ChipInventoryError> {
        if let Some(extension) = self {
            extension.extend_prover(inventory)
        } else {
            Ok(())
        }
    }
}

/// A helper trait for downcasting types that may be enums.
pub trait AnyEnum {
    /// Recursively "unwraps" enum and casts to `Any` for downcasting.
    fn as_any_kind(&self) -> &dyn Any;

    /// Recursively "unwraps" enum and casts to `Any` for downcasting.
    fn as_any_kind_mut(&mut self) -> &mut dyn Any;
}

impl AnyEnum for () {
    fn as_any_kind(&self) -> &dyn Any {
        self
    }
    fn as_any_kind_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;

    use super::*;
    use crate::system::memory::interface::MemoryInterface;

    #[allow(dead_code)]
    #[derive(Copy, Clone)]
    enum EnumA {
        A(u8),
        B(u32),
    }

    enum EnumB {
        C(u64),
        D(EnumA),
    }

    #[derive(AnyEnum)]
    enum EnumC {
        C(u64),
        #[any_enum]
        D(EnumA),
    }

    impl AnyEnum for EnumA {
        fn as_any_kind(&self) -> &dyn Any {
            match self {
                EnumA::A(a) => a,
                EnumA::B(b) => b,
            }
        }

        fn as_any_kind_mut(&mut self) -> &mut dyn Any {
            match self {
                EnumA::A(a) => a,
                EnumA::B(b) => b,
            }
        }
    }

    impl AnyEnum for EnumB {
        fn as_any_kind(&self) -> &dyn Any {
            match self {
                EnumB::C(c) => c,
                EnumB::D(d) => d.as_any_kind(),
            }
        }

        fn as_any_kind_mut(&mut self) -> &mut dyn Any {
            match self {
                EnumB::C(c) => c,
                EnumB::D(d) => d.as_any_kind_mut(),
            }
        }
    }

    #[test]
    fn test_any_enum_downcast() {
        let a = EnumA::A(1);
        assert_eq!(a.as_any_kind().downcast_ref::<u8>(), Some(&1));
        let b = EnumB::D(a);
        assert!(b.as_any_kind().downcast_ref::<u64>().is_none());
        assert!(b.as_any_kind().downcast_ref::<EnumA>().is_none());
        assert_eq!(b.as_any_kind().downcast_ref::<u8>(), Some(&1));
        let c = EnumB::C(3);
        assert_eq!(c.as_any_kind().downcast_ref::<u64>(), Some(&3));
        let d = EnumC::D(a);
        assert!(d.as_any_kind().downcast_ref::<u64>().is_none());
        assert!(d.as_any_kind().downcast_ref::<EnumA>().is_none());
        assert_eq!(d.as_any_kind().downcast_ref::<u8>(), Some(&1));
        let e = EnumC::C(3);
        assert_eq!(e.as_any_kind().downcast_ref::<u64>(), Some(&3));
    }

    #[test]
    fn test_system_bus_indices() {
        let config = SystemConfig::default().with_continuations();
        let complex = SystemComplex::<BabyBear>::new(config);
        assert_eq!(complex.base.execution_bus().index(), 0);
        assert_eq!(complex.base.memory_bus().index(), 1);
        assert_eq!(complex.base.program_bus().index(), 2);
        assert_eq!(complex.base.range_checker_bus().index(), 3);
        match &complex.memory_controller().interface_chip {
            MemoryInterface::Persistent { boundary_chip, .. } => {
                assert_eq!(boundary_chip.air.merkle_bus.index, 4);
                assert_eq!(boundary_chip.air.compression_bus.index, 5);
            }
            _ => unreachable!(),
        };
    }
}
