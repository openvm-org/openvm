//! Traits and builders to compose collections of chips into a virtual machine.
//!
//! A full VM extension consists of three components, represented by sub-traits:
//! - [VmExecutionExtension]
//! - [VmCircuitExtension]
//! - [VmProverExtension]: there may be multiple implementations of `VmProverExtension` for the same
//!   `VmCircuitExtension` for different prover backends.
//!
//! It is intended that `VmExecutionExtension` and `VmCircuitExtension` are implemented on the
//! same struct and `VmProverExtension` is implemented on a separate struct (usually a ZST) to
//! get around Rust orphan rules.
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
    engine::StarkEngine,
    interaction::BusIndex,
    keygen::types::MultiStarkProvingKey,
    prover::{
        cpu::CpuBackend,
        hal::ProverBackend,
        types::{AirProvingContext, ProvingContext},
    },
    rap::AnyRap,
    AirRef, AnyChip, Chip,
};
use rustc_hash::FxHashMap;
use tracing::info_span;

use super::{GenerationError, PhantomSubExecutor, SystemConfig};
use crate::{
    arch::Arena,
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

/// Extension of VM execution. Allows registration of custom execution of new instructions by
/// opcode.
pub trait VmExecutionExtension<F> {
    /// Enum of executor variants
    type Executor: AnyEnum;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Self::Executor>,
    ) -> Result<(), ExecutorInventoryError>;
}

/// Extension of the VM circuit. Allows _in-order_ addition of new AIRs with interactions.
pub trait VmCircuitExtension<SC: StarkGenericConfig> {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError>;
}

/// Extension of VM trace generation. The generics are `E` for [StarkEngine], `RA` for record arena,
/// and `EXT` for execution and circuit extension.
///
/// Note that this trait differs from [VmExecutionExtension] and [VmCircuitExtension]. This trait is
/// meant to be implemented on a separate ZST which may be different for different [ProverBackend]s.
/// This is done to get around Rust orphan rules.
pub trait VmProverExtension<E, RA, EXT>
where
    E: StarkEngine,
    EXT: VmExecutionExtension<Val<E::SC>> + VmCircuitExtension<E::SC>,
{
    /// The chips added to `inventory` should exactly match the order of AIRs in the
    /// [VmCircuitExtension] implementation of `EXT`.
    ///
    /// We do not provide access to the [ExecutorInventory] because the process to find an executor
    /// from the inventory seems more cumbersome than to simply re-construct any necessary executors
    /// directly within this function implementation.
    fn extend_prover(
        &self,
        extension: &EXT,
        inventory: &mut ChipInventory<E::SC, RA, E::PB>,
    ) -> Result<(), ChipInventoryError>;
}

// ======================= Different Inventory Struct Definitions =============================

pub struct ExecutorInventory<E> {
    config: SystemConfig,
    /// Lookup table to executor ID.
    /// This is stored in a hashmap because it is _not_ expected to be used in the hot path.
    /// A direct opcode -> executor mapping should be generated before runtime execution.
    pub instruction_lookup: FxHashMap<VmOpcode, ExecutorId>,
    pub executors: Vec<E>,
    /// `ext_start[i]` will have the starting index in `executors` for extension `i`
    ext_start: Vec<usize>,
}

// @dev: We need ExecutorInventoryBuilder separate from ExecutorInventory because of how
// ExecutorInventory::extend works: we want to build an inventory with some big E3 enum that
// includes both enum types E1, E2. However the interface for an ExecutionExtension will only know
// about the enum E2. In order to be able to allow access to the old executors with type E1 without
// referring to the type E1, we need to create this separate builder struct.
pub struct ExecutorInventoryBuilder<'a, F, E> {
    /// Chips that are already included in the chipset and may be used
    /// as dependencies. The order should be that depended-on chips are ordered
    /// **before** their dependents.
    old_executors: Vec<&'a dyn AnyEnum>,
    new_inventory: ExecutorInventory<E>,
    phantom_executors: FxHashMap<PhantomDiscriminant, Arc<dyn PhantomSubExecutor<F>>>,
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
    pub executor_idx_to_insertion_idx: Vec<usize>,
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
    #[allow(clippy::new_without_default)]
    pub fn new(config: SystemConfig) -> Self {
        Self {
            config,
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

    /// Extend the inventory with a new extension.
    /// A new inventory with different type generics is returned with the combined inventory.
    pub fn extend<F, E3, EXT>(
        self,
        other: &EXT,
    ) -> Result<ExecutorInventory<E3>, ExecutorInventoryError>
    where
        F: 'static,
        E: Into<E3> + AnyEnum,
        E3: AnyEnum,
        EXT: VmExecutionExtension<F>,
        EXT::Executor: Into<E3>,
    {
        let mut builder: ExecutorInventoryBuilder<F, EXT::Executor> = self.builder();
        other.extend_execution(&mut builder)?;
        let other_inventory = builder.new_inventory;
        let other_phantom_executors = builder.phantom_executors;
        let mut inventory_ext = self.transmute();
        inventory_ext.append(other_inventory.transmute())?;
        let phantom_chip: &mut PhantomExecutor<F> = inventory_ext
            .find_executor_mut()
            .next()
            .expect("system always has phantom chip");
        let phantom_executors = &mut phantom_chip.phantom_executors;
        for (discriminant, sub_executor) in other_phantom_executors {
            if phantom_executors
                .insert(discriminant, sub_executor)
                .is_some()
            {
                return Err(ExecutorInventoryError::PhantomSubExecutorExists { discriminant });
            }
        }

        Ok(inventory_ext)
    }

    pub fn builder<F, E2>(&self) -> ExecutorInventoryBuilder<'_, F, E2>
    where
        F: 'static,
        E: AnyEnum,
    {
        let old_executors = self.executors.iter().map(|e| e as &dyn AnyEnum).collect();
        ExecutorInventoryBuilder {
            old_executors,
            new_inventory: ExecutorInventory::new(self.config.clone()),
            phantom_executors: Default::default(),
        }
    }

    pub fn transmute<E2>(self) -> ExecutorInventory<E2>
    where
        E: Into<E2>,
    {
        ExecutorInventory {
            config: self.config,
            instruction_lookup: self.instruction_lookup,
            executors: self.executors.into_iter().map(|e| e.into()).collect(),
            ext_start: self.ext_start,
        }
    }

    /// Append `other` to current inventory. This means `self` comes earlier in the dependency
    /// chain.
    fn append(&mut self, mut other: ExecutorInventory<E>) -> Result<(), ExecutorInventoryError> {
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

    /// Returns the system config of the inventory.
    pub fn config(&self) -> &SystemConfig {
        &self.config
    }
}

impl<F, E> ExecutorInventoryBuilder<'_, F, E> {
    pub fn add_executor(
        &mut self,
        executor: impl Into<E>,
        opcodes: impl IntoIterator<Item = VmOpcode>,
    ) -> Result<(), ExecutorInventoryError> {
        self.new_inventory.add_executor(executor, opcodes)
    }

    pub fn add_phantom_sub_executor<PE>(
        &mut self,
        phantom_sub: PE,
        discriminant: PhantomDiscriminant,
    ) -> Result<(), ExecutorInventoryError>
    where
        E: AnyEnum,
        F: 'static,
        PE: PhantomSubExecutor<F> + 'static,
    {
        let existing = self
            .phantom_executors
            .insert(discriminant, Arc::new(phantom_sub));
        if existing.is_some() {
            return Err(ExecutorInventoryError::PhantomSubExecutorExists { discriminant });
        }
        Ok(())
    }

    pub fn find_executor<EX: 'static>(&self) -> impl Iterator<Item = &'_ EX>
    where
        E: AnyEnum,
    {
        self.old_executors
            .iter()
            .filter_map(|e| e.as_any_kind().downcast_ref())
    }

    /// Returns the maximum number of bits used to represent addresses in memory
    pub fn pointer_max_bits(&self) -> usize {
        self.new_inventory.config().memory_config.pointer_max_bits
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
    pub fn keygen<E: StarkEngine<SC = SC>>(self, engine: &E) -> MultiStarkProvingKey<SC> {
        let mut builder = engine.keygen_builder();
        for air in self.into_airs() {
            builder.add_air(air);
        }
        builder.generate_pk()
    }

    /// Returns the maximum number of bits used to represent addresses in memory
    pub fn pointer_max_bits(&self) -> usize {
        self.config.memory_config.pointer_max_bits
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
    pub fn new(airs: AirInventory<SC>) -> Self {
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

    // NOTE[jpw]: this is currently unused, it is for debugging purposes
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
        tracing::debug!("add_executor_chip: {}", type_name::<C>());
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
                    .unwrap_or_else(|| {
                        panic!(
                            "Attempt to subtract num_airs={num_airs} by {}",
                            insertion_idx + 1
                        )
                    })
            })
            .collect()
    }

    pub fn timestamp_max_bits(&self) -> usize {
        self.airs.config().memory_config.timestamp_max_bits
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
    RA: Arena,
    PB: ProverBackend,
    SCC: SystemChipComplex<RA, PB>,
{
    pub fn system_config(&self) -> &SystemConfig {
        self.inventory.config()
    }

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
            .chain(info_span!("system_trace_gen").in_scope(|| {
                self.system
                    .generate_proving_ctx(system_records, sys_record_arenas)
            }))
            .chain(
                zip(self.inventory.chips.iter().enumerate().rev(), record_arenas).map(
                    |((insertion_idx, chip), records)| {
                        // Only create a span if record is not empty:
                        let _span = (!records.is_empty()).then(|| {
                            let air_name = self.inventory.airs.ext_airs[insertion_idx].name();
                            info_span!("single_trace_gen", air = air_name).entered()
                        });
                        chip.generate_proving_ctx(records)
                    },
                ),
            )
            .enumerate()
            .filter(|(_air_id, ctx)| {
                (!ctx.cached_mains.is_empty() || ctx.common_main.is_some())
                    && ctx.main_trace_height() > 0
            })
            .collect();

        Ok(ProvingContext {
            per_air: ctx_without_empties,
        })
    }
}

// ============ Blanket implementation of VM extension traits for Option<E> ===========

impl<F, EXT: VmExecutionExtension<F>> VmExecutionExtension<F> for Option<EXT> {
    type Executor = EXT::Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Self::Executor>,
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
    use openvm_circuit_derive::AnyEnum;
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

    use super::*;
    use crate::{arch::VmCircuitConfig, system::memory::interface::MemoryInterfaceAirs};

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
        let config = SystemConfig::default();
        let inventory: AirInventory<BabyBearPoseidon2Config> = config.create_airs().unwrap();
        let system = inventory.system();
        let port = system.port();
        assert_eq!(port.execution_bus.index(), 0);
        assert_eq!(port.memory_bridge.memory_bus().index(), 1);
        assert_eq!(port.program_bus.index(), 2);
        assert_eq!(port.memory_bridge.range_bus().index(), 3);
        match &system.memory.interface {
            MemoryInterfaceAirs::Persistent { boundary, .. } => {
                assert_eq!(boundary.merkle_bus.index, 4);
                assert_eq!(boundary.compression_bus.index, 5);
            }
            _ => unreachable!(),
        };
    }
}
