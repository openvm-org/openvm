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
    sync::Arc,
};

use getset::{CopyGetters, Getters};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerAir},
    Chip, ColumnsAir,
};
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::GpuBackend;
use openvm_instructions::{PhantomDiscriminant, VmOpcode};
use openvm_stark_backend::{
    interaction::BusIndex,
    keygen::{types::MultiStarkProvingKey, MultiStarkKeygenBuilder},
    prover::{AirProvingContext, MatrixDimensions, ProverBackend, ProvingContext},
    AirRef, AnyAir, StarkEngine, StarkProtocolConfig, Val,
};
use rustc_hash::FxHashMap;
use tracing::info_span;

#[cfg(feature = "cuda")]
use super::cuda::postflight::{GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript};
use super::{GenerationError, PhantomSubExecutor, Postflight, PostflightError, SystemConfig};
#[cfg(feature = "cuda")]
use crate::system::cuda::SystemChipInventoryGPU;
use crate::system::{
    memory::{BOUNDARY_AIR_OFFSET, MERKLE_AIR_OFFSET},
    phantom::PhantomExecutor,
    SystemAirInventory, SystemChipComplex, SystemChipInventory,
};

/// Global AIR ID in the VM circuit verifying key.
pub const PROGRAM_AIR_ID: usize = 0;
/// ProgramAir is the first AIR so its cached trace should be the first main trace.
pub const PROGRAM_CACHED_TRACE_INDEX: usize = 0;
pub const CONNECTOR_AIR_ID: usize = 1;
/// Starting AIR index of memory AIRs in the VM circuit.
pub const MEMORY_AIRS_START_IDX: usize = 2;
/// AIR index of the boundary AIR in the VM circuit.
pub const BOUNDARY_AIR_ID: usize = MEMORY_AIRS_START_IDX + BOUNDARY_AIR_OFFSET;
/// If VM has continuations enabled, all AIRs of MemoryController are added after ConnectorChip.
/// Merkle AIR commits start/final memory states.
pub const MERKLE_AIR_ID: usize = MEMORY_AIRS_START_IDX + MERKLE_AIR_OFFSET;

pub type ExecutorId = u32;

/// AIR trait object combining [`AnyAir`] (used by stark-backend for proving) with
/// [`ColumnsAir`] (OpenVM-internal column-name introspection used by external tooling). The
/// blanket impl below makes every type satisfying both traits also satisfy this one, so existing
/// concrete AIRs need no changes.
///
/// Trait upcasting (stable since Rust 1.86) coerces `Arc<dyn AnyAirWithColumns<SC>>` to
/// `Arc<dyn AnyAir<SC>>` in argument position, so [`AirRefWithColumns`] passes transparently to
/// stark-backend APIs that expect [`AirRef`](openvm_stark_backend::AirRef).
pub trait AnyAirWithColumns<SC: StarkProtocolConfig>: AnyAir<SC> + ColumnsAir {}

impl<SC, T> AnyAirWithColumns<SC> for T
where
    SC: StarkProtocolConfig,
    T: AnyAir<SC> + ColumnsAir,
{
}

/// Reference-counted dyn pointer to an AIR with column-name introspection.
pub type AirRefWithColumns<SC> = Arc<dyn AnyAirWithColumns<SC>>;

// ======================= VM Extension Traits =============================

/// Extension of VM execution. Allows registration of custom execution of new instructions by
/// opcode.
pub trait VmExecutionExtension {
    /// Enum of executor variants
    type Executor: AnyEnum;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<Self::Executor>,
    ) -> Result<(), ExecutorInventoryError>;
}

/// Extension of the VM circuit. Allows _in-order_ addition of new AIRs with interactions.
pub trait VmCircuitExtension<SC: StarkProtocolConfig> {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError>;
}

/// Backend-specific trace generation for one VM extension.
///
/// Note that this trait differs from [VmExecutionExtension] and [VmCircuitExtension]. This trait is
/// meant to be implemented on a separate ZST which may be different for different [ProverBackend]s.
/// This is done to get around Rust orphan rules.
pub trait VmProverExtension<E, EXT>
where
    E: StarkEngine,
    EXT: VmExecutionExtension + VmCircuitExtension<E::SC>,
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
        inventory: &mut ChipInventory<E::SC, E::PB>,
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

// @dev: We need ExecutorInventoryBuilder separate from ExecutorInventory because extension
// composition builds a combined executor enum while each extension only knows its own executor
// enum. The builder keeps access to existing executors without naming the final combined enum.
pub struct ExecutorInventoryBuilder<'a, E> {
    /// Chips that are already included in the chipset and may be used
    /// as dependencies. The order should be that depended-on chips are ordered
    /// **before** their dependents.
    old_executors: Vec<&'a dyn AnyEnum>,
    new_inventory: ExecutorInventory<E>,
    phantom_executors: FxHashMap<PhantomDiscriminant, Arc<dyn PhantomSubExecutor>>,
}

#[derive(Clone, Getters, CopyGetters)]
pub struct AirInventory<SC: StarkProtocolConfig> {
    #[get = "pub"]
    config: SystemConfig,
    /// The system AIRs required by the circuit architecture.
    #[get = "pub"]
    system: SystemAirInventory,
    /// List of all non-system AIRs in the circuit, in insertion order, which is the **reverse** of
    /// the order they appear in the verifying key.
    ///
    /// Note that the system will ensure that the first AIR in the list is always the
    /// [VariableRangeCheckerAir].
    #[get = "pub"]
    ext_airs: Vec<AirRefWithColumns<SC>>,
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
struct InventoryChip<PB: ProverBackend> {
    value: Box<dyn Any>,
    constant_trace_height: Option<usize>,
    postflight_generator: Option<PostflightGenerator<PB>>,
}

impl<PB: ProverBackend> InventoryChip<PB> {
    fn new<C: 'static>(value: C, constant_trace_height: Option<usize>) -> Self {
        Self {
            value: Box::new(value),
            constant_trace_height,
            postflight_generator: None,
        }
    }

    fn with_postflight_generator(mut self, generator: PostflightGenerator<PB>) -> Self {
        self.postflight_generator = Some(generator);
        self
    }

    fn as_any(&self) -> &dyn Any {
        self.value.as_ref()
    }
}

#[derive(Getters)]
pub struct ChipInventory<SC, PB>
where
    SC: StarkProtocolConfig,
    PB: ProverBackend,
{
    /// Read-only view of AIRs, as constructed via the [VmCircuitExtension] trait.
    #[get = "pub"]
    airs: AirInventory<SC>,
    /// Chips that are being built.
    chips: Vec<InventoryChip<PB>>,

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

type PostflightGenerator<PB> = Box<
    dyn for<'a> Fn(
            &dyn Any,
            &Postflight<'a, <PB as ProverBackend>::Val>,
        ) -> Result<AirProvingContext<PB>, PostflightError>
        + Send
        + Sync,
>;

fn erase_postflight_generator<PB, C, G>(generate: G) -> PostflightGenerator<PB>
where
    PB: ProverBackend,
    C: 'static,
    G: for<'a> Fn(&C, &Postflight<'a, PB::Val>) -> Result<AirProvingContext<PB>, PostflightError>
        + Send
        + Sync
        + 'static,
{
    Box::new(move |chip, postflight| {
        let chip = chip
            .downcast_ref::<C>()
            .expect("postflight generator was registered with this concrete chip type");
        generate(chip, postflight)
    })
}

/// The collection of all chips in the VM. The chips should correspond 1-to-1 with the associated
/// [AirInventory]. The [VmChipComplex] coordinates the trace generation for all chips in the VM
/// after construction.
#[derive(Getters)]
pub struct VmChipComplex<SC, PB, SCC>
where
    SC: StarkProtocolConfig,
    PB: ProverBackend,
{
    /// System chip complex responsible for trace generation of [SystemAirInventory]
    pub system: SCC,
    pub inventory: ChipInventory<SC, PB>,
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
    pub fn extend<CombinedE, EXT>(
        self,
        other: &EXT,
    ) -> Result<ExecutorInventory<CombinedE>, ExecutorInventoryError>
    where
        E: Into<CombinedE> + AnyEnum,
        CombinedE: AnyEnum,
        EXT: VmExecutionExtension,
        EXT::Executor: Into<CombinedE>,
    {
        let mut builder: ExecutorInventoryBuilder<EXT::Executor> = self.builder();
        other.extend_execution(&mut builder)?;
        let other_inventory = builder.new_inventory;
        let other_phantom_executors = builder.phantom_executors;
        let mut inventory_ext = self.transmute();
        inventory_ext.append(other_inventory.transmute())?;
        let phantom_chip: &mut PhantomExecutor = inventory_ext
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

    pub fn builder<NewE>(&self) -> ExecutorInventoryBuilder<'_, NewE>
    where
        E: AnyEnum,
    {
        let old_executors = self.executors.iter().map(|e| e as &dyn AnyEnum).collect();
        ExecutorInventoryBuilder {
            old_executors,
            new_inventory: ExecutorInventory::new(self.config.clone()),
            phantom_executors: Default::default(),
        }
    }

    pub fn transmute<TargetE>(self) -> ExecutorInventory<TargetE>
    where
        E: Into<TargetE>,
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

impl<E> ExecutorInventoryBuilder<'_, E> {
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
        PE: PhantomSubExecutor + 'static,
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

impl<SC: StarkProtocolConfig> AirInventory<SC> {
    /// Outside of this crate, [AirInventory] must be constructed via [SystemConfig].
    pub(crate) fn new(
        config: SystemConfig,
        system: SystemAirInventory,
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

    pub fn add_air<A: AnyAirWithColumns<SC> + 'static>(&mut self, air: A) {
        self.add_air_ref(Arc::new(air));
    }

    pub fn add_air_ref(&mut self, air: AirRefWithColumns<SC>) {
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
    pub fn into_airs(self) -> impl Iterator<Item = AirRefWithColumns<SC>> {
        self.system
            .into_airs()
            .into_iter()
            .chain(self.ext_airs.into_iter().rev())
    }

    /// Generates the proving key for this circuit, marking the system AIRs that must be present
    /// in any valid proof (see [`SystemConfig::is_required_air_id`]) as required.
    pub fn keygen(self, config: &SC) -> MultiStarkProvingKey<SC> {
        let system_config = self.config.clone();
        let mut keygen_builder = MultiStarkKeygenBuilder::new(config.clone());
        for (air_id, air) in self.into_airs().enumerate() {
            if system_config.is_required_air_id(air_id) {
                keygen_builder.add_required_air(air as AirRef<_>);
            } else {
                keygen_builder.add_air(air as AirRef<_>);
            }
        }
        keygen_builder.generate_pk().unwrap()
    }

    /// This is O(1). Returns the total number of AIRs and equals the length of [`Self::into_airs`].
    pub fn num_airs(&self) -> usize {
        self.config.num_airs() + self.ext_airs.len()
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

impl<SC, PB> ChipInventory<SC, PB>
where
    SC: StarkProtocolConfig,
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

    pub(crate) fn num_chips(&self) -> usize {
        self.chips.len()
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
    pub fn add_periphery_chip<C: Chip<PB> + 'static>(&mut self, chip: C) {
        let constant_trace_height = chip.constant_trace_height();
        self.add_periphery_chip_with_height(chip, constant_trace_height);
    }

    pub fn add_periphery_chip_with_height<C: 'static>(
        &mut self,
        chip: C,
        constant_trace_height: Option<usize>,
    ) {
        self.chips
            .push(InventoryChip::new(chip, constant_trace_height));
    }

    /// Adds a chip and associates it to the next executor.
    /// **Caution:** you must add chips in the order matching the order that executors were added in
    /// the [VmExecutionExtension] implementation.
    pub fn add_executor_chip<C: 'static>(&mut self, chip: C) {
        tracing::debug!("add_executor_chip: {}", type_name::<C>());
        self.executor_idx_to_insertion_idx.push(self.chips.len());
        self.chips.push(InventoryChip::new(chip, None));
    }

    /// Adds a periphery chip with its CPU trace generator over postflight history.
    pub fn add_periphery_chip_with_tracegen<C, G>(&mut self, chip: C, generate: G)
    where
        C: Chip<PB> + 'static,
        G: for<'a> Fn(
                &C,
                &Postflight<'a, PB::Val>,
            ) -> Result<AirProvingContext<PB>, PostflightError>
            + Send
            + Sync
            + 'static,
    {
        let constant_trace_height = chip.constant_trace_height();
        self.add_periphery_chip_with_height_and_tracegen(chip, constant_trace_height, generate);
    }

    /// Adds a periphery chip with an explicit trace height and CPU trace generator.
    pub fn add_periphery_chip_with_height_and_tracegen<C, G>(
        &mut self,
        chip: C,
        constant_trace_height: Option<usize>,
        generate: G,
    ) where
        C: 'static,
        G: for<'a> Fn(
                &C,
                &Postflight<'a, PB::Val>,
            ) -> Result<AirProvingContext<PB>, PostflightError>
            + Send
            + Sync
            + 'static,
    {
        self.chips.push(
            InventoryChip::new(chip, constant_trace_height)
                .with_postflight_generator(erase_postflight_generator(generate)),
        );
    }

    /// Adds an executor chip with its CPU trace generator over postflight history.
    pub fn add_executor_chip_with_tracegen<C, G>(&mut self, chip: C, generate: G)
    where
        C: 'static,
        G: for<'a> Fn(
                &C,
                &Postflight<'a, PB::Val>,
            ) -> Result<AirProvingContext<PB>, PostflightError>
            + Send
            + Sync
            + 'static,
    {
        tracing::debug!("add_executor_chip: {}", type_name::<C>());
        self.executor_idx_to_insertion_idx.push(self.chips.len());
        self.chips.push(
            InventoryChip::new(chip, None)
                .with_postflight_generator(erase_postflight_generator(generate)),
        );
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

    /// Returns constant trace heights for all AIRs in verifying key order.
    /// System AIRs get `None` (their constant heights are handled separately).
    /// Extension chips follow in the same order as AIRs in the verifying key
    /// (reversed insertion order).
    pub fn constant_trace_heights(&self) -> Vec<Option<usize>> {
        let num_system = self.airs.config().num_airs();
        let mut heights = vec![None; num_system];
        heights.extend(
            self.chips
                .iter()
                .rev()
                .map(|chip| chip.constant_trace_height),
        );
        heights
    }
}

// SharedVariableRangeCheckerChip is only used by the CPU backend.
impl<SC> ChipInventory<SC, CpuBackend<SC>>
where
    SC: StarkProtocolConfig,
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

impl<SC, PB, SCC> VmChipComplex<SC, PB, SCC>
where
    SC: StarkProtocolConfig,
    PB: ProverBackend,
    SCC: SystemChipComplex<PB>,
{
    pub fn system_config(&self) -> &SystemConfig {
        self.inventory.config()
    }
}

impl<SC> VmChipComplex<SC, CpuBackend<SC>, SystemChipInventory<SC>>
where
    SC: StarkProtocolConfig,
    Val<SC>: super::VmField,
{
    /// Generates CPU traces directly from immutable preflight history.
    pub(crate) fn generate_proving_ctx_from_postflight(
        &mut self,
        postflight: &Postflight<'_, Val<SC>>,
    ) -> Result<ProvingContext<CpuBackend<SC>>, GenerationError> {
        let sys_ctxs = {
            let _span = info_span!("system_trace_gen").entered();
            self.system.generate_proving_ctx_from_postflight(postflight)
        };

        let mut exec_ctxs = Vec::new();
        exec_ctxs.resize_with(self.inventory.chips.len(), || None);
        {
            let _span = info_span!("executor_trace_gen").entered();
            for (chain_pos, (insertion_idx, chip)) in
                self.inventory.chips.iter().enumerate().rev().enumerate()
            {
                let air_name = self.inventory.airs.ext_airs[insertion_idx].name();
                let _air_span = info_span!("single_trace_gen", air = air_name).entered();
                let generator = chip.postflight_generator.as_ref().ok_or_else(|| {
                    GenerationError::ExtensionTracegen(format!(
                        "AIR {air_name} has no postflight trace generator"
                    ))
                })?;
                exec_ctxs[chain_pos] = Some(
                    generator(chip.as_any(), postflight)
                        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?,
                );
            }
        }

        let ctx_without_empties = sys_ctxs
            .into_iter()
            .chain(exec_ctxs.into_iter().map(|ctx| ctx.unwrap()))
            .enumerate()
            .filter(|(_air_id, ctx)| ctx.common_main.height() > 0)
            .collect();
        Ok(ProvingContext::new(ctx_without_empties))
    }
}

#[cfg(feature = "cuda")]
impl<SC> VmChipComplex<SC, GpuBackend, SystemChipInventoryGPU>
where
    SC: StarkProtocolConfig,
{
    /// Generates a complete GPU proving context from one preflight segment.
    pub(crate) fn generate_proving_ctx_from_postflight(
        &mut self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
        mut generate_extension: impl FnMut(
            &dyn Any,
        )
            -> Result<AirProvingContext<GpuBackend>, GenerationError>,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError> {
        let num_ext_airs = self.inventory.chips.len();
        let air_names = self
            .inventory
            .airs
            .ext_airs
            .iter()
            .map(|air| air.name().to_string())
            .collect::<Vec<_>>();
        let mut exec_ctxs = Vec::new();
        exec_ctxs.resize_with(num_ext_airs, || None);

        // System connector and Merkle requests must be generated first.
        // Extension chips then run in reverse insertion order so shared
        // periphery chips are generated after their consumers.
        let sys_ctxs = {
            let _span = info_span!("system_trace_gen").entered();
            self.system
                .generate_proving_ctx_from_postflight(program, transcript, replay_plan)
                .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?
        };
        debug_assert_eq!(sys_ctxs.len(), self.system_config().num_airs());
        {
            let _span = info_span!("executor_trace_gen").entered();
            for (chain_pos, (insertion_idx, chip)) in
                self.inventory.chips.iter().enumerate().rev().enumerate()
            {
                let _air_span =
                    info_span!("single_trace_gen", air = air_names[insertion_idx]).entered();
                exec_ctxs[chain_pos] = Some(generate_extension(chip.as_any()).map_err(
                    |error| match error {
                        GenerationError::ExtensionTracegen(message) => {
                            GenerationError::ExtensionTracegen(format!(
                                "AIR `{}`: {message}",
                                air_names[insertion_idx]
                            ))
                        }
                        error => error,
                    },
                )?);
            }
        }
        let ctx_without_empties = sys_ctxs
            .into_iter()
            .chain(exec_ctxs.into_iter().map(|ctx| ctx.unwrap()))
            .enumerate()
            .filter(|(_air_id, ctx)| ctx.common_main.height() > 0)
            .collect();
        Ok(ProvingContext::new(ctx_without_empties))
    }
}

// ============ Blanket implementation of VM extension traits for Option<E> ===========

impl<EXT: VmExecutionExtension> VmExecutionExtension for Option<EXT> {
    type Executor = EXT::Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<Self::Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        if let Some(extension) = self {
            extension.extend_execution(inventory)
        } else {
            Ok(())
        }
    }
}

impl<SC: StarkProtocolConfig, EXT: VmCircuitExtension<SC>> VmCircuitExtension<SC> for Option<EXT> {
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
    use crate::arch::VmCircuitConfig;

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
        assert_eq!(system.memory.interface.boundary.merkle_bus.index, 4);
        assert_eq!(system.memory.interface.boundary.compression_bus.index, 5);
    }
}
