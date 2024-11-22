use std::{any::Any, cell::RefCell, rc::Rc, sync::Arc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
use ax_poseidon2_air::poseidon2::air::SBOX_DEGREE;
use ax_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    prover::types::{AirProofInput, CommittedTraceData, ProofInput},
    Chip, ChipUsageGetter,
};
use axvm_circuit_derive::AnyEnum;
use axvm_instructions::{
    program::Program, Poseidon2Opcode, PublishOpcode, SystemOpcode, UsizeOpcode,
};
use derive_more::derive::From;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::Matrix;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use super::{vm_poseidon2_config, ExecutionBus, InstructionExecutor, Streams, SystemConfig};
use crate::{
    intrinsics::hashes::poseidon2::Poseidon2Chip,
    kernels::{
        adapters::native_adapter::NativeAdapterChip,
        public_values::{core::PublicValuesCoreChip, PublicValuesChip},
    },
    system::{
        connector::VmConnectorChip,
        memory::{
            merkle::MemoryMerkleBus, offline_checker::MemoryBus, Equipartition, MemoryController,
            MemoryControllerRef, CHUNK, MERKLE_AIR_OFFSET,
        },
        phantom::PhantomChip,
        program::{ProgramBus, ProgramChip},
    },
};

// TODO: Make these public after chip_set.rs is removed
const PROGRAM_AIR_ID: usize = 0;
/// ProgramAir is the first AIR so its cached trace should be the first main trace.
const PROGRAM_CACHED_TRACE_INDEX: usize = 0;
const CONNECTOR_AIR_ID: usize = 1;
/// If PublicValuesAir is **enabled**, its AIR ID is 2. PublicValuesAir is always disabled when
/// continuations is enabled.
const PUBLIC_VALUES_AIR_ID: usize = 2;
/// If VM has continuations enabled, all AIRs of MemoryController are added after ConnectorChip.
/// Merkle AIR commits start/final memory states.
const MERKLE_AIR_ID: usize = CONNECTOR_AIR_ID + 1 + MERKLE_AIR_OFFSET;

pub const EXECUTION_BUS: ExecutionBus = ExecutionBus(0);
pub const MEMORY_BUS: MemoryBus = MemoryBus(1);
pub const PROGRAM_BUS: ProgramBus = ProgramBus(2);
pub const RANGE_CHECKER_BUS: usize = 3;

/// Builder for processing unit. Processing units extend an existing system unit.
pub struct VmExtensionBuilder<'a, F: PrimeField32> {
    system: &'a SystemBase<F>,
    streams: &'a Arc<Mutex<Streams<F>>>,
    /// Bus indices are in range [0, bus_idx_max)
    bus_idx_max: usize,
    /// Chips that are already included in the chipset and may be used
    /// as dependencies. The order should be that depended-on chips are ordered
    /// **before** their dependents.
    chips: Vec<&'a dyn AnyEnum>,
}

impl<'a, F: PrimeField32> VmExtensionBuilder<'a, F> {
    pub fn new(
        system: &'a SystemBase<F>,
        streams: &'a Arc<Mutex<Streams<F>>>,
        bus_idx_max: usize,
    ) -> Self {
        Self {
            system,
            streams,
            bus_idx_max,
            chips: Vec::new(),
        }
    }

    pub fn memory_controller(&self) -> &MemoryControllerRef<F> {
        &self.system.memory_controller
    }

    pub fn system_base(&self) -> &SystemBase<F> {
        self.system
    }

    pub fn new_bus(&mut self) -> usize {
        let idx = self.bus_idx_max;
        self.bus_idx_max += 1;
        idx
    }

    /// Looks through built chips to see if there exists any of type `C` by downcasting.
    /// Returns all chips of type `C` in the chipset.
    ///
    /// Note: the type `C` will usually be a smart pointer to a chip.
    pub fn find_chip<C: 'static>(&self) -> Vec<&C> {
        self.chips
            .iter()
            .filter_map(|c| c.as_any_kind().downcast_ref())
            .collect()
    }

    /// Shareable streams. Clone to get a shared mutable reference.
    pub fn streams(&self) -> &Arc<Mutex<Streams<F>>> {
        self.streams
    }

    fn add_chip<E: AnyEnum>(&mut self, chip: &'a E) {
        self.chips.push(chip);
    }
}

/// Configuration for a processor extension.
///
/// There are two associated types:
/// - `Executor`: enum for chips that are [`InstructionExecutor`]s.
/// -
pub trait VmExtension<F: PrimeField32> {
    /// Enum of chips that implement [`InstructionExecutor`] for instruction execution.
    /// `Executor` **must** implement `Chip<SC>` but the trait bound is omitted to omit the
    /// `StarkGenericConfig` generic parameter.
    type Executor: InstructionExecutor<F> + AnyEnum;
    /// Enum of periphery chips that do not implement [`InstructionExecutor`].
    /// `Periphery` **must** implement `Chip<SC>` but the trait bound is omitted to omit the
    /// `StarkGenericConfig` generic parameter.
    type Periphery: AnyEnum;

    fn build(
        &self,
        builder: &mut VmExtensionBuilder<F>,
    ) -> VmInventory<Self::Executor, Self::Periphery>;
}

#[derive(Clone, Debug)]
pub struct VmInventory<E, P> {
    /// Lookup table to executor ID. We store executors separately due to mutable borrow issues.
    instruction_lookup: FxHashMap<AxVmOpcode, ExecutorId>,
    executors: Vec<E>,
    periphery: Vec<P>,
    /// Order of insertion. The reverse of this will be the order the chips are destroyed
    /// to generate trace.
    insertion_order: Vec<ChipId>,
}

type ExecutorId = usize;
/// TODO: create newtype
type AxVmOpcode = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChipId {
    Executor(usize),
    Periphery(usize),
}

impl<E, P> Default for VmInventory<E, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E, P> VmInventory<E, P> {
    pub fn new() -> Self {
        Self {
            instruction_lookup: FxHashMap::default(),
            executors: Vec::new(),
            periphery: Vec::new(),
            insertion_order: Vec::new(),
        }
    }

    pub fn transmute<E2, P2>(self) -> VmInventory<E2, P2>
    where
        E: Into<E2>,
        P: Into<P2>,
    {
        VmInventory {
            instruction_lookup: self.instruction_lookup,
            executors: self.executors.into_iter().map(|e| e.into()).collect(),
            periphery: self.periphery.into_iter().map(|p| p.into()).collect(),
            insertion_order: self.insertion_order,
        }
    }

    /// Append `other` to current inventory. This means `self` comes earlier in the dependency chain.
    pub fn append(&mut self, mut other: VmInventory<E, P>) {
        let num_executors = self.executors.len();
        let num_periphery = self.periphery.len();
        for (_, id) in other.instruction_lookup.iter_mut() {
            *id += num_executors;
        }
        for chip_id in other.insertion_order.iter_mut() {
            match chip_id {
                ChipId::Executor(id) => *id += num_executors,
                ChipId::Periphery(id) => *id += num_periphery,
            }
        }
        self.executors.append(&mut other.executors);
        self.periphery.append(&mut other.periphery);
        self.insertion_order.append(&mut other.insertion_order);
    }

    /// Inserts an executor with the collection of opcodes that it handles.
    /// If some executor already owns one of the opcodes, it will be replaced and the old
    /// executor ID is returned.
    #[must_use]
    pub fn add_executor(
        &mut self,
        executor: E,
        opcodes: impl IntoIterator<Item = AxVmOpcode>,
    ) -> Option<ExecutorId> {
        let id = self.executors.len();
        self.executors.push(executor);
        self.insertion_order.push(ChipId::Executor(id));
        for opcode in opcodes {
            if let Some(old_id) = self.instruction_lookup.insert(opcode, id) {
                return Some(old_id);
            }
        }
        None
    }

    pub fn add_periphery_chip(&mut self, periphery_chip: P) {
        let id = self.periphery.len();
        self.periphery.push(periphery_chip);
        self.insertion_order.push(ChipId::Periphery(id));
    }

    pub fn get_executor(&self, opcode: AxVmOpcode) -> Option<&E> {
        let id = self.instruction_lookup.get(&opcode)?;
        self.executors.get(*id)
    }

    pub fn get_mut_executor(&mut self, opcode: AxVmOpcode) -> Option<&mut E> {
        let id = self.instruction_lookup.get(&opcode)?;
        self.executors.get_mut(*id)
    }

    pub fn executors(&self) -> &[E] {
        &self.executors
    }

    pub fn periphery(&self) -> &[P] {
        &self.periphery
    }

    pub fn num_airs(&self) -> usize {
        self.executors.len() + self.periphery.len()
    }
}

// PublicValuesChip needs F: PrimeField32 due to Adapter
/// The minimum collection of chips that any VM must have.
pub struct VmChipComplex<F: PrimeField32, E, P> {
    pub config: SystemConfig,
    // ATTENTION: chip destruction should follow the **reverse** of the following field order:
    pub base: SystemBase<F>,
    /// Extendable collection of chips for executing instructions.
    /// System ensures it contains:
    /// - PhantomChip
    /// - PublicValuesChip if continuations disabled
    /// - Poseidon2Chip if continuations enabled
    pub inventory: VmInventory<E, P>,

    streams: Arc<Mutex<Streams<F>>>,
    /// System buses use indices [0, bus_idx_max)
    bus_idx_max: usize,
}

/// The base [VmChipComplex] with only system chips.
pub type SystemComplex<F> = VmChipComplex<F, SystemExecutor<F>, SystemPeriphery<F>>;

/// Base system chips.
/// The following don't execute instructions, but are essential
/// for the VM architecture.
pub struct SystemBase<F> {
    // RangeCheckerChip **must** be the last chip to have trace generation called on
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
    pub memory_controller: MemoryControllerRef<F>,
    pub connector_chip: VmConnectorChip<F>,
    pub program_chip: ProgramChip<F>,
}

#[derive(ChipUsageGetter, Chip, AnyEnum, From)]
pub enum SystemExecutor<F: PrimeField32> {
    PublicValues(PublicValuesChip<F>),
    Phantom(PhantomChip<F>),
}

#[derive(ChipUsageGetter, Chip, AnyEnum, From)]
pub enum SystemPeriphery<F: PrimeField32> {
    /// Poseidon2 chip with direct compression interactions
    Poseidon2(Poseidon2Chip<F>),
}

impl<F: PrimeField32> SystemComplex<F> {
    pub fn new(config: SystemConfig) -> Self {
        let range_bus =
            VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, config.memory_config.decomp);
        let mut bus_idx_max = RANGE_CHECKER_BUS;

        let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));
        let memory_controller = if config.continuation_enabled {
            bus_idx_max += 1;
            Rc::new(RefCell::new(MemoryController::with_persistent_memory(
                MEMORY_BUS,
                config.memory_config,
                range_checker.clone(),
                MemoryMerkleBus(bus_idx_max - 1),
                Equipartition::<F, CHUNK>::new(),
            )))
        } else {
            Rc::new(RefCell::new(MemoryController::with_volatile_memory(
                MEMORY_BUS,
                config.memory_config,
                range_checker.clone(),
            )))
        };
        let program_chip = ProgramChip::default();
        let connector_chip = VmConnectorChip::new(EXECUTION_BUS, PROGRAM_BUS);

        let mut inventory = VmInventory::new();
        // PublicValuesChip is required when num_public_values > 0 in single segment mode.
        if !config.continuation_enabled && config.num_public_values > 0 {
            assert_eq!(inventory.executors().len(), Self::PV_EXECUTOR_IDX);
            let chip = PublicValuesChip::new(
                NativeAdapterChip::new(EXECUTION_BUS, PROGRAM_BUS, memory_controller.clone()),
                PublicValuesCoreChip::new(
                    config.num_public_values,
                    PublishOpcode::default_offset(),
                    config.max_constraint_degree as u32,
                ),
                memory_controller.clone(),
            );
            inventory
                .add_executor(chip.into(), [PublishOpcode::default_offset()])
                .unwrap();
        }
        if config.continuation_enabled {
            assert_eq!(inventory.periphery().len(), Self::POSEIDON2_PERIPHERY_IDX);
            // Add direct poseidon2 chip for persistent memory.
            // This is **not** an instruction executor.
            // Currently we never use poseidon2 opcodes when continuations is enabled: we will need
            // special handling when that happens
            let direct_bus_idx = bus_idx_max;
            bus_idx_max += 1;
            let chip = Poseidon2Chip::from_poseidon2_config(
                vm_poseidon2_config(),
                config.max_constraint_degree.min(SBOX_DEGREE),
                EXECUTION_BUS,
                PROGRAM_BUS,
                memory_controller.clone(),
                direct_bus_idx,
                Poseidon2Opcode::default_offset(),
            );
            inventory.add_periphery_chip(chip.into());
        }
        let streams = Arc::new(Mutex::new(Streams::default()));
        let phantom_opcode = SystemOpcode::PHANTOM.with_default_offset();
        let mut phantom_chip = PhantomChip::new(
            EXECUTION_BUS,
            PROGRAM_BUS,
            memory_controller.clone(),
            phantom_opcode,
        );
        phantom_chip.set_streams(streams.clone());
        inventory
            .add_executor(phantom_chip.into(), [phantom_opcode])
            .unwrap();

        let base = SystemBase {
            program_chip,
            connector_chip,
            memory_controller,
            range_checker_chip: range_checker,
        };

        Self {
            config,
            base,
            inventory,
            bus_idx_max,
            streams,
        }
    }
}

impl<F: PrimeField32, E, P> VmChipComplex<F, E, P> {
    /// **If** public values chip exists, then its executor index is 0.
    const PV_EXECUTOR_IDX: ExecutorId = 0;
    /// **If** internal poseidon2 chip exists, then its periphery index is 0.
    const POSEIDON2_PERIPHERY_IDX: usize = 0;

    // @dev: Remember to update self.bus_idx_max after dropping this!
    pub fn extension_builder(&self) -> VmExtensionBuilder<F>
    where
        E: AnyEnum,
        P: AnyEnum,
    {
        let mut builder = VmExtensionBuilder::new(&self.base, &self.streams, self.bus_idx_max);
        // Add range checker for convenience, the other system base chips aren't included - they can be accessed directly from builder
        builder.add_chip(&self.base.range_checker_chip);
        for chip in self.inventory.executors() {
            builder.add_chip(chip);
        }
        for chip in self.inventory.periphery() {
            builder.add_chip(chip);
        }

        builder
    }

    /// Extend the chip complex with a new extension.
    /// A new chip complex with different type generics is returned with the combined inventory.
    pub fn extend<E3, P3, Ext>(mut self, config: &Ext) -> VmChipComplex<F, E3, P3>
    where
        Ext: VmExtension<F>,
        E: Into<E3> + AnyEnum,
        P: Into<P3> + AnyEnum,
        Ext::Executor: Into<E3>,
        Ext::Periphery: Into<P3>,
    {
        let mut builder = self.extension_builder();
        let inventory_ext = config.build(&mut builder);
        self.bus_idx_max = builder.bus_idx_max;
        let mut ext_complex = self.transmute();
        ext_complex.append(inventory_ext.transmute());
        ext_complex
    }

    pub fn transmute<E2, P2>(self) -> VmChipComplex<F, E2, P2>
    where
        E: Into<E2>,
        P: Into<P2>,
    {
        VmChipComplex {
            config: self.config,
            base: self.base,
            inventory: self.inventory.transmute(),
            bus_idx_max: self.bus_idx_max,
            streams: self.streams,
        }
    }

    /// Appends `other` to the current inventory.
    /// This means `self` comes earlier in the dependency chain.
    pub fn append(&mut self, other: VmInventory<E, P>) {
        self.inventory.append(other);
    }

    pub fn num_airs(&self) -> usize {
        3 + self.base.memory_controller.borrow().num_airs() + self.inventory.num_airs()
    }

    pub fn public_values_chip(&self) -> Option<&PublicValuesChip<F>>
    where
        E: AnyEnum,
    {
        let chip = self.inventory.executors().get(Self::PV_EXECUTOR_IDX)?;
        chip.as_any_kind().downcast_ref()
    }

    pub fn poseidon2_chip(&self) -> Option<&Poseidon2Chip<F>>
    where
        P: AnyEnum,
    {
        let chip = self
            .inventory
            .periphery()
            .get(Self::POSEIDON2_PERIPHERY_IDX)?;
        chip.as_any_kind().downcast_ref()
    }

    pub(crate) fn set_program(&mut self, program: Program<F>) {
        self.base.program_chip.set_program(program);
    }

    pub(crate) fn generate_proof_input<SC: StarkGenericConfig>(
        mut self,
        cached_program: Option<CommittedTraceData<SC>>,
    ) -> ProofInput<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
        E: Chip<SC> + AnyEnum,
        P: Chip<SC> + AnyEnum,
    {
        let has_pv_chip = self.public_values_chip().is_some();
        // ATTENTION: The order of AIR proof input generation MUST be consistent with `airs`.
        let mut builder = VmProofInputBuilder::new();
        let SystemBase {
            range_checker_chip,
            memory_controller,
            connector_chip,
            program_chip,
        } = self.base;
        // System: Program Chip
        debug_assert_eq!(builder.curr_air_id, PROGRAM_AIR_ID);
        builder.add_air_proof_input(program_chip.generate_air_proof_input(cached_program));
        // System: Connector Chip
        debug_assert_eq!(builder.curr_air_id, CONNECTOR_AIR_ID);
        builder.add_air_proof_input(connector_chip.generate_air_proof_input());

        // Go through all chips in inventory in reverse order they were added (to resolve dependencies)
        // Important Note: for air_id ordering reasons, we want to generate_air_proof_input for
        // public values and memory chips **last** but include them into the `builder` **first**.
        let mut public_values_input = None;
        let mut insertion_order = self.inventory.insertion_order;
        insertion_order.reverse();
        let mut non_sys_inputs = Vec::with_capacity(insertion_order.len());
        for chip_id in insertion_order {
            let height = None;
            // let height = self.overridden_executor_heights.as_ref().and_then(
            //     |overridden_heights| {
            //         let executor_name: ExecutorName = (&executor).into();
            //         overridden_heights.get(&executor_name).copied()
            //     },
            // );
            let air_proof_input = match chip_id {
                ChipId::Executor(id) => {
                    let chip = self.inventory.executors.pop().unwrap();
                    assert_eq!(id, self.inventory.executors.len());
                    generate_air_proof_input(chip, height)
                }
                ChipId::Periphery(id) => {
                    let chip = self.inventory.periphery.pop().unwrap();
                    assert_eq!(id, self.inventory.periphery.len());
                    generate_air_proof_input(chip, height)
                }
            };
            if has_pv_chip && chip_id == ChipId::Executor(Self::PV_EXECUTOR_IDX) {
                public_values_input = Some(air_proof_input);
            } else {
                non_sys_inputs.push(air_proof_input);
            }
        }

        if let Some(input) = public_values_input {
            debug_assert_eq!(builder.curr_air_id, PUBLIC_VALUES_AIR_ID);
            builder.add_air_proof_input(input);
        }
        // System: Memory Controller
        {
            // memory
            let memory_controller = Rc::try_unwrap(memory_controller)
                .expect("other chips still hold a reference to memory chip")
                .into_inner();

            let air_proof_inputs = memory_controller.generate_air_proof_inputs();
            for air_proof_input in air_proof_inputs {
                builder.add_air_proof_input(air_proof_input);
            }
        }
        // Non-system chips
        non_sys_inputs
            .into_iter()
            .for_each(|input| builder.add_air_proof_input(input));
        // System: Range Checker Chip
        builder.add_air_proof_input(range_checker_chip.generate_air_proof_input());

        builder.build()
    }
}

struct VmProofInputBuilder<SC: StarkGenericConfig> {
    curr_air_id: usize,
    proof_input_per_air: Vec<(usize, AirProofInput<SC>)>,
}

impl<SC: StarkGenericConfig> VmProofInputBuilder<SC> {
    fn new() -> Self {
        Self {
            curr_air_id: 0,
            proof_input_per_air: vec![],
        }
    }
    /// Adds air proof input if one of the main trace matrices is non-empty.
    /// Always increments the internal `curr_air_id` regardless of whether a new air proof input was added or not.
    fn add_air_proof_input(&mut self, air_proof_input: AirProofInput<SC>) {
        let h = if !air_proof_input.raw.cached_mains.is_empty() {
            air_proof_input.raw.cached_mains[0].height()
        } else {
            air_proof_input
                .raw
                .common_main
                .as_ref()
                .map(|trace| trace.height())
                .unwrap()
        };
        if h > 0 {
            self.proof_input_per_air
                .push((self.curr_air_id, air_proof_input));
        }
        self.curr_air_id += 1;
    }

    fn build(self) -> ProofInput<SC> {
        ProofInput {
            per_air: self.proof_input_per_air,
        }
    }
}

/// Generates an AIR proof input of the chip with the given height, if any.
///
/// Assumption: an all-0 row is a valid dummy row for `chip`.
pub fn generate_air_proof_input<SC: StarkGenericConfig, C: Chip<SC>>(
    chip: C,
    height: Option<usize>,
) -> AirProofInput<SC> {
    let mut proof_input = chip.generate_air_proof_input();
    if let Some(height) = height {
        let height = height.next_power_of_two();
        let main = proof_input.raw.common_main.as_mut().unwrap();
        assert!(
            height >= main.height(),
            "Overridden height must be greater than or equal to the used height"
        );
        main.pad_to_height(height, AbstractField::ZERO);
    }
    proof_input
}

/// A helper trait for downcasting types that may be enums.
pub trait AnyEnum {
    /// Recursively "unwraps" enum and casts to `Any` for downcasting.
    fn as_any_kind(&self) -> &dyn Any;
}

impl AnyEnum for () {
    fn as_any_kind(&self) -> &dyn Any {
        self
    }
}

impl AnyEnum for Arc<VariableRangeCheckerChip> {
    fn as_any_kind(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    impl AnyEnum for EnumB {
        fn as_any_kind(&self) -> &dyn Any {
            match self {
                EnumB::C(c) => c,
                EnumB::D(d) => d.as_any_kind(),
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
}
