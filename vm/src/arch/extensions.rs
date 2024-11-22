use std::{any::Any, cell::RefCell, rc::Rc, sync::Arc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
use ax_poseidon2_air::poseidon2::air::SBOX_DEGREE;
use axvm_instructions::{
    instruction::Instruction, program::Program, Poseidon2Opcode, PublishOpcode, SystemOpcode,
    UsizeOpcode,
};
use derive_more::derive::From;
use enum_dispatch::enum_dispatch;
use getset::Getters;
use p3_field::PrimeField32;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use strum::{EnumDiscriminants, IntoEnumIterator};

use super::{
    vm_poseidon2_config, ExecutionBus, ExecutionState, InstructionExecutor, Streams, SystemConfig,
    MEMORY_MERKLE_BUS, POSEIDON2_DIRECT_BUS,
};
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
            MemoryControllerRef, CHUNK,
        },
        phantom::PhantomChip,
        program::{ExecutionError, ProgramBus, ProgramChip},
    },
};

const EXECUTION_BUS: ExecutionBus = ExecutionBus(0);
const MEMORY_BUS: MemoryBus = MemoryBus(1);
const PROGRAM_BUS: ProgramBus = ProgramBus(2);
const RANGE_CHECKER_BUS: usize = 3;

/// Builder for processing unit. Processing units extend an existing system unit.
pub struct VmExtensionBuilder<'a, F: PrimeField32> {
    system: &'a SystemBase<F>,
    /// Bus indices are in range [0, bus_idx_max)
    bus_idx_max: usize,
    /// Chips that are already included in the chipset and may be used
    /// as dependencies. The order should be that depended-on chips are ordered
    /// **before** their dependents.
    chips: Vec<&'a dyn AnyEnum>,
}

impl<'a, F: PrimeField32> VmExtensionBuilder<'a, F> {
    pub fn new(system: &'a SystemBase<F>, bus_idx_max: usize) -> Self {
        Self {
            system,
            bus_idx_max,
            chips: Vec::new(),
        }
    }

    pub fn memory_controller(&self) -> &MemoryControllerRef<F> {
        &self.system.memory_controller
    }

    pub fn system_base(&self) -> &SystemBase<F> {
        &self.system
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

#[derive(Clone, Copy, Debug)]
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

#[derive(ChipUsageGetter, Chip)]
#[enum_dispatch(InstructionExecutor<F>)]
pub enum SystemExecutor<F: PrimeField32> {
    Phantom(PhantomChip<F>),
    PublicValues(PublicValuesChip<F>),
}

#[derive(ChipUsageGetter, Chip, From)]
pub enum SystemPeriphery<F: PrimeField32> {
    /// Range checker chip.
    /// **Warning**: this is not included in the inventory because it is used by all system chips.
    RangeChecker(Arc<VariableRangeCheckerChip>),
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

        let phantom_opcode = SystemOpcode::PHANTOM.with_default_offset();
        let phantom_chip = PhantomChip::new(
            EXECUTION_BUS,
            PROGRAM_BUS,
            memory_controller.clone(),
            phantom_opcode,
        );
        inventory
            .add_executor(phantom_chip.into(), [phantom_opcode])
            .unwrap();

        // PublicValuesChip is required when num_public_values > 0 in single segment mode.
        if !config.continuation_enabled && config.num_public_values > 0 {
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
        }
    }
}

impl<F, E, P> VmChipComplex<F, E, P>
where
    F: PrimeField32,
    E: AnyEnum,
    P: AnyEnum,
{
    /// **If** public values chip exists, then its executor index is 1.
    const PV_EXECUTOR_IDX: ExecutorId = 1;
    /// **If** internal poseidon2 chip exists, then its periphery index is 0.
    const POSEIDON2_PERIPHERY_IDX: usize = 0;

    // @dev: Remember to update self.bus_idx_max after dropping this!
    pub fn extension_builder(&self) -> VmExtensionBuilder<F, E, P> {
        let mut builder = VmExtensionBuilder::new(&self.base, self.bus_idx_max);
        builder.add_chip(&self.base.range_checker_chip.clone());
        for chip in self.inventory.executors() {
            builder.add_chip(chip);
        }
        for chip in self.inventory.periphery() {
            builder.add_chip(chip);
        }

        builder
    }

    pub fn num_airs(&self) -> usize {
        3 + self.memory_controller().borrow().num_airs() + self.inventory.num_airs()
    }

    pub fn public_values_chip(&self) -> Option<&PublicValuesChip<F>> {
        let chip = self.inventory.executors().get(Self::PV_EXECUTOR_IDX)?;
        chip.as_any_kind().downcast_ref()
    }

    pub fn poseidon2_chip(&self) -> Option<&Poseidon2Chip<F>> {
        let chip = self
            .inventory
            .periphery()
            .get(Self::POSEIDON2_PERIPHERY_IDX)?;
        chip.as_any_kind().downcast_ref()
    }

    pub(crate) fn set_program(&mut self, program: Program<F>) {
        self.program_chip.set_program(program);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    enum EnumA {
        A(u8),
        B(u32),
    }

    enum EnumB {
        C(u64),
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
    }
}
