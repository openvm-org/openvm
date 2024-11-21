use std::{any::Any, cell::RefCell, rc::Rc, sync::Arc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
use axvm_instructions::{
    instruction::Instruction, program::Program, PublishOpcode, SystemOpcode, UsizeOpcode,
};
use enum_dispatch::enum_dispatch;
use p3_field::PrimeField32;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use strum::EnumDiscriminants;

use super::{
    ExecutionBus, ExecutionState, InstructionExecutor, Streams, SystemConfig, MEMORY_MERKLE_BUS,
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
pub struct ProcessingBuilder<'a, F: PrimeField32> {
    system: &'a SystemUnit<F>,
    /// Bus indices are in range [0, bus_idx_max)
    bus_idx_max: usize,
    /// Chips that are already included in the chipset and may be used
    /// as dependencies. The order should be that depended-on chips are ordered
    /// **before** their dependents.
    chips: Vec<&'a dyn AnyEnum>,
}

impl<'a, F: PrimeField32> ProcessingBuilder<'a, F> {
    pub fn memory_controller(&self) -> &MemoryControllerRef<F> {
        &self.system.memory_controller
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
}

/// Configuration for a processor extension.
///
/// There are two associated types:
/// - `Executor`: enum for chips that are [`InstructionExecutor`]s.
/// -
pub trait ProcessingConfig<F: PrimeField32> {
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
        builder: &mut ProcessingBuilder<F>,
    ) -> ProcessingUnit<Self::Executor, Self::Periphery>;
}

#[derive(Clone, Debug)]
pub struct ProcessingUnit<E, P> {
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

impl<E, P> Default for ProcessingUnit<E, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E, P> ProcessingUnit<E, P> {
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
pub struct SystemUnit<F: PrimeField32> {
    // ATTENTION: chip destruction should follow the following field order:
    /// Contains:
    /// - PhantomChip
    /// - PublicValuesChip if continuations disabled
    /// - Poseidon2Chip if continuations enabled
    pub processing: ProcessingUnit<SystemExecutor<F>, ()>,
    // The following don't execute instructions, they are the backbone of the system
    pub program_chip: ProgramChip<F>,
    pub connector_chip: VmConnectorChip<F>,
    /// PublicValuesChip is disabled when num_public_values == 0.
    pub memory_controller: MemoryControllerRef<F>,
    // RangeCheckerChip **must** be the last chip to have trace generation called on
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
}

#[derive(ChipUsageGetter, Chip)]
#[enum_dispatch(InstructionExecutor<F>)]
pub enum SystemExecutor<F: PrimeField32> {
    Phantom(PhantomChip<F>),
    PublicValues(PublicValuesChip<F>),
    // The poseidon2 with direct compression interactions
    // TODO: make this periphery after it is separated from hasher chip
    Poseidon2(Poseidon2Chip<F>),
}

impl<F: PrimeField32> SystemUnit<F> {
    /// **If** public values chip exists, then its internal index is 1.
    const PV_CHIP_IDX: ExecutorId = 1;

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

        let mut processing = ProcessingUnit::new();

        let phantom_opcode = SystemOpcode::PHANTOM.with_default_offset();
        let phantom_chip = PhantomChip::new(
            EXECUTION_BUS,
            PROGRAM_BUS,
            memory_controller.clone(),
            phantom_opcode,
        );
        processing
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
            processing
                .add_executor(chip.into(), [PublishOpcode::default_offset()])
                .unwrap();
        }
        // TODO: need to handle Poseidon2

        Self {
            processing,
            program_chip,
            connector_chip,
            memory_controller,
            range_checker_chip: range_checker,
        }
    }

    pub fn public_values_chip(&self) -> Option<&PublicValuesChip<F>> {
        let ex_chip = self.processing.executors().get(Self::PV_CHIP_IDX)?;
        match ex_chip {
            SystemExecutor::PublicValues(chip) => Some(chip),
            _ => None,
        }
    }

    pub fn num_airs(&self) -> usize {
        3 + self.memory_controller.borrow().num_airs() + self.processing.num_airs()
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
