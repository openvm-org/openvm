use std::{any::Any, cell::RefCell, rc::Rc};

use p3_field::PrimeField32;
use rustc_hash::FxHashMap;

use super::{ExecutionBus, InstructionExecutor};
use crate::{
    kernels::public_values::PublicValuesChip,
    system::{
        connector::VmConnectorChip,
        memory::{merkle::MemoryMerkleBus, offline_checker::MemoryBus, MemoryControllerRef},
        program::{ProgramBus, ProgramChip},
    },
};

const EXECUTION_BUS: ExecutionBus = ExecutionBus(0);
const MEMORY_BUS: MemoryBus = MemoryBus(1);
const PROGRAM_BUS: ProgramBus = ProgramBus(2);
const RANGE_CHECKER_BUS: usize = 3;
const MEMORY_MERKLE_BUS: MemoryMerkleBus = MemoryMerkleBus(4);
const POSEIDON2_DIRECT_BUS: usize = 6;

// PublicValuesChip needs F: PrimeField32 due to Adapter
pub struct SystemChipset<F: PrimeField32> {
    // ATTENTION: chip destruction should follow the following field order:
    pub program_chip: ProgramChip<F>,
    pub connector_chip: VmConnectorChip<F>,
    /// PublicValuesChip is disabled when num_public_values == 0.
    pub public_values_chip: Option<Rc<RefCell<PublicValuesChip<F>>>>,
    pub memory_controller: MemoryControllerRef<F>,
}

pub struct ChipsetBuilder<F: PrimeField32> {
    system: SystemChipset<F>,
    /// Bus indices are in range [0, bus_idx_max)
    bus_idx_max: usize,
    /// Chips that are already included in the chipset and may be used
    /// as dependencies. The order should be that depended-on chips are ordered
    /// **before** their dependents.
    chips: Vec<Box<dyn AnyEnum>>,
}

impl<F: PrimeField32> ChipsetBuilder<F> {
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

pub trait ChipsetConfig<F: PrimeField32> {
    /// This is expected to be an enum to dispatch [`InstructionExecutor`]s.
    type Executor: InstructionExecutor<F>;
    /// Should implement `Chip<SC>` but we don't impose a trait bound to avoid the generic `StarkGenericConfig`.
    /// This is expected to be an enum of chip types.
    type Chip;

    fn create_chipset(
        &self,
        builder: &mut ChipsetBuilder<F>,
    ) -> Chipset<Self::Executor, Self::Chip>;
}

pub struct Chipset<E, C> {
    /// TODO: usize -> AxVmOpcode(usize)
    pub executors: FxHashMap<usize, E>,
    pub chips: Vec<C>,
}

/// A helper trait for downcasting types that may be enums.
pub trait AnyEnum {
    /// Recursively "unwraps" enum and casts to `Any` for downcasting.
    fn as_any_kind(&self) -> &dyn Any;
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
