use std::{cell::RefCell, rc::Rc, sync::Arc};

use enum_dispatch::enum_dispatch;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::rap::AnyRap;

use crate::{
    arch::columns::ExecutionState,
    cpu::{CpuChip, trace::Instruction},
    field_arithmetic::FieldArithmeticChip,
    field_extension::FieldExtensionArithmeticChip,
    memory::offline_checker::MemoryChip,
    poseidon2::Poseidon2Chip,
    program::ProgramChip,
};

#[enum_dispatch]
pub trait OpCodeExecutor<F> {
    fn execute(
        &mut self,
        instruction: &Instruction<F>,
        prev_state: ExecutionState<usize>,
    ) -> ExecutionState<usize>;
}

pub trait MachineChip<F> {
    fn generate_trace(&mut self) -> RowMajorMatrix<F>;
    fn air<'a, SC: StarkGenericConfig>(&'a self) -> &'a dyn AnyRap<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>;
    fn get_public_values(&mut self) -> Vec<F> {
        vec![]
    }
    fn current_trace_height(&self) -> usize;
    fn current_trace_cells(&self) -> usize {
        self.current_trace_height() * self.air().width()
    }
}

impl<F, C: OpCodeExecutor<F>> OpCodeExecutor<F> for Rc<RefCell<C>> {
    fn execute(
        &mut self,
        instruction: &Instruction<F>,
        prev_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        self.borrow_mut().execute(instruction, prev_state)
    }
}

#[enum_dispatch(OpCodeExecutor<F>)]
pub enum OpCodeExecutorVariant<F: PrimeField32> {
    FieldArithmetic(Rc<RefCell<FieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionArithmeticChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<16, F>>>),
}

pub enum MachineChipVariant<F: PrimeField32> {
    Cpu(Rc<RefCell<CpuChip<1, F>>>),
    Program(Rc<RefCell<ProgramChip<F>>>),
    Memory(Rc<RefCell<MemoryChip<1, F>>>),
    FieldArithmetic(Rc<RefCell<FieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionArithmeticChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<16, F>>>),
    RangeChecker(Arc<RangeCheckerGateChip>),
}

impl<F: PrimeField32> MachineChip<F> for MachineChipVariant<F> {
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        match self {
            MachineChipVariant::Cpu(chip) => chip.borrow_mut().generate_trace(),
            MachineChipVariant::Program(chip) => chip.borrow_mut().generate_trace(),
            MachineChipVariant::Memory(chip) => chip.borrow_mut().generate_trace(),
            MachineChipVariant::FieldArithmetic(chip) => chip.borrow_mut().generate_trace(),
            MachineChipVariant::FieldExtension(chip) => chip.borrow_mut().generate_trace(),
            MachineChipVariant::Poseidon2(chip) => chip.borrow_mut().generate_trace(),
            MachineChipVariant::RangeChecker(chip) => chip.generate_trace(),
        }
    }

    fn air<SC: StarkGenericConfig>(&self) -> &dyn AnyRap<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        match self {
            MachineChipVariant::Cpu(chip) => chip.borrow().air(),
            MachineChipVariant::Program(chip) => chip.borrow().air(),
            MachineChipVariant::Memory(chip) => chip.borrow().air(),
            MachineChipVariant::FieldArithmetic(chip) => chip.borrow().air(),
            MachineChipVariant::FieldExtension(chip) => chip.borrow().air(),
            MachineChipVariant::Poseidon2(chip) => chip.borrow().air(),
            MachineChipVariant::RangeChecker(chip) => &chip.air,
        }
    }

    fn get_public_values(&mut self) -> Vec<F> {
        match self {
            MachineChipVariant::Cpu(chip) => chip.borrow_mut().get_public_values(),
            MachineChipVariant::Program(chip) => chip.borrow_mut().get_public_values(),
            MachineChipVariant::Memory(chip) => chip.borrow_mut().get_public_values(),
            MachineChipVariant::FieldArithmetic(chip) => chip.borrow_mut().get_public_values(),
            MachineChipVariant::FieldExtension(chip) => chip.borrow_mut().get_public_values(),
            MachineChipVariant::Poseidon2(chip) => chip.borrow_mut().get_public_values(),
            MachineChipVariant::RangeChecker(_) => vec![],
        }
    }

    fn current_trace_height(&self) -> usize {
        match self {
            MachineChipVariant::Cpu(chip) => chip.borrow().current_trace_height(),
            MachineChipVariant::Program(chip) => chip.borrow().current_trace_height(),
            MachineChipVariant::Memory(chip) => chip.borrow().current_trace_height(),
            MachineChipVariant::FieldArithmetic(chip) => chip.borrow().current_trace_height(),
            MachineChipVariant::FieldExtension(chip) => chip.borrow().current_trace_height(),
            MachineChipVariant::Poseidon2(chip) => chip.borrow().current_trace_height(),
            MachineChipVariant::RangeChecker(chip) => chip.count.len(),
        }
    }
}
