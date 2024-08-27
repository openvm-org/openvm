use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::rap::AnyRap;
use enum_dispatch::enum_dispatch;
use p3_air::BaseAir;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use crate::{
    arch::columns::ExecutionState,
    cpu::{trace::Instruction, CpuChip},
    field_arithmetic::FieldArithmeticChip,
    field_extension::chip::FieldExtensionArithmeticChip,
    memory::manager::MemoryChipRef,
    poseidon2::Poseidon2Chip,
    program::ProgramChip,
};

#[enum_dispatch]
pub trait InstructionExecutor<F> {
    fn execute(
        &mut self,
        instruction: &Instruction<F>,
        prev_state: ExecutionState<usize>,
    ) -> ExecutionState<usize>;
}

#[enum_dispatch]
pub trait MachineChip<F> {
    fn generate_trace(&mut self) -> RowMajorMatrix<F>;
    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>;
    fn generate_public_values(&mut self) -> Vec<F> {
        vec![]
    }
    fn current_trace_height(&self) -> usize;
    fn trace_width(&self) -> usize;
    fn current_trace_cells(&self) -> usize {
        self.current_trace_height() * self.trace_width()
    }
}

impl<F, C: InstructionExecutor<F>> InstructionExecutor<F> for Rc<RefCell<C>> {
    fn execute(
        &mut self,
        instruction: &Instruction<F>,
        prev_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        self.borrow_mut().execute(instruction, prev_state)
    }
}

impl<F, C: MachineChip<F>> MachineChip<F> for Rc<RefCell<C>> {
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        self.borrow_mut().generate_trace()
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        self.borrow().air()
    }

    fn generate_public_values(&mut self) -> Vec<F> {
        self.borrow_mut().generate_public_values()
    }

    fn current_trace_height(&self) -> usize {
        self.borrow().current_trace_height()
    }

    fn trace_width(&self) -> usize {
        self.borrow().trace_width()
    }
}

#[derive(Debug)]
#[enum_dispatch(InstructionExecutor<F>)]
pub enum InstructionExecutorVariant<F: PrimeField32> {
    FieldArithmetic(Rc<RefCell<FieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionArithmeticChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<16, F>>>),
}

#[derive(Debug)]
#[enum_dispatch(MachineChip<F>)]
pub enum MachineChipVariant<F: PrimeField32> {
    Cpu(Rc<RefCell<CpuChip<F>>>),
    Program(Rc<RefCell<ProgramChip<F>>>),
    Memory(MemoryChipRef<F>),
    FieldArithmetic(Rc<RefCell<FieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionArithmeticChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<16, F>>>),
    RangeChecker(Arc<RangeCheckerGateChip>),
}

impl<F: PrimeField32> MachineChip<F> for Arc<RangeCheckerGateChip> {
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        RangeCheckerGateChip::generate_trace(self)
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.air.range_max as usize
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}
