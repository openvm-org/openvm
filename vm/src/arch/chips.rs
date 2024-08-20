use std::{cell::RefCell, rc::Rc};

use enum_dispatch::enum_dispatch;
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

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

#[enum_dispatch]
pub trait MachineChip<F> {
    fn generate_trace(&mut self) -> RowMajorMatrix<F>;
    fn air<SC: StarkGenericConfig>(&self) -> &dyn AnyRap<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>;
    fn get_public_values(&mut self) -> Vec<F> {
        vec![]
    }
}

impl<F: Field, C: OpCodeExecutor<F>> OpCodeExecutor<F> for Rc<RefCell<C>> {
    fn execute(
        &mut self,
        instruction: &Instruction<F>,
        prev_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        self.borrow_mut().execute(instruction, prev_state)
    }
}

impl<F: Field, C: MachineChip<F>> MachineChip<F> for Rc<RefCell<C>> {
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        self.borrow_mut().generate_trace()
    }

    fn air<SC: StarkGenericConfig>(&self) -> &dyn AnyRap<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        self.borrow().air()
    }

    fn get_public_values(&mut self) -> Vec<F> {
        todo!()
    }
}

#[enum_dispatch(OpCodeExecutor<F>)]
pub enum OpCodeExecutorVariant<F: PrimeField32> {
    FieldArithmetic(FieldArithmeticChip<F>),
    FieldExtension(FieldExtensionArithmeticChip<F>),
    Poseidon2(Poseidon2Chip<16, F>),
}

#[enum_dispatch(MachineChip<F>)]
pub enum MachineChipVariant<F: PrimeField32> {
    Cpu(CpuChip<1, F>),
    Program(ProgramChip<F>),
    Memory(Rc<RefCell<MemoryChip<1, F>>>),
    //Memory(MemoryChip<1, F>),
    FieldArithmetic(FieldArithmeticChip<F>),
    FieldExtension(FieldExtensionArithmeticChip<F>),
    Poseidon2(Poseidon2Chip<16, F>),
}
