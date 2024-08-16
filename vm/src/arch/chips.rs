use enum_dispatch::enum_dispatch;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::StarkGenericConfig;

use afs_stark_backend::rap::AnyRap;

use crate::{arch::columns::ExecutionState, cpu::trace::Instruction};

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
    fn air<SC: StarkGenericConfig>(&self) -> &dyn AnyRap<SC>;
    fn get_public_values(&mut self) -> Vec<F> {
        vec![]
    }
}

#[enum_dispatch(OpCodeExecutor<F>)]
pub enum OpCodeExecutorVariant<F> {
    A(A),
}

#[enum_dispatch(MachineChip<F>)]
pub enum MachineChipVariant<F> {
    A(A),
}
