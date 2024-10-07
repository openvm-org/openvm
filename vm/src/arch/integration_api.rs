use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::Matrix;

use super::{ExecutionState, InstructionExecutor, Result};
use crate::{
    memory::{MemoryChip, MemoryChipRef},
    program::Instruction,
};

/// The interface between primitive AIR and machine adapter AIR.
pub trait MachineAdapterInterface<T> {
    /// The memory read data that should be exposed for downstream use
    type Reads;
    /// The memory write data that are expected to be provided by the integrator
    type Writes;
    /// The parts of the instruction that should be exposed to the integrator.
    /// May include the `to_pc`.
    /// Typically this should not include address spaces.
    type ProcessedInstruction;
}

/// The adapter owns all memory accesses and timestamp changes.
/// The adapter AIR should also own `ExecutionBridge` and `MemoryBridge`.
pub trait MachineAdapter<F: PrimeField32> {
    type Record;
    type Air: BaseAir<F>;
    type Cols<T>;
    type Interface<T: AbstractField>: MachineAdapterInterface<T>;

    /// Given instruction, perform memory reads and return only the read data that the integrator needs to use.
    /// This is called at the start of instruction execution.
    ///
    /// The implementor may choose to store data in this struct, for example in an [Option], which will later be taken
    /// when `postprocess` is called.
    fn preprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> Result<<Self::Interface<F> as MachineAdapterInterface<F>>::Reads>;

    /// Given instruction and the data to write, perform memory writes and return the `(record, timestamp_delta)` of the full
    /// adapter record for this instruction. This **must** be called after `preprocess`.
    fn postprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        output: InstructionOutput<F, Self::Interface<F>>,
    ) -> Result<(ExecutionState<usize>, Self::Record)>;

    /// Should mutate `row_slice` to populate with values corresponding to `record`.
    fn generate_trace_row(&self, row_slice: &mut Self::Cols<F>, record: Self::Record);

    /// [Air](p3_air::Air) constraints owned by the adapter.
    /// The `interface` is given as abstract expressions so it can be directly used in other AIR constraints.
    /// Returns `
    ///
    /// Adapters should document the max constraint degree as a function of the constraint degrees of `reads, writes, instruction`.
    fn eval_adapter_constraints<
        AB: InteractionBuilder<F = F> + PairBuilder + AirBuilderWithPublicValues,
    >(
        air: &Self::Air,
        builder: &mut AB,
        local: &Self::Cols<AB::Var>,
        interface: IntegrationInterface<AB::Expr, Self::Interface<AB::Expr>>,
    ) -> AB::Expr;

    fn air(&self) -> Self::Air;
}

/// Trait to be implemented on primitive chip to integrate with the machine.
pub trait MachineIntegration<F: PrimeField32, A: MachineAdapter<F>> {
    /// Minimum data that must be recorded to be able to generate trace for one row of `PrimitiveAir`.
    type Record;
    /// Columns of the primitive AIR.
    type Cols<T>;
    /// The primitive AIR with main constraints that do not depend on memory and other architecture-specifics.
    type Air: BaseAir<F>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: Instruction<F>,
        from_pc: F,
        reads: <A::Interface<F> as MachineAdapterInterface<F>>::Reads,
    ) -> Result<(InstructionOutput<F, A::Interface<F>>, Self::Record)>;

    fn get_opcode_name(&self, opcode: usize) -> String;

    // Should mutate `row_slice` to populate with values corresponding to `record`.
    fn generate_trace_row(&self, row_slice: &mut Self::Cols<F>, record: Self::Record);

    /// Returns `(to_pc, interface)`.
    // `local_adapter` provided for flexibility - likely only needed for `from_pc` and `is_valid`
    fn eval_primitive<AB: InteractionBuilder<F = F> + PairBuilder + AirBuilderWithPublicValues>(
        air: &Self::Air,
        builder: &mut AB,
        local: &Self::Cols<AB::Var>,
        local_adapter: &A::Cols<AB::Var>,
    ) -> IntegrationInterface<AB::Expr, A::Interface<AB::Expr>>;

    fn air(&self) -> Self::Air;
}

pub struct InstructionOutput<T, I: MachineAdapterInterface<T>> {
    pub to_pc: T,
    pub writes: I::Writes,
}

pub struct IntegrationInterface<T, I: MachineAdapterInterface<T>> {
    pub to_pc: T,
    pub reads: I::Reads,
    pub writes: I::Writes,
    pub instruction: I::ProcessedInstruction,
}

pub struct MachineChipWrapper<F: PrimeField32, A: MachineAdapter<F>, M: MachineIntegration<F, A>> {
    pub adapter: A,
    pub inner: M,
    pub records: Vec<(A::Record, M::Record)>,
    memory: MemoryChipRef<F>,
}

pub struct MachineAirWrapper<F: PrimeField32, A: MachineAdapter<F>, M: MachineIntegration<F, A>> {
    pub adapter: A::Air,
    pub inner: M::Air,
}

impl<F, A, M> BaseAir<F> for MachineAirWrapper<F, A, M>
where
    F: PrimeField32,
    A: MachineAdapter<F>,
    M: MachineIntegration<F, A>,
{
    fn width(&self) -> usize {
        self.adapter.width() + self.inner.width()
    }
}

impl<F, A, M, AB> Air<AB> for MachineAirWrapper<F, A, M>
where
    F: PrimeField32,
    A: MachineAdapter<F>,
    M: MachineIntegration<F, A>,
    AB: InteractionBuilder<F = F> + PairBuilder + AirBuilderWithPublicValues,
    [AB::Var]: Borrow<A::Cols<AB::Var>>,
    [AB::Var]: Borrow<M::Cols<AB::Var>>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let (local_adapter, local_inner) = local.split_at(self.adapter.width());
        let local_adapter: &A::Cols<AB::Var> = (*local_adapter).borrow();
        let local_inner: &M::Cols<AB::Var> = (*local_inner).borrow();

        let interface = M::eval_primitive(&self.inner, builder, local_inner, local_adapter);
        A::eval_adapter_constraints(&self.adapter, builder, local_adapter, interface);
    }
}

impl<F, A, M> InstructionExecutor<F> for MachineChipWrapper<F, A, M>
where
    F: PrimeField32,
    A: MachineAdapter<F>,
    M: MachineIntegration<F, A>,
{
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>> {
        let mut memory = self.memory.borrow_mut();
        let reads = self
            .adapter
            .preprocess(&mut memory, &instruction, from_state)?;
        let from_pc = F::from_canonical_usize(from_state.pc);
        let (output, inner_record) = self
            .inner
            .execute_instruction(instruction, from_pc, reads)?;
        let (to_state, adapter_record) = self.adapter.postprocess(&mut memory, output)?;
        self.records.push((adapter_record, inner_record));
        Ok(to_state)
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        self.inner.get_opcode_name(opcode)
    }
}

/*TODO
impl<F, A, M> MachineChip<F> for MachineChipWrapper<F, A, M>
where
    F: PrimeField32,
    A: MachineAdapter<F>,
    M: MachineIntegration<F, A>,
    [F]: BorrowMut<A::Cols<F>>,
    [F]: BorrowMut<M::Cols<F>>,
{
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let height = self.records.len().next_power_of_two();
        let inner_width = self.inner.air().width();
        let adapter_width = self.adapter.air().width();
        let width = inner_width + adapter_width;
        let mut values = vec![F::zero(); height * width];
        // This zip only goes through records. The padding rows between records.len()..height
        // are filled with zeros.
        for (row, record) in values.chunks_exact_mut(width).zip(self.records) {
            todo!()
        }
        RowMajorMatrix::new(values, width)
    }
}
*/
