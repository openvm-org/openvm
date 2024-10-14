use std::{borrow::Borrow, marker::PhantomData, sync::Arc};

use afs_derive::AlignedBorrow;
use afs_primitives::utils::next_power_of_two_or_zero;
use afs_stark_backend::{
    air_builders::{
        debug::DebugConstraintBuilder, prover::ProverConstraintFolder, symbolic::SymbolicRapBuilder,
    },
    config::{StarkGenericConfig, Val},
    rap::{get_air_name, AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;

use super::{ExecutionState, InstructionExecutor, Result, VmChip};
use crate::{
    memory::{MemoryChip, MemoryChipRef},
    program::Instruction,
};

/// The interface between primitive AIR and machine adapter AIR.
pub trait VmAdapterInterface<T> {
    /// The memory read data that should be exposed for downstream use
    type Reads;
    /// The memory write data that are expected to be provided by the integrator
    type Writes;
    /// The parts of the instruction that should be exposed to the integrator.
    /// May include the `to_pc`.
    /// Typically this should not include address spaces.
    type ProcessedInstruction;
}

pub type Reads<T, I> = <I as VmAdapterInterface<T>>::Reads;
pub type Writes<T, I> = <I as VmAdapterInterface<T>>::Writes;

/// The adapter owns all memory accesses and timestamp changes.
/// The adapter AIR should also own `ExecutionBridge` and `MemoryBridge`.
pub trait VmAdapterChip<F: Field> {
    /// Records generated by adapter before main instruction execution
    type ReadRecord: Send;
    /// Records generated by adapter after main instruction execution
    type WriteRecord: Send;
    /// AdapterAir should not have public values
    type Air: BaseAir<F> + Clone;
    type Interface<T: AbstractField>: VmAdapterInterface<T>;

    /// Given instruction, perform memory reads and return only the read data that the integrator needs to use.
    /// This is called at the start of instruction execution.
    ///
    /// The implementor may choose to store data in this struct, for example in an [Option], which will later be taken
    /// when `postprocess` is called.
    #[allow(clippy::type_complexity)]
    fn preprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
    ) -> Result<(Reads<F, Self::Interface<F>>, Self::ReadRecord)>;

    /// Given instruction and the data to write, perform memory writes and return the `(record, timestamp_delta)` of the full
    /// adapter record for this instruction. This **must** be called after `preprocess`.
    fn postprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
        output: AdapterRuntimeContext<F, Self::Interface<F>>,
        read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<usize>, Self::WriteRecord)>;

    /// Should mutate `row_slice` to populate with values corresponding to `record`.
    /// The provided `row_slice` will have length equal to `self.air().width()`.
    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
    );

    fn air(&self) -> &Self::Air;
}

pub trait VmAdapterAir<AB: AirBuilder>: BaseAir<AB::F> {
    type Interface: VmAdapterInterface<AB::Expr>;

    /// [Air](p3_air::Air) constraints owned by the adapter.
    /// The `interface` is given as abstract expressions so it can be directly used in other AIR constraints.
    ///
    /// Adapters should document the max constraint degree as a function of the constraint degrees of `reads, writes, instruction`.
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        interface: AdapterAirContext<AB::Expr, Self::Interface>,
    );
}

/// Trait to be implemented on primitive chip to integrate with the machine.
pub trait VmCoreChip<F: PrimeField32, A: VmAdapterChip<F>> {
    /// Minimum data that must be recorded to be able to generate trace for one row of `PrimitiveAir`.
    type Record: Send;
    /// The primitive AIR with main constraints that do not depend on memory and other architecture-specifics.
    type Air: BaseAirWithPublicValues<F> + Clone;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: F,
        reads: Reads<F, A::Interface<F>>,
    ) -> Result<(AdapterRuntimeContext<F, A::Interface<F>>, Self::Record)>;

    fn get_opcode_name(&self, opcode: usize) -> String;

    /// Should mutate `row_slice` to populate with values corresponding to `record`.
    /// The provided `row_slice` will have length equal to `self.air().width()`.
    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record);

    fn air(&self) -> &Self::Air;
}

pub trait VmCoreAir<AB, I>: BaseAirWithPublicValues<AB::F>
where
    AB: AirBuilder,
    I: VmAdapterInterface<AB::Expr>,
{
    /// Returns `(to_pc, interface)`.
    // `local_adapter` provided for flexibility - likely only needed for `from_pc` and `is_valid`
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        local_adapter: &[AB::Var],
    ) -> AdapterAirContext<AB::Expr, I>;
}

pub struct AdapterRuntimeContext<T, I: VmAdapterInterface<T>> {
    /// Leave as `None` to allow the adapter to decide the `to_pc` automatically.
    pub to_pc: Option<T>,
    pub writes: I::Writes,
}

pub struct AdapterAirContext<T, I: VmAdapterInterface<T>> {
    /// Leave as `None` to allow the adapter to decide the `to_pc` automatically.
    pub to_pc: Option<T>,
    pub reads: I::Reads,
    pub writes: I::Writes,
    pub instruction: I::ProcessedInstruction,
}

#[derive(Clone, Debug)]
pub struct VmChipWrapper<F: PrimeField32, A: VmAdapterChip<F>, C: VmCoreChip<F, A>> {
    pub adapter: A,
    pub core: C,
    pub records: Vec<(A::ReadRecord, A::WriteRecord, C::Record)>,
    memory: MemoryChipRef<F>,
}

impl<F, A, C> VmChipWrapper<F, A, C>
where
    F: PrimeField32,
    A: VmAdapterChip<F>,
    C: VmCoreChip<F, A>,
{
    pub fn new(adapter: A, core: C, memory: MemoryChipRef<F>) -> Self {
        Self {
            adapter,
            core,
            records: vec![],
            memory,
        }
    }
}

impl<F, A, M> InstructionExecutor<F> for VmChipWrapper<F, A, M>
where
    F: PrimeField32,
    A: VmAdapterChip<F>,
    M: VmCoreChip<F, A>,
{
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>> {
        let mut memory = self.memory.borrow_mut();
        let (reads, read_record) = self.adapter.preprocess(&mut memory, &instruction)?;
        let from_pc = F::from_canonical_usize(from_state.pc);
        let (output, core_record) = self
            .core
            .execute_instruction(&instruction, from_pc, reads)?;
        let (to_state, write_record) = self.adapter.postprocess(
            &mut memory,
            &instruction,
            from_state,
            output,
            &read_record,
        )?;
        self.records.push((read_record, write_record, core_record));
        Ok(to_state)
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        self.core.get_opcode_name(opcode)
    }
}

impl<F, A, M> VmChip<F> for VmChipWrapper<F, A, M>
where
    F: PrimeField32,
    A: VmAdapterChip<F> + Sync,
    M: VmCoreChip<F, A> + Sync,
{
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let height = next_power_of_two_or_zero(self.records.len());
        let core_width = self.core.air().width();
        let adapter_width = self.adapter.air().width();
        let width = core_width + adapter_width;
        let mut values = vec![F::zero(); height * width];
        // This zip only goes through records.
        // The padding rows between records.len()..height are filled with zeros.
        values
            .par_chunks_mut(width)
            .zip(self.records.into_par_iter())
            .for_each(|(row_slice, record)| {
                let (adapter_row, core_row) = row_slice.split_at_mut(adapter_width);
                self.adapter
                    .generate_trace_row(adapter_row, record.0, record.1);
                self.core.generate_trace_row(core_row, record.2);
            });
        RowMajorMatrix::new(values, width)
    }

    fn air_name(&self) -> String {
        format!(
            "<{},{}>",
            get_air_name(self.adapter.air()),
            get_air_name(self.core.air())
        )
    }

    fn current_trace_height(&self) -> usize {
        self.records.len()
    }

    fn trace_width(&self) -> usize {
        self.adapter.air().width() + self.core.air().width()
    }
}

// Note[jpw]: the statement we want is:
// - when A::Air is an AdapterAir for all AirBuilders needed by stark-backend
// - and when M::Air is an CoreAir for all AirBuilders needed by stark-backend,
// then VmAirWrapper<A::Air, M::Air> is an Air for all AirBuilders needed
// by stark-backend, which is equivalent to saying it implements AnyRap<SC>
// The where clauses to achieve this statement is unfortunately really verbose.
impl<SC, A, C> Chip<SC> for VmChipWrapper<Val<SC>, A, C>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
    A: VmAdapterChip<Val<SC>>,
    C: VmCoreChip<Val<SC>, A>,
    A::Air: 'static,
    A::Air: VmAdapterAir<SymbolicRapBuilder<Val<SC>>>,
    A::Air: for<'a> VmAdapterAir<ProverConstraintFolder<'a, SC>>,
    A::Air: for<'a> VmAdapterAir<DebugConstraintBuilder<'a, SC>>,
    C::Air: 'static,
    C::Air: VmCoreAir<
        SymbolicRapBuilder<Val<SC>>,
        <A::Air as VmAdapterAir<SymbolicRapBuilder<Val<SC>>>>::Interface,
    >,
    C::Air: for<'a> VmCoreAir<
        ProverConstraintFolder<'a, SC>,
        <A::Air as VmAdapterAir<ProverConstraintFolder<'a, SC>>>::Interface,
    >,
    C::Air: for<'a> VmCoreAir<
        DebugConstraintBuilder<'a, SC>,
        <A::Air as VmAdapterAir<DebugConstraintBuilder<'a, SC>>>::Interface,
    >,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        let air: VmAirWrapper<A::Air, C::Air> = VmAirWrapper {
            adapter: self.adapter.air().clone(),
            core: self.core.air().clone(),
        };
        Arc::new(air)
    }
}

pub struct VmAirWrapper<A, C> {
    pub adapter: A,
    pub core: C,
}

impl<F, A, C> BaseAir<F> for VmAirWrapper<A, C>
where
    A: BaseAir<F>,
    C: BaseAir<F>,
{
    fn width(&self) -> usize {
        self.adapter.width() + self.core.width()
    }
}

impl<F, A, M> BaseAirWithPublicValues<F> for VmAirWrapper<A, M>
where
    A: BaseAir<F>,
    M: BaseAirWithPublicValues<F>,
{
    fn num_public_values(&self) -> usize {
        self.core.num_public_values()
    }
}

// Current cached trace is not supported
impl<F, A, M> PartitionedBaseAir<F> for VmAirWrapper<A, M>
where
    A: BaseAir<F>,
    M: BaseAir<F>,
{
}

impl<AB, A, M> Air<AB> for VmAirWrapper<A, M>
where
    AB: AirBuilder,
    A: VmAdapterAir<AB>,
    M: VmCoreAir<AB, A::Interface>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let (local_adapter, local_core) = local.split_at(self.adapter.width());

        let ctx = self.core.eval(builder, local_core, local_adapter);
        self.adapter.eval(builder, local_adapter, ctx);
    }
}

/// The most common adapter interface.
/// Performs `NUM_READS` batch reads of size `READ_SIZE` and
/// `NUM_WRITES` batch writes of size `WRITE_SIZE`.
///
pub struct BasicAdapterInterface<
    T,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
>(PhantomData<T>);

impl<
        T: AbstractField,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterInterface<T>
    for BasicAdapterInterface<T, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    type Reads = [[T; READ_SIZE]; NUM_READS];
    type Writes = [[T; WRITE_SIZE]; NUM_WRITES];
    type ProcessedInstruction = MinimalInstruction<T>;
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MinimalInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
}
