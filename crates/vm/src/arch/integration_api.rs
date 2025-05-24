use std::{array::from_fn, borrow::Borrow, marker::PhantomData, sync::Arc};

use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    AirRef, Chip, ChipUsageGetter,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::{
    execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
    ExecutionState, InsExecutorE1, InstructionExecutor, Result, VmStateMut,
};
use crate::system::memory::{
    online::{GuestMemory, TracingMemory},
    MemoryAuxColsFactory, MemoryController, SharedMemoryHelper,
};

/// The interface between primitive AIR and machine adapter AIR.
pub trait VmAdapterInterface<T> {
    /// The memory read data that should be exposed for downstream use
    type Reads;
    /// The memory write data that are expected to be provided by the integrator
    type Writes;
    /// The parts of the instruction that should be exposed to the integrator.
    /// This will typically include `is_valid`, which indicates whether the trace row
    /// is being used and `opcode` to indicate which opcode is being executed if the
    /// VmChip supports multiple opcodes.
    type ProcessedInstruction;
}

pub trait VmAdapterAir<AB: AirBuilder>: BaseAir<AB::F> {
    type Interface: VmAdapterInterface<AB::Expr>;

    /// [Air](openvm_stark_backend::p3_air::Air) constraints owned by the adapter.
    /// The `interface` is given as abstract expressions so it can be directly used in other AIR
    /// constraints.
    ///
    /// Adapters should document the max constraint degree as a function of the constraint degrees
    /// of `reads, writes, instruction`.
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        interface: AdapterAirContext<AB::Expr, Self::Interface>,
    );

    /// Return the `from_pc` expression.
    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var;
}

// TODO: delete
/// Trait to be implemented on primitive chip to integrate with the machine.
pub trait VmCoreChip<F, I: VmAdapterInterface<F>> {
    /// Minimum data that must be recorded to be able to generate trace for one row of
    /// `PrimitiveAir`.
    type Record: Send + Serialize + DeserializeOwned;
    /// The primitive AIR with main constraints that do not depend on memory and other
    /// architecture-specifics.
    type Air: BaseAirWithPublicValues<F> + Clone;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)>;

    fn get_opcode_name(&self, opcode: usize) -> String;

    /// Populates `row_slice` with values corresponding to `record`.
    /// The provided `row_slice` will have length equal to `self.air().width()`.
    /// This function will be called for each row in the trace which is being used, and all other
    /// rows in the trace will be filled with zeroes.
    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record);

    /// Returns a list of public values to publish.
    fn generate_public_values(&self) -> Vec<F> {
        vec![]
    }

    fn air(&self) -> &Self::Air;

    /// Finalize the trace, especially the padded rows if the all-zero rows don't satisfy the
    /// constraints. This is done **after** records are consumed and the trace matrix is
    /// generated. Most implementations should just leave the default implementation if padding
    /// with rows of all 0s satisfies the constraints.
    fn finalize(&self, _trace: &mut RowMajorMatrix<F>, _num_records: usize) {
        // do nothing by default
    }
}

pub trait VmCoreAir<AB, I>: BaseAirWithPublicValues<AB::F>
where
    AB: AirBuilder,
    I: VmAdapterInterface<AB::Expr>,
{
    /// Returns `(to_pc, interface)`.
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I>;

    /// The offset the opcodes by this chip start from.
    /// This is usually just `CorrespondingOpcode::CLASS_OFFSET`,
    /// but sometimes (for modular chips, for example) it also depends on something else.
    fn start_offset(&self) -> usize;

    fn start_offset_expr(&self) -> AB::Expr {
        AB::Expr::from_canonical_usize(self.start_offset())
    }

    fn expr_to_global_expr(&self, local_expr: impl Into<AB::Expr>) -> AB::Expr {
        self.start_offset_expr() + local_expr.into()
    }

    fn opcode_to_global_expr(&self, local_opcode: impl LocalOpcode) -> AB::Expr {
        self.expr_to_global_expr(AB::Expr::from_canonical_usize(local_opcode.local_usize()))
    }
}

// TODO: delete
pub struct AdapterRuntimeContext<T, I: VmAdapterInterface<T>> {
    /// Leave as `None` to allow the adapter to decide the `to_pc` automatically.
    pub to_pc: Option<u32>,
    pub writes: I::Writes,
}

impl<T, I: VmAdapterInterface<T>> AdapterRuntimeContext<T, I> {
    /// Leave `to_pc` as `None` to allow the adapter to decide the `to_pc` automatically.
    pub fn without_pc(writes: impl Into<I::Writes>) -> Self {
        Self {
            to_pc: None,
            writes: writes.into(),
        }
    }
}

pub struct AdapterAirContext<T, I: VmAdapterInterface<T>> {
    /// Leave as `None` to allow the adapter to decide the `to_pc` automatically.
    pub to_pc: Option<T>,
    pub reads: I::Reads,
    pub writes: I::Writes,
    pub instruction: I::ProcessedInstruction,
}

/// Interface for trace generation of a single instruction.The trace is provided as a mutable
/// buffer during both instruction execution and trace generation.
/// It is expected that no additional memory allocation is necessary and the trace buffer
/// is sufficient, with possible overwriting.
pub trait TraceStep<F, CTX> {
    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        // TODO(ayush): combine to a single struct
        trace: &mut [F],
        trace_offset: &mut usize,
        // TODO(ayush): move air inside step and remove width
        width: usize,
    ) -> Result<()>;

    /// Populates `trace`. This function will always be called after
    /// [`TraceStep::execute`], so the `trace` should already contain context necessary to
    /// fill in the rest of it.
    // TODO(ayush): come up with a better abstraction for chips that fill a dynamic number of rows
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut [F],
        width: usize,
        rows_used: usize,
    ) where
        Self: Send + Sync,
        F: Send + Sync,
    {
        trace[..rows_used * width]
            .par_chunks_exact_mut(width)
            .for_each(|row_slice| {
                self.fill_trace_row(mem_helper, row_slice);
            });
        trace[rows_used * width..]
            .par_chunks_exact_mut(width)
            .for_each(|row_slice| {
                self.fill_dummy_trace_row(mem_helper, row_slice);
            });
    }

    /// Populates `row_slice`. This function will always be called after
    /// [`TraceStep::execute`], so the `row_slice` should already contain context necessary to
    /// fill in the rest of the row. This function will be called for each row in the trace which is
    /// being used, and all other rows in the trace will be filled with zeroes.
    ///
    /// The provided `row_slice` will have length equal to the width of the AIR.
    fn fill_trace_row(&self, _mem_helper: &MemoryAuxColsFactory<F>, _row_slice: &mut [F]) {
        unreachable!("fill_trace_row is not implemented")
    }

    /// Populates `row_slice`. This function will be called on dummy rows.
    /// By default the trace is padded with empty (all 0) rows to make the height a power of 2.
    ///
    /// The provided `row_slice` will have length equal to the width of the AIR.
    fn fill_dummy_trace_row(&self, _mem_helper: &MemoryAuxColsFactory<F>, _row_slice: &mut [F]) {
        // By default, the row is filled with zeroes
    }
    /// Returns a list of public values to publish.
    fn generate_public_values(&self) -> Vec<F> {
        vec![]
    }

    /// Displayable opcode name for logging and debugging purposes.
    fn get_opcode_name(&self, opcode: usize) -> String;
}

// TODO(ayush): rename to ChipWithExecutionContext or something
pub struct NewVmChipWrapper<F, AIR, STEP> {
    pub air: AIR,
    pub step: STEP,
    pub trace_buffer: Vec<F>,
    // TODO(ayush): width should be a constant?
    width: usize,
    buffer_idx: usize,
    mem_helper: SharedMemoryHelper<F>,
}

impl<F, AIR, STEP> NewVmChipWrapper<F, AIR, STEP>
where
    F: Field,
    AIR: BaseAir<F>,
{
    pub fn new(air: AIR, step: STEP, height: usize, mem_helper: SharedMemoryHelper<F>) -> Self {
        assert!(height == 0 || height.is_power_of_two());
        let width = air.width();
        let trace_buffer = F::zero_vec(height * width);
        Self {
            air,
            step,
            trace_buffer,
            width,
            buffer_idx: 0,
            mem_helper,
        }
    }
}

impl<F, AIR, STEP> InstructionExecutor<F> for NewVmChipWrapper<F, AIR, STEP>
where
    F: PrimeField32,
    STEP: TraceStep<F, ()> // TODO: CTX?
        + StepExecutorE1<F>,
{
    fn execute(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>> {
        let mut pc = from_state.pc;
        let state = VmStateMut {
            pc: &mut pc,
            memory: &mut memory.memory,
            ctx: &mut (),
        };
        self.step.execute(
            state,
            instruction,
            &mut self.trace_buffer,
            &mut self.buffer_idx,
            self.width,
        )?;

        Ok(ExecutionState {
            pc,
            timestamp: memory.memory.timestamp,
        })
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        self.step.get_opcode_name(opcode)
    }
}

// Note[jpw]: the statement we want is:
// - `Air` is an `Air<AB>` for all `AB: AirBuilder`s needed by stark-backend
// which is equivalent to saying it implements AirRef<SC>
// The where clauses to achieve this statement is unfortunately really verbose.
impl<SC, AIR, STEP> Chip<SC> for NewVmChipWrapper<Val<SC>, AIR, STEP>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
    STEP: TraceStep<Val<SC>, ()> + Send + Sync,
    AIR: Clone + AnyRap<SC> + 'static,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(mut self) -> AirProofInput<SC> {
        assert_eq!(self.buffer_idx % self.width, 0);
        let rows_used = self.current_trace_height();
        let height = next_power_of_two_or_zero(rows_used);
        // This should be automatic since trace_buffer's height is a power of two:
        assert!(height.checked_mul(self.width).unwrap() <= self.trace_buffer.len());
        self.trace_buffer.truncate(height * self.width);
        let mem_helper = self.mem_helper.as_borrowed();
        self.step
            .fill_trace(&mem_helper, &mut self.trace_buffer, self.width, rows_used);
        drop(self.mem_helper);
        let trace = RowMajorMatrix::new(self.trace_buffer, self.width);
        // self.inner.finalize(&mut trace, num_records);

        AirProofInput::simple(trace, self.step.generate_public_values())
    }
}

impl<F, AIR, C> ChipUsageGetter for NewVmChipWrapper<F, AIR, C>
where
    C: Sync,
{
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.buffer_idx / self.width
    }
    fn trace_width(&self) -> usize {
        self.width
    }
}

// TODO[jpw]: switch read,write to store into abstract buffer, then fill_trace_row using buffer
/// A helper trait for expressing generic state accesses within the implementation of
/// [TraceStep]. Note that this is only a helper trait when the same interface of state access
/// is reused or shared by multiple implementations. It is not required to implement this trait if
/// it is easier to implement the [TraceStep] trait directly without this trait.
pub trait AdapterTraceStep<F, CTX> {
    /// Adapter row width
    const WIDTH: usize;
    type ReadData;
    type WriteData;
    /// The minimal amount of information needed to generate the sub-row of the trace matrix.
    /// This type has a lifetime so other context, such as references to other chips, can be
    /// provided.
    type TraceContext<'a>
    where
        Self: 'a;

    fn start(pc: u32, memory: &TracingMemory<F>, adapter_row: &mut [F]);

    fn read(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData;

    fn write(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    );

    // Note[jpw]: should we reuse TraceSubRowGenerator trait instead?
    /// Post-execution filling of rest of adapter row.
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        ctx: Self::TraceContext<'_>,
        adapter_row: &mut [F],
    );
}

pub trait AdapterExecutorE1<F>
where
    F: PrimeField32,
{
    type ReadData;
    type WriteData;

    fn read<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Self::ReadData
    where
        Ctx: E1E2ExecutionCtx;

    fn write<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
        data: &Self::WriteData,
    ) where
        Ctx: E1E2ExecutionCtx;
}

// TODO: Rename core/step to operator
pub trait StepExecutorE1<F> {
    fn execute_e1<Ctx>(
        &mut self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx;

    fn execute_metered(
        &mut self,
        state: &mut VmStateMut<GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()>;
}

impl<F, A, S> InsExecutorE1<F> for NewVmChipWrapper<F, A, S>
where
    F: PrimeField32,
    S: StepExecutorE1<F>,
{
    fn execute_e1<Ctx>(
        &mut self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        self.step.execute_e1(state, instruction)
    }

    fn execute_metered(
        &mut self,
        state: &mut VmStateMut<GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()>
    where
        F: PrimeField32,
    {
        self.step.execute_metered(state, instruction, chip_index)
    }
}

#[derive(Clone, Copy, derive_new::new)]
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

        let ctx = self
            .core
            .eval(builder, local_core, self.adapter.get_from_pc(local_adapter));
        self.adapter.eval(builder, local_adapter, ctx);
    }
}

// =================================================================================================
// Concrete adapter interfaces
// =================================================================================================

/// The most common adapter interface.
/// Performs `NUM_READS` batch reads of size `READ_SIZE` and
/// `NUM_WRITES` batch writes of size `WRITE_SIZE`.
pub struct BasicAdapterInterface<
    T,
    PI,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
>(PhantomData<T>, PhantomData<PI>);

impl<
        T,
        PI,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterInterface<T>
    for BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    type Reads = [[T; READ_SIZE]; NUM_READS];
    type Writes = [[T; WRITE_SIZE]; NUM_WRITES];
    type ProcessedInstruction = PI;
}

pub struct VecHeapAdapterInterface<
    T,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
>(PhantomData<T>);

impl<
        T,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterInterface<T>
    for VecHeapAdapterInterface<
        T,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    type Reads = [[[T; READ_SIZE]; BLOCKS_PER_READ]; NUM_READS];
    type Writes = [[T; WRITE_SIZE]; BLOCKS_PER_WRITE];
    type ProcessedInstruction = MinimalInstruction<T>;
}

pub struct VecHeapTwoReadsAdapterInterface<
    T,
    const BLOCKS_PER_READ1: usize,
    const BLOCKS_PER_READ2: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
>(PhantomData<T>);

impl<
        T,
        const BLOCKS_PER_READ1: usize,
        const BLOCKS_PER_READ2: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterInterface<T>
    for VecHeapTwoReadsAdapterInterface<
        T,
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    type Reads = (
        [[T; READ_SIZE]; BLOCKS_PER_READ1],
        [[T; READ_SIZE]; BLOCKS_PER_READ2],
    );
    type Writes = [[T; WRITE_SIZE]; BLOCKS_PER_WRITE];
    type ProcessedInstruction = MinimalInstruction<T>;
}

/// Similar to `BasicAdapterInterface`, but it flattens the reads and writes into a single flat
/// array for each
pub struct FlatInterface<T, PI, const READ_CELLS: usize, const WRITE_CELLS: usize>(
    PhantomData<T>,
    PhantomData<PI>,
);

impl<T, PI, const READ_CELLS: usize, const WRITE_CELLS: usize> VmAdapterInterface<T>
    for FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>
{
    type Reads = [T; READ_CELLS];
    type Writes = [T; WRITE_CELLS];
    type ProcessedInstruction = PI;
}

/// An interface that is fully determined during runtime. This should **only** be used as a last
/// resort when static compile-time guarantees cannot be made.
#[derive(Serialize, Deserialize)]
pub struct DynAdapterInterface<T>(PhantomData<T>);

impl<T> VmAdapterInterface<T> for DynAdapterInterface<T> {
    /// Any reads can be flattened into a single vector.
    type Reads = DynArray<T>;
    /// Any writes can be flattened into a single vector.
    type Writes = DynArray<T>;
    /// Any processed instruction can be flattened into a single vector.
    type ProcessedInstruction = DynArray<T>;
}

/// Newtype to implement `From`.
#[derive(Clone, Debug, Default)]
pub struct DynArray<T>(pub Vec<T>);

// =================================================================================================
// Definitions of ProcessedInstruction types for use in integration API
// =================================================================================================

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MinimalInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
}

// This ProcessedInstruction is used by rv32_rdwrite
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ImmInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
    pub immediate: T,
}

// This ProcessedInstruction is used by rv32_jalr
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct SignedImmInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
    pub immediate: T,
    /// Sign of the immediate (1 if negative, 0 if positive)
    pub imm_sign: T,
}

// =================================================================================================
// Conversions between adapter interfaces
// =================================================================================================

mod conversions {
    use super::*;

    // AdapterAirContext: VecHeapAdapterInterface -> DynInterface
    impl<
            T,
            const NUM_READS: usize,
            const BLOCKS_PER_READ: usize,
            const BLOCKS_PER_WRITE: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        >
        From<
            AdapterAirContext<
                T,
                VecHeapAdapterInterface<
                    T,
                    NUM_READS,
                    BLOCKS_PER_READ,
                    BLOCKS_PER_WRITE,
                    READ_SIZE,
                    WRITE_SIZE,
                >,
            >,
        > for AdapterAirContext<T, DynAdapterInterface<T>>
    {
        fn from(
            ctx: AdapterAirContext<
                T,
                VecHeapAdapterInterface<
                    T,
                    NUM_READS,
                    BLOCKS_PER_READ,
                    BLOCKS_PER_WRITE,
                    READ_SIZE,
                    WRITE_SIZE,
                >,
            >,
        ) -> Self {
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads: ctx.reads.into(),
                writes: ctx.writes.into(),
                instruction: ctx.instruction.into(),
            }
        }
    }

    // AdapterRuntimeContext: VecHeapAdapterInterface -> DynInterface
    impl<
            T,
            const NUM_READS: usize,
            const BLOCKS_PER_READ: usize,
            const BLOCKS_PER_WRITE: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        >
        From<
            AdapterRuntimeContext<
                T,
                VecHeapAdapterInterface<
                    T,
                    NUM_READS,
                    BLOCKS_PER_READ,
                    BLOCKS_PER_WRITE,
                    READ_SIZE,
                    WRITE_SIZE,
                >,
            >,
        > for AdapterRuntimeContext<T, DynAdapterInterface<T>>
    {
        fn from(
            ctx: AdapterRuntimeContext<
                T,
                VecHeapAdapterInterface<
                    T,
                    NUM_READS,
                    BLOCKS_PER_READ,
                    BLOCKS_PER_WRITE,
                    READ_SIZE,
                    WRITE_SIZE,
                >,
            >,
        ) -> Self {
            AdapterRuntimeContext {
                to_pc: ctx.to_pc,
                writes: ctx.writes.into(),
            }
        }
    }

    // AdapterAirContext: DynInterface -> VecHeapAdapterInterface
    impl<
            T,
            const NUM_READS: usize,
            const BLOCKS_PER_READ: usize,
            const BLOCKS_PER_WRITE: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        > From<AdapterAirContext<T, DynAdapterInterface<T>>>
        for AdapterAirContext<
            T,
            VecHeapAdapterInterface<
                T,
                NUM_READS,
                BLOCKS_PER_READ,
                BLOCKS_PER_WRITE,
                READ_SIZE,
                WRITE_SIZE,
            >,
        >
    {
        fn from(ctx: AdapterAirContext<T, DynAdapterInterface<T>>) -> Self {
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads: ctx.reads.into(),
                writes: ctx.writes.into(),
                instruction: ctx.instruction.into(),
            }
        }
    }

    // AdapterRuntimeContext: DynInterface -> VecHeapAdapterInterface
    impl<
            T,
            const NUM_READS: usize,
            const BLOCKS_PER_READ: usize,
            const BLOCKS_PER_WRITE: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        > From<AdapterRuntimeContext<T, DynAdapterInterface<T>>>
        for AdapterRuntimeContext<
            T,
            VecHeapAdapterInterface<
                T,
                NUM_READS,
                BLOCKS_PER_READ,
                BLOCKS_PER_WRITE,
                READ_SIZE,
                WRITE_SIZE,
            >,
        >
    {
        fn from(ctx: AdapterRuntimeContext<T, DynAdapterInterface<T>>) -> Self {
            AdapterRuntimeContext {
                to_pc: ctx.to_pc,
                writes: ctx.writes.into(),
            }
        }
    }

    // AdapterAirContext: DynInterface -> VecHeapTwoReadsAdapterInterface
    impl<
            T: Clone,
            const BLOCKS_PER_READ1: usize,
            const BLOCKS_PER_READ2: usize,
            const BLOCKS_PER_WRITE: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        > From<AdapterAirContext<T, DynAdapterInterface<T>>>
        for AdapterAirContext<
            T,
            VecHeapTwoReadsAdapterInterface<
                T,
                BLOCKS_PER_READ1,
                BLOCKS_PER_READ2,
                BLOCKS_PER_WRITE,
                READ_SIZE,
                WRITE_SIZE,
            >,
        >
    {
        fn from(ctx: AdapterAirContext<T, DynAdapterInterface<T>>) -> Self {
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads: ctx.reads.into(),
                writes: ctx.writes.into(),
                instruction: ctx.instruction.into(),
            }
        }
    }

    // AdapterRuntimeContext: DynInterface -> VecHeapAdapterInterface
    impl<
            T,
            const BLOCKS_PER_READ1: usize,
            const BLOCKS_PER_READ2: usize,
            const BLOCKS_PER_WRITE: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        > From<AdapterRuntimeContext<T, DynAdapterInterface<T>>>
        for AdapterRuntimeContext<
            T,
            VecHeapTwoReadsAdapterInterface<
                T,
                BLOCKS_PER_READ1,
                BLOCKS_PER_READ2,
                BLOCKS_PER_WRITE,
                READ_SIZE,
                WRITE_SIZE,
            >,
        >
    {
        fn from(ctx: AdapterRuntimeContext<T, DynAdapterInterface<T>>) -> Self {
            AdapterRuntimeContext {
                to_pc: ctx.to_pc,
                writes: ctx.writes.into(),
            }
        }
    }

    // AdapterRuntimeContext: BasicInterface -> VecHeapAdapterInterface
    impl<
            T,
            PI,
            const BASIC_NUM_READS: usize,
            const BASIC_NUM_WRITES: usize,
            const NUM_READS: usize,
            const BLOCKS_PER_READ: usize,
            const BLOCKS_PER_WRITE: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        >
        From<
            AdapterRuntimeContext<
                T,
                BasicAdapterInterface<
                    T,
                    PI,
                    BASIC_NUM_READS,
                    BASIC_NUM_WRITES,
                    READ_SIZE,
                    WRITE_SIZE,
                >,
            >,
        >
        for AdapterRuntimeContext<
            T,
            VecHeapAdapterInterface<
                T,
                NUM_READS,
                BLOCKS_PER_READ,
                BLOCKS_PER_WRITE,
                READ_SIZE,
                WRITE_SIZE,
            >,
        >
    {
        fn from(
            ctx: AdapterRuntimeContext<
                T,
                BasicAdapterInterface<
                    T,
                    PI,
                    BASIC_NUM_READS,
                    BASIC_NUM_WRITES,
                    READ_SIZE,
                    WRITE_SIZE,
                >,
            >,
        ) -> Self {
            assert_eq!(BASIC_NUM_WRITES, BLOCKS_PER_WRITE);
            let mut writes_it = ctx.writes.into_iter();
            let writes = from_fn(|_| writes_it.next().unwrap());
            AdapterRuntimeContext {
                to_pc: ctx.to_pc,
                writes,
            }
        }
    }

    // AdapterAirContext: BasicInterface -> VecHeapAdapterInterface
    impl<
            T,
            PI: Into<MinimalInstruction<T>>,
            const BASIC_NUM_READS: usize,
            const BASIC_NUM_WRITES: usize,
            const NUM_READS: usize,
            const BLOCKS_PER_READ: usize,
            const BLOCKS_PER_WRITE: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        >
        From<
            AdapterAirContext<
                T,
                BasicAdapterInterface<
                    T,
                    PI,
                    BASIC_NUM_READS,
                    BASIC_NUM_WRITES,
                    READ_SIZE,
                    WRITE_SIZE,
                >,
            >,
        >
        for AdapterAirContext<
            T,
            VecHeapAdapterInterface<
                T,
                NUM_READS,
                BLOCKS_PER_READ,
                BLOCKS_PER_WRITE,
                READ_SIZE,
                WRITE_SIZE,
            >,
        >
    {
        fn from(
            ctx: AdapterAirContext<
                T,
                BasicAdapterInterface<
                    T,
                    PI,
                    BASIC_NUM_READS,
                    BASIC_NUM_WRITES,
                    READ_SIZE,
                    WRITE_SIZE,
                >,
            >,
        ) -> Self {
            assert_eq!(BASIC_NUM_READS, NUM_READS * BLOCKS_PER_READ);
            let mut reads_it = ctx.reads.into_iter();
            let reads = from_fn(|_| from_fn(|_| reads_it.next().unwrap()));
            assert_eq!(BASIC_NUM_WRITES, BLOCKS_PER_WRITE);
            let mut writes_it = ctx.writes.into_iter();
            let writes = from_fn(|_| writes_it.next().unwrap());
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads,
                writes,
                instruction: ctx.instruction.into(),
            }
        }
    }

    // AdapterAirContext: FlatInterface -> BasicInterface
    impl<
            T,
            PI,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
            const READ_CELLS: usize,
            const WRITE_CELLS: usize,
        >
        From<
            AdapterAirContext<
                T,
                BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        > for AdapterAirContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>
    {
        /// ## Panics
        /// If `READ_CELLS != NUM_READS * READ_SIZE` or `WRITE_CELLS != NUM_WRITES * WRITE_SIZE`.
        /// This is a runtime assertion until Rust const generics expressions are stabilized.
        fn from(
            ctx: AdapterAirContext<
                T,
                BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        ) -> AdapterAirContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>> {
            assert_eq!(READ_CELLS, NUM_READS * READ_SIZE);
            assert_eq!(WRITE_CELLS, NUM_WRITES * WRITE_SIZE);
            let mut reads_it = ctx.reads.into_iter().flatten();
            let reads = from_fn(|_| reads_it.next().unwrap());
            let mut writes_it = ctx.writes.into_iter().flatten();
            let writes = from_fn(|_| writes_it.next().unwrap());
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads,
                writes,
                instruction: ctx.instruction,
            }
        }
    }

    // AdapterAirContext: BasicInterface -> FlatInterface
    impl<
            T,
            PI,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
            const READ_CELLS: usize,
            const WRITE_CELLS: usize,
        > From<AdapterAirContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>>
        for AdapterAirContext<
            T,
            BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
        >
    {
        /// ## Panics
        /// If `READ_CELLS != NUM_READS * READ_SIZE` or `WRITE_CELLS != NUM_WRITES * WRITE_SIZE`.
        /// This is a runtime assertion until Rust const generics expressions are stabilized.
        fn from(
            AdapterAirContext {
                to_pc,
                reads,
                writes,
                instruction,
            }: AdapterAirContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>,
        ) -> AdapterAirContext<
            T,
            BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
        > {
            assert_eq!(READ_CELLS, NUM_READS * READ_SIZE);
            assert_eq!(WRITE_CELLS, NUM_WRITES * WRITE_SIZE);
            let mut reads_it = reads.into_iter();
            let reads: [[T; READ_SIZE]; NUM_READS] =
                from_fn(|_| from_fn(|_| reads_it.next().unwrap()));
            let mut writes_it = writes.into_iter();
            let writes: [[T; WRITE_SIZE]; NUM_WRITES] =
                from_fn(|_| from_fn(|_| writes_it.next().unwrap()));
            AdapterAirContext {
                to_pc,
                reads,
                writes,
                instruction,
            }
        }
    }

    // AdapterRuntimeContext: BasicInterface -> FlatInterface
    impl<
            T,
            PI,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
            const READ_CELLS: usize,
            const WRITE_CELLS: usize,
        >
        From<
            AdapterRuntimeContext<
                T,
                BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        > for AdapterRuntimeContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>
    {
        /// ## Panics
        /// If `WRITE_CELLS != NUM_WRITES * WRITE_SIZE`.
        /// This is a runtime assertion until Rust const generics expressions are stabilized.
        fn from(
            ctx: AdapterRuntimeContext<
                T,
                BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        ) -> AdapterRuntimeContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>> {
            assert_eq!(WRITE_CELLS, NUM_WRITES * WRITE_SIZE);
            let mut writes_it = ctx.writes.into_iter().flatten();
            let writes = from_fn(|_| writes_it.next().unwrap());
            AdapterRuntimeContext {
                to_pc: ctx.to_pc,
                writes,
            }
        }
    }

    // AdapterRuntimeContext: FlatInterface -> BasicInterface
    impl<
            T: FieldAlgebra,
            PI,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
            const READ_CELLS: usize,
            const WRITE_CELLS: usize,
        > From<AdapterRuntimeContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>>
        for AdapterRuntimeContext<
            T,
            BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
        >
    {
        /// ## Panics
        /// If `WRITE_CELLS != NUM_WRITES * WRITE_SIZE`.
        /// This is a runtime assertion until Rust const generics expressions are stabilized.
        fn from(
            ctx: AdapterRuntimeContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>,
        ) -> AdapterRuntimeContext<
            T,
            BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
        > {
            assert_eq!(WRITE_CELLS, NUM_WRITES * WRITE_SIZE);
            let mut writes_it = ctx.writes.into_iter();
            let writes: [[T; WRITE_SIZE]; NUM_WRITES] =
                from_fn(|_| from_fn(|_| writes_it.next().unwrap()));
            AdapterRuntimeContext {
                to_pc: ctx.to_pc,
                writes,
            }
        }
    }

    impl<T> From<Vec<T>> for DynArray<T> {
        fn from(v: Vec<T>) -> Self {
            Self(v)
        }
    }

    impl<T> From<DynArray<T>> for Vec<T> {
        fn from(v: DynArray<T>) -> Vec<T> {
            v.0
        }
    }

    impl<T, const N: usize, const M: usize> From<[[T; N]; M]> for DynArray<T> {
        fn from(v: [[T; N]; M]) -> Self {
            Self(v.into_iter().flatten().collect())
        }
    }

    impl<T, const N: usize, const M: usize> From<DynArray<T>> for [[T; N]; M] {
        fn from(v: DynArray<T>) -> Self {
            assert_eq!(v.0.len(), N * M, "Incorrect vector length {}", v.0.len());
            let mut it = v.0.into_iter();
            from_fn(|_| from_fn(|_| it.next().unwrap()))
        }
    }

    impl<T, const N: usize, const M: usize, const R: usize> From<[[[T; N]; M]; R]> for DynArray<T> {
        fn from(v: [[[T; N]; M]; R]) -> Self {
            Self(
                v.into_iter()
                    .flat_map(|x| x.into_iter().flatten())
                    .collect(),
            )
        }
    }

    impl<T, const N: usize, const M: usize, const R: usize> From<DynArray<T>> for [[[T; N]; M]; R] {
        fn from(v: DynArray<T>) -> Self {
            assert_eq!(
                v.0.len(),
                N * M * R,
                "Incorrect vector length {}",
                v.0.len()
            );
            let mut it = v.0.into_iter();
            from_fn(|_| from_fn(|_| from_fn(|_| it.next().unwrap())))
        }
    }

    impl<T, const N: usize, const M1: usize, const M2: usize> From<([[T; N]; M1], [[T; N]; M2])>
        for DynArray<T>
    {
        fn from(v: ([[T; N]; M1], [[T; N]; M2])) -> Self {
            let vec =
                v.0.into_iter()
                    .flatten()
                    .chain(v.1.into_iter().flatten())
                    .collect();
            Self(vec)
        }
    }

    impl<T, const N: usize, const M1: usize, const M2: usize> From<DynArray<T>>
        for ([[T; N]; M1], [[T; N]; M2])
    {
        fn from(v: DynArray<T>) -> Self {
            assert_eq!(
                v.0.len(),
                N * (M1 + M2),
                "Incorrect vector length {}",
                v.0.len()
            );
            let mut it = v.0.into_iter();
            (
                from_fn(|_| from_fn(|_| it.next().unwrap())),
                from_fn(|_| from_fn(|_| it.next().unwrap())),
            )
        }
    }

    // AdapterAirContext: BasicInterface -> DynInterface
    impl<
            T,
            PI: Into<DynArray<T>>,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        >
        From<
            AdapterAirContext<
                T,
                BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        > for AdapterAirContext<T, DynAdapterInterface<T>>
    {
        fn from(
            ctx: AdapterAirContext<
                T,
                BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        ) -> Self {
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads: ctx.reads.into(),
                writes: ctx.writes.into(),
                instruction: ctx.instruction.into(),
            }
        }
    }

    // AdapterRuntimeContext: BasicInterface -> DynInterface
    impl<
            T,
            PI,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        >
        From<
            AdapterRuntimeContext<
                T,
                BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        > for AdapterRuntimeContext<T, DynAdapterInterface<T>>
    {
        fn from(
            ctx: AdapterRuntimeContext<
                T,
                BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
            >,
        ) -> Self {
            AdapterRuntimeContext {
                to_pc: ctx.to_pc,
                writes: ctx.writes.into(),
            }
        }
    }

    // AdapterAirContext: DynInterface -> BasicInterface
    impl<
            T,
            PI,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        > From<AdapterAirContext<T, DynAdapterInterface<T>>>
        for AdapterAirContext<
            T,
            BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
        >
    where
        PI: From<DynArray<T>>,
    {
        fn from(ctx: AdapterAirContext<T, DynAdapterInterface<T>>) -> Self {
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads: ctx.reads.into(),
                writes: ctx.writes.into(),
                instruction: ctx.instruction.into(),
            }
        }
    }

    // AdapterRuntimeContext: DynInterface -> BasicInterface
    impl<
            T,
            PI,
            const NUM_READS: usize,
            const NUM_WRITES: usize,
            const READ_SIZE: usize,
            const WRITE_SIZE: usize,
        > From<AdapterRuntimeContext<T, DynAdapterInterface<T>>>
        for AdapterRuntimeContext<
            T,
            BasicAdapterInterface<T, PI, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
        >
    {
        fn from(ctx: AdapterRuntimeContext<T, DynAdapterInterface<T>>) -> Self {
            AdapterRuntimeContext {
                to_pc: ctx.to_pc,
                writes: ctx.writes.into(),
            }
        }
    }

    // AdapterAirContext: FlatInterface -> DynInterface
    impl<T: Clone, PI: Into<DynArray<T>>, const READ_CELLS: usize, const WRITE_CELLS: usize>
        From<AdapterAirContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>>
        for AdapterAirContext<T, DynAdapterInterface<T>>
    {
        fn from(ctx: AdapterAirContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>) -> Self {
            AdapterAirContext {
                to_pc: ctx.to_pc,
                reads: ctx.reads.to_vec().into(),
                writes: ctx.writes.to_vec().into(),
                instruction: ctx.instruction.into(),
            }
        }
    }

    // AdapterRuntimeContext: FlatInterface -> DynInterface
    impl<T: Clone, PI, const READ_CELLS: usize, const WRITE_CELLS: usize>
        From<AdapterRuntimeContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>>
        for AdapterRuntimeContext<T, DynAdapterInterface<T>>
    {
        fn from(
            ctx: AdapterRuntimeContext<T, FlatInterface<T, PI, READ_CELLS, WRITE_CELLS>>,
        ) -> Self {
            AdapterRuntimeContext {
                to_pc: ctx.to_pc,
                writes: ctx.writes.to_vec().into(),
            }
        }
    }

    impl<T> From<MinimalInstruction<T>> for DynArray<T> {
        fn from(m: MinimalInstruction<T>) -> Self {
            Self(vec![m.is_valid, m.opcode])
        }
    }

    impl<T> From<DynArray<T>> for MinimalInstruction<T> {
        fn from(m: DynArray<T>) -> Self {
            let mut m = m.0.into_iter();
            MinimalInstruction {
                is_valid: m.next().unwrap(),
                opcode: m.next().unwrap(),
            }
        }
    }

    impl<T> From<DynArray<T>> for ImmInstruction<T> {
        fn from(m: DynArray<T>) -> Self {
            let mut m = m.0.into_iter();
            ImmInstruction {
                is_valid: m.next().unwrap(),
                opcode: m.next().unwrap(),
                immediate: m.next().unwrap(),
            }
        }
    }

    impl<T> From<ImmInstruction<T>> for DynArray<T> {
        fn from(instruction: ImmInstruction<T>) -> Self {
            DynArray::from(vec![
                instruction.is_valid,
                instruction.opcode,
                instruction.immediate,
            ])
        }
    }
}
