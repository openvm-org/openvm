use std::{
    any::type_name,
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    sync::Arc,
};

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
use serde::{Deserialize, Serialize};

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

pub struct AdapterAirContext<T, I: VmAdapterInterface<T>> {
    /// Leave as `None` to allow the adapter to decide the `to_pc` automatically.
    pub to_pc: Option<T>,
    pub reads: I::Reads,
    pub writes: I::Writes,
    pub instruction: I::ProcessedInstruction,
}

/// Given some minimum metadata of type `Layout` that specifies the record size, the `RecordArena`
/// should allocate a buffer, of size possibly larger than the record, and then return mutable
/// pointers to the record within the buffer.
pub trait RecordArena<'a, Layout, RecordMut> {
    /// Allocates underlying buffer and returns a mutable reference `RecordMut`.
    /// Note that calling this function may not call an underlying memory allocation as the record
    /// arena may be virtual.
    fn alloc(&'a mut self, layout: Layout) -> RecordMut;
}

/// ZST to represent empty layout. Used when the layout can be inferred from other context (such as
/// AIR or record types).
pub struct EmptyLayout;

/// Interface for trace generation of a single instruction.The trace is provided as a mutable
/// buffer during both instruction execution and trace generation.
/// It is expected that no additional memory allocation is necessary and the trace buffer
/// is sufficient, with possible overwriting.
pub trait TraceStep<F, CTX> {
    type RecordLayout;
    type RecordMut<'a>;

    fn execute<'buf, RA>(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        arena: &'buf mut RA,
    ) -> Result<()>
    where
        RA: RecordArena<'buf, Self::RecordLayout, Self::RecordMut<'buf>>;

    /// Returns a list of public values to publish.
    fn generate_public_values(&self) -> Vec<F> {
        vec![]
    }

    /// Displayable opcode name for logging and debugging purposes.
    fn get_opcode_name(&self, opcode: usize) -> String;
}

// TODO[jpw]: this might be temporary trait before moving trace to CTX
pub trait RowMajorMatrixArena<F> {
    fn with_capacity(height: usize, width: usize) -> Self;
    fn width(&self) -> usize;
    fn trace_offset(&self) -> usize;
    fn into_matrix(self) -> RowMajorMatrix<F>;
}

// TODO[jpw]: revisit if this trait makes sense
pub trait TraceFiller<F, CTX> {
    /// Populates `trace`. This function will always be called after
    /// [`TraceStep::execute`], so the `trace` should already contain the records necessary to fill
    /// in the rest of it.
    // TODO(ayush): come up with a better abstraction for chips that fill a dynamic number of rows
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) where
        Self: Send + Sync,
        F: Send + Sync + Clone,
    {
        let width = trace.width();
        trace.values[..rows_used * width]
            .par_chunks_exact_mut(width)
            .for_each(|row_slice| {
                self.fill_trace_row(mem_helper, row_slice);
            });
        trace.values[rows_used * width..]
            .par_chunks_exact_mut(width)
            .for_each(|row_slice| {
                self.fill_dummy_trace_row(mem_helper, row_slice);
            });
    }

    /// Populates `row_slice`. This function will always be called after
    /// [`TraceStep::execute`], so the `row_slice` should already contain context necessary to
    /// fill in the rest of the row. This function will be called for each row in the trace which
    /// is being used, and for all other rows in the trace see `fill_dummy_trace_row`.
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
}

pub struct MatrixRecordArena<F> {
    pub trace_buffer: Vec<F>,
    // TODO(ayush): width should be a constant?
    pub width: usize,
    pub trace_offset: usize,
}

/// The minimal information that [AdapterCoreRecordArena] needs to know to allocate a row
/// **WARNING**: `adapter_width` is number of field elements, not in bytes
pub struct AdapterCoreLayout {
    pub adapter_width: usize,
}

/// A record arena struct that can be used by chips that
/// - have a single row per instruction
/// - have trace row = [adapter_row, core_row]
// TEMP[jpw]: buffer should be inside CTX
pub struct AdapterCoreRecordArena<F> {
    pub trace_buffer: Vec<F>,
    // TODO(ayush): width should be a constant?
    pub width: usize,
    pub trace_offset: usize,
}

/// RecordArena implementation for [AdapterCoreRecordArena]
/// A is the adapter record type and C is the core record type
impl<'a, F: Field, A, C> RecordArena<'a, AdapterCoreLayout, (&'a mut A, &'a mut C)>
    for AdapterCoreRecordArena<F>
where
    A: Sized,
    C: Sized,
    [u8]: BorrowMut<A> + BorrowMut<C>,
{
    fn alloc(&'a mut self, layout: AdapterCoreLayout) -> (&'a mut A, &'a mut C) {
        let buffer = self.alloc_single_row();
        let (adapter_buffer, core_buffer) =
            buffer.split_at_mut(layout.adapter_width * size_of::<F>());

        let adapter_record: &mut A = adapter_buffer.borrow_mut();
        let core_record: &mut C = core_buffer.borrow_mut();

        (adapter_record, core_record)
    }
}

impl<F: Field> AdapterCoreRecordArena<F> {
    pub fn alloc_single_row(&mut self) -> &mut [u8] {
        let start = self.trace_offset;
        self.trace_offset += self.width;
        let row_slice = &mut self.trace_buffer[start..self.trace_offset];
        let size = size_of_val(row_slice);
        let ptr = row_slice as *mut [F] as *mut u8;
        // SAFETY:
        // - `ptr` is non-null
        // - `size` is correct
        // - alignment of `u8` is always satisfied
        unsafe { &mut *std::ptr::slice_from_raw_parts_mut(ptr, size) }
    }
}

impl<F: Field> RowMajorMatrixArena<F> for AdapterCoreRecordArena<F> {
    fn with_capacity(height: usize, width: usize) -> Self {
        let trace_buffer = F::zero_vec(height * width);
        Self {
            trace_buffer,
            width,
            trace_offset: 0,
        }
    }

    fn width(&self) -> usize {
        self.width
    }

    fn trace_offset(&self) -> usize {
        self.trace_offset
    }

    fn into_matrix(self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(self.trace_buffer, self.width)
    }
}

/// A trait that allows for custom implementation of `borrow` given the necessary information
/// This is useful for record structs that have dynamic size
pub trait CustomBorrow<'a, T, I> {
    fn custom_borrow(&'a mut self, metadata: I) -> T;
}

/// The minimal information that [MultiRowRecordArena] needs to know to allocate a record
pub struct MultiRowLayout<I> {
    pub num_rows: u32,
    pub metadata: I,
}

// TEMP[jpw]: buffer should be inside CTX
pub struct MultiRowRecordArena<F> {
    pub trace_buffer: Vec<F>,
    // TODO(ayush): width should be a constant?
    pub width: usize,
    pub trace_offset: usize,
}

impl<F: Field> MultiRowRecordArena<F> {
    pub fn alloc_buffer(&mut self, num_rows: u32) -> &mut [u8] {
        let start = self.trace_offset;
        self.trace_offset += num_rows as usize * self.width;
        let row_slice = &mut self.trace_buffer[start..self.trace_offset];
        let size = size_of_val(row_slice);
        let ptr = row_slice as *mut [F] as *mut u8;
        // SAFETY:
        // - `ptr` is non-null
        // - `size` is correct
        // - alignment of `u8` is always satisfied
        unsafe { &mut *std::ptr::slice_from_raw_parts_mut(ptr, size) }
    }
}

/// RecordArena implementation for [MultiRowRecordArena]
/// R is the RecordMut type
impl<'a, I, F: Field, R> RecordArena<'a, MultiRowLayout<I>, R> for MultiRowRecordArena<F>
where
    [u8]: CustomBorrow<'a, R, I>,
{
    fn alloc(&'a mut self, layout: MultiRowLayout<I>) -> R {
        let buffer = self.alloc_buffer(layout.num_rows);
        let record: R = buffer.custom_borrow(layout.metadata);
        record
    }
}

impl<F: Field> RowMajorMatrixArena<F> for MultiRowRecordArena<F> {
    fn with_capacity(height: usize, width: usize) -> Self {
        let trace_buffer = F::zero_vec(height * width);
        Self {
            trace_buffer,
            width,
            trace_offset: 0,
        }
    }

    fn width(&self) -> usize {
        self.width
    }

    fn trace_offset(&self) -> usize {
        self.trace_offset
    }

    fn into_matrix(self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(self.trace_buffer, self.width)
    }
}

// TODO(ayush): rename to ChipWithExecutionContext or something
pub struct NewVmChipWrapper<F, AIR, STEP, RA> {
    pub air: AIR,
    pub step: STEP,
    pub arena: RA,
    mem_helper: SharedMemoryHelper<F>,
}

impl<F, AIR, STEP, RA> NewVmChipWrapper<F, AIR, STEP, RA>
where
    F: Field,
    AIR: BaseAir<F>,
    RA: RowMajorMatrixArena<F>,
{
    pub fn new(air: AIR, step: STEP, height: usize, mem_helper: SharedMemoryHelper<F>) -> Self {
        let width = air.width();
        assert!(height == 0 || height.is_power_of_two());
        assert!(
            align_of::<F>() >= align_of::<u32>(),
            "type {} should have at least alignment of u32",
            type_name::<F>()
        );
        let arena = RA::with_capacity(height, width);
        Self {
            air,
            step,
            arena,
            mem_helper,
        }
    }
}

impl<F, AIR, STEP, RA> InstructionExecutor<F> for NewVmChipWrapper<F, AIR, STEP, RA>
where
    F: PrimeField32,
    STEP: TraceStep<F, ()> // TODO: CTX?
        + StepExecutorE1<F>,
    for<'buf> RA: RecordArena<'buf, STEP::RecordLayout, STEP::RecordMut<'buf>>,
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
        self.step.execute(state, instruction, &mut self.arena)?;

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
impl<SC, AIR, STEP, RA> Chip<SC> for NewVmChipWrapper<Val<SC>, AIR, STEP, RA>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
    STEP: TraceStep<Val<SC>, ()> + TraceFiller<Val<SC>, ()> + Send + Sync,
    AIR: Clone + AnyRap<SC> + 'static,
    RA: RowMajorMatrixArena<Val<SC>>,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let width = self.arena.width();
        assert_eq!(self.arena.trace_offset() % width, 0);
        let rows_used = self.arena.trace_offset() / width;
        let height = next_power_of_two_or_zero(rows_used);
        let mut trace = self.arena.into_matrix();
        // This should be automatic since trace_buffer's height is a power of two:
        assert!(height.checked_mul(width).unwrap() <= trace.values.len());
        trace.values.truncate(height * width);
        let mem_helper = self.mem_helper.as_borrowed();
        self.step.fill_trace(&mem_helper, &mut trace, rows_used);
        drop(self.mem_helper);

        AirProofInput::simple(trace, self.step.generate_public_values())
    }
}

impl<F, AIR, C, RA> ChipUsageGetter for NewVmChipWrapper<F, AIR, C, RA>
where
    C: Sync,
    RA: RowMajorMatrixArena<F>,
{
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.arena.trace_offset() / self.arena.width()
    }
    fn trace_width(&self) -> usize {
        self.arena.width()
    }
}

/// A helper trait for expressing generic state accesses within the implementation of
/// [TraceStep]. Note that this is only a helper trait when the same interface of state access
/// is reused or shared by multiple implementations. It is not required to implement this trait if
/// it is easier to implement the [TraceStep] trait directly without this trait.
pub trait AdapterTraceStep<F, CTX> {
    const WIDTH: usize;
    type ReadData;
    type WriteData;
    // @dev This can either be a &mut _ type or a struct with &mut _ fields.
    // The latter is helpful if we want to directly write certain values in place into a trace
    // matrix.
    type RecordMut<'a>
    where
        Self: 'a;

    // /// The minimal amount of information needed to generate the sub-row of the trace matrix.
    // /// This type has a lifetime so other context, such as references to other chips, can be
    // /// provided.
    // type TraceContext<'a>
    // where
    //     Self: 'a;

    fn start(pc: u32, memory: &TracingMemory<F>, record: &mut Self::RecordMut<'_>);

    fn read(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData;

    fn write(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        data: &Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    );
}

// NOTE[jpw]: cannot reuse `TraceSubRowGenerator` trait because we need associated constant
// `WIDTH`.
pub trait AdapterTraceFiller<F, CTX>: AdapterTraceStep<F, CTX> {
    /// Post-execution filling of rest of adapter row.
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, adapter_row: &mut [F]);
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
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx;

    fn execute_metered(
        &self,
        state: &mut VmStateMut<GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()>;
}

const DEFAULT_RECORDS_CAPACITY: usize = 1 << 20;

impl<F, A, S, RA> InsExecutorE1<F> for NewVmChipWrapper<F, A, S, RA>
where
    F: PrimeField32,
    S: StepExecutorE1<F>,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        self.step.execute_e1(state, instruction)
    }

    fn execute_metered(
        &self,
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
