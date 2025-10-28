use std::{array::from_fn, borrow::Borrow, marker::PhantomData, sync::Arc};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::FieldAlgebra,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::{cpu::CpuBackend, types::AirProvingContext},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    Chip,
};
use serde::{Deserialize, Serialize};

use crate::{
    arch::RowMajorMatrixArena,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory, SharedMemoryHelper},
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

/// Helper trait for CPU tracegen.
pub trait TraceFiller<F>: Send + Sync {
    /// Populates `trace`. This function will always be called after
    /// [`TraceExecutor::execute`], so the `trace` should already contain the records necessary to
    /// fill in the rest of it.
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) where
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
                self.fill_dummy_trace_row(row_slice);
            });
    }

    /// Populates `row_slice`. This function will always be called after
    /// [`TraceExecutor::execute`], so the `row_slice` should already contain context necessary to
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
    fn fill_dummy_trace_row(&self, _row_slice: &mut [F]) {
        // By default, the row is filled with zeroes
    }

    /// Returns a list of public values to publish.
    fn generate_public_values(&self) -> Vec<F> {
        vec![]
    }
}

/// We want a blanket implementation of `Chip<MatrixRecordArena, CpuBackend>` on any struct that
/// implements [TraceFiller] but due to Rust orphan rules, we need a wrapper struct.
// @dev: You could make a macro, but it's hard to handle generics in the struct definition.
#[derive(derive_new::new)]
pub struct VmChipWrapper<F, FILLER> {
    pub inner: FILLER,
    pub mem_helper: SharedMemoryHelper<F>,
}

impl<SC, FILLER, RA> Chip<RA, CpuBackend<SC>> for VmChipWrapper<Val<SC>, FILLER>
where
    SC: StarkGenericConfig,
    FILLER: TraceFiller<Val<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
{
    fn generate_proving_ctx(&self, arena: RA) -> AirProvingContext<CpuBackend<SC>> {
        let rows_used = arena.trace_offset() / arena.width();
        let mut trace = arena.into_matrix();
        let mem_helper = self.mem_helper.as_borrowed();
        self.inner.fill_trace(&mem_helper, &mut trace, rows_used);

        AirProvingContext::simple(Arc::new(trace), self.inner.generate_public_values())
    }
}

/// A helper trait for expressing generic state accesses within the implementation of
/// [TraceExecutor]. Note that this is only a helper trait when the same interface of state access
/// is reused or shared by multiple implementations. It is not required to implement this trait if
/// it is easier to implement the [TraceExecutor] trait directly without this trait.
pub trait AdapterTraceExecutor<F>: Clone {
    const WIDTH: usize;
    type ReadData;
    type WriteData;
    // @dev This can either be a &mut _ type or a struct with &mut _ fields.
    // The latter is helpful if we want to directly write certain values in place into a trace
    // matrix.
    type RecordMut<'a>
    where
        Self: 'a;

    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>);

    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData;

    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    );
}

// NOTE[jpw]: cannot reuse `TraceSubRowGenerator` trait because we need associated constant
// `WIDTH`.
pub trait AdapterTraceFiller<F>: Send + Sync {
    const WIDTH: usize;
    /// Post-execution filling of rest of adapter row.
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, adapter_row: &mut [F]);
}

// ============================== Adapter|Core Air Wrapper ===============================

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
