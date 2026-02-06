//! Adapter wrappers that convert between Vec/Block-based interfaces and Flat/Basic interfaces.
//!
//! These wrappers allow using `Rv32VecHeapAdapter*` types with cores that expect flat
//! `BasicAdapterInterface` data formats.

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterTraceExecutor, BasicAdapterInterface, ImmInstruction,
        MinimalInstruction, VecHeapAdapterInterface, VecHeapBranchAdapterInterface, VmAdapterAir,
        VmAdapterInterface,
    },
    system::memory::online::TracingMemory,
};
use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::{
    interaction::InteractionBuilder, p3_air::BaseAir, p3_field::PrimeField32,
    rap::BaseAirWithPublicValues,
};

// =================================================================================================
// ALU Adapter Wrappers (with reads and writes)
// =================================================================================================

/// Wrapper that converts a `VecHeapAdapterInterface` (block-based) to `BasicAdapterInterface`
/// (flat).
///
/// This allows using `Rv32VecHeapAdapterAir` with cores that expect flat read/write data.
///
/// # Type Parameters
/// - `A`: The inner adapter AIR (e.g., `Rv32VecHeapAdapterAir`)
/// - `NUM_READS`: Number of read operands
/// - `BLOCKS_PER_READ`: Number of blocks per read operand
/// - `BLOCKS_PER_WRITE`: Number of blocks per write operand
/// - `BLOCK_SIZE`: Size of each block
/// - `TOTAL_READ_SIZE`: Total read size per operand (`BLOCKS_PER_READ * BLOCK_SIZE`)
/// - `TOTAL_WRITE_SIZE`: Total write size per operand (`BLOCKS_PER_WRITE * BLOCK_SIZE`)
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct VecToFlatAluAdapterAir<
    A,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
    const TOTAL_WRITE_SIZE: usize,
>(pub A);

impl<
        F,
        A,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
        const TOTAL_WRITE_SIZE: usize,
    > BaseAir<F>
    for VecToFlatAluAdapterAir<
        A,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        BLOCK_SIZE,
        TOTAL_READ_SIZE,
        TOTAL_WRITE_SIZE,
    >
where
    A: BaseAir<F>,
{
    fn width(&self) -> usize {
        self.0.width()
    }
}

impl<
        AB,
        A,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
        const TOTAL_WRITE_SIZE: usize,
    > VmAdapterAir<AB>
    for VecToFlatAluAdapterAir<
        A,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        BLOCK_SIZE,
        TOTAL_READ_SIZE,
        TOTAL_WRITE_SIZE,
    >
where
    AB: InteractionBuilder,
    A: VmAdapterAir<
        AB,
        Interface = VecHeapAdapterInterface<
            AB::Expr,
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            BLOCK_SIZE,
            BLOCK_SIZE,
        >,
    >,
{
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        NUM_READS,
        1,
        TOTAL_READ_SIZE,
        TOTAL_WRITE_SIZE,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        // Runtime assertions for const generic relationships
        assert_eq!(
            TOTAL_READ_SIZE,
            BLOCKS_PER_READ * BLOCK_SIZE,
            "TOTAL_READ_SIZE must equal BLOCKS_PER_READ * BLOCK_SIZE"
        );
        assert_eq!(
            TOTAL_WRITE_SIZE,
            BLOCKS_PER_WRITE * BLOCK_SIZE,
            "TOTAL_WRITE_SIZE must equal BLOCKS_PER_WRITE * BLOCK_SIZE"
        );

        type InnerI<T, const NR: usize, const BPR: usize, const BPW: usize, const BS: usize> =
            VecHeapAdapterInterface<T, NR, BPR, BPW, BS, BS>;

        let inner_reads: <InnerI<
            AB::Expr,
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            BLOCK_SIZE,
        > as VmAdapterInterface<AB::Expr>>::Reads = core::array::from_fn(|read_i| {
            core::array::from_fn(|block_i| {
                core::array::from_fn(|in_block_i| {
                    let byte_i = block_i * BLOCK_SIZE + in_block_i;
                    ctx.reads[read_i][byte_i].clone()
                })
            })
        });

        let inner_writes: <InnerI<
            AB::Expr,
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            BLOCK_SIZE,
        > as VmAdapterInterface<AB::Expr>>::Writes = core::array::from_fn(|block_i| {
            core::array::from_fn(|in_block_i| {
                let byte_i = block_i * BLOCK_SIZE + in_block_i;
                ctx.writes[0][byte_i].clone()
            })
        });

        let inner_ctx: AdapterAirContext<
            AB::Expr,
            InnerI<AB::Expr, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE, BLOCK_SIZE>,
        > = AdapterAirContext {
            to_pc: ctx.to_pc,
            reads: inner_reads,
            writes: inner_writes,
            instruction: ctx.instruction,
        };

        self.0.eval(builder, local, inner_ctx)
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        self.0.get_from_pc(local)
    }
}

impl<
        F,
        A,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
        const TOTAL_WRITE_SIZE: usize,
    > BaseAirWithPublicValues<F>
    for VecToFlatAluAdapterAir<
        A,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        BLOCK_SIZE,
        TOTAL_READ_SIZE,
        TOTAL_WRITE_SIZE,
    >
where
    A: BaseAirWithPublicValues<F>,
{
    fn num_public_values(&self) -> usize {
        self.0.num_public_values()
    }
}

/// Wrapper that converts block-based read/write data to flat format for ALU operations.
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct VecToFlatAluAdapterExecutor<
    A,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
    const TOTAL_WRITE_SIZE: usize,
>(pub A);

impl<
        F,
        A,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
        const TOTAL_WRITE_SIZE: usize,
    > AdapterTraceExecutor<F>
    for VecToFlatAluAdapterExecutor<
        A,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        BLOCK_SIZE,
        TOTAL_READ_SIZE,
        TOTAL_WRITE_SIZE,
    >
where
    F: PrimeField32,
    A: AdapterTraceExecutor<
        F,
        ReadData = [[[u8; BLOCK_SIZE]; BLOCKS_PER_READ]; NUM_READS],
        WriteData = [[u8; BLOCK_SIZE]; BLOCKS_PER_WRITE],
    >,
{
    const WIDTH: usize = A::WIDTH;
    type ReadData = [[u8; TOTAL_READ_SIZE]; NUM_READS];
    type WriteData = [[u8; TOTAL_WRITE_SIZE]; 1];
    type RecordMut<'a>
        = A::RecordMut<'a>
    where
        Self: 'a;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        A::start(pc, memory, record);
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let data_inner = <A as AdapterTraceExecutor<F>>::read(&self.0, memory, instruction, record);

        core::array::from_fn(|i| {
            let mut out = [0u8; TOTAL_READ_SIZE];
            for (block_idx, block) in data_inner[i].iter().enumerate() {
                let start = block_idx * BLOCK_SIZE;
                out[start..start + BLOCK_SIZE].copy_from_slice(&block[..]);
            }
            out
        })
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let data_inner: <A as AdapterTraceExecutor<F>>::WriteData = core::array::from_fn(|i| {
            let start = i * BLOCK_SIZE;
            data[0][start..start + BLOCK_SIZE]
                .try_into()
                .expect("slice length matches BLOCK_SIZE")
        });

        <A as AdapterTraceExecutor<F>>::write(&self.0, memory, instruction, data_inner, record);
    }
}

// =================================================================================================
// Branch Adapter Wrappers (reads only, no writes)
// =================================================================================================

/// Wrapper that converts a `VecHeapBranchAdapterInterface` (block-based) to `BasicAdapterInterface`
/// (flat).
///
/// This allows using `Rv32VecHeapBranchAdapterAir` with cores that expect flat read data.
/// Branch operations have no writes.
///
/// # Type Parameters
/// - `A`: The inner adapter AIR (e.g., `Rv32VecHeapBranchAdapterAir`)
/// - `NUM_READS`: Number of read operands
/// - `BLOCKS_PER_READ`: Number of blocks per read operand
/// - `BLOCK_SIZE`: Size of each block
/// - `TOTAL_READ_SIZE`: Total read size per operand (`BLOCKS_PER_READ * BLOCK_SIZE`)
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct VecToFlatBranchAdapterAir<
    A,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
>(pub A);

impl<
        F,
        A,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > BaseAir<F>
    for VecToFlatBranchAdapterAir<A, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
where
    A: BaseAir<F>,
{
    fn width(&self) -> usize {
        self.0.width()
    }
}

impl<
        AB,
        A,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > VmAdapterAir<AB>
    for VecToFlatBranchAdapterAir<A, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
where
    AB: InteractionBuilder,
    A: VmAdapterAir<
        AB,
        Interface = VecHeapBranchAdapterInterface<AB::Expr, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>,
    >,
{
    type Interface =
        BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, NUM_READS, 0, TOTAL_READ_SIZE, 0>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        // Runtime assertion for const generic relationship
        assert_eq!(
            TOTAL_READ_SIZE,
            BLOCKS_PER_READ * BLOCK_SIZE,
            "TOTAL_READ_SIZE must equal BLOCKS_PER_READ * BLOCK_SIZE"
        );

        type InnerI<T, const NR: usize, const BPR: usize, const BS: usize> =
            VecHeapBranchAdapterInterface<T, NR, BPR, BS>;

        let inner_reads: <InnerI<AB::Expr, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> as VmAdapterInterface<AB::Expr>>::Reads =
            core::array::from_fn(|read_i| {
                core::array::from_fn(|block_i| {
                    core::array::from_fn(|in_block_i| {
                        let byte_i = block_i * BLOCK_SIZE + in_block_i;
                        ctx.reads[read_i][byte_i].clone()
                    })
                })
            });

        let inner_ctx: AdapterAirContext<
            AB::Expr,
            InnerI<AB::Expr, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>,
        > = AdapterAirContext {
            to_pc: ctx.to_pc,
            reads: inner_reads,
            writes: (),
            instruction: ctx.instruction,
        };

        self.0.eval(builder, local, inner_ctx)
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        self.0.get_from_pc(local)
    }
}

impl<
        F,
        A,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > BaseAirWithPublicValues<F>
    for VecToFlatBranchAdapterAir<A, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
where
    A: BaseAirWithPublicValues<F>,
{
    fn num_public_values(&self) -> usize {
        self.0.num_public_values()
    }
}

/// Wrapper that converts block-based read data to flat format for branch operations.
/// Branch operations have no writes.
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct VecToFlatBranchAdapterExecutor<
    A,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
>(pub A);

impl<
        F,
        A,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > AdapterTraceExecutor<F>
    for VecToFlatBranchAdapterExecutor<A, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
where
    F: PrimeField32,
    A: AdapterTraceExecutor<
        F,
        ReadData = [[[u8; BLOCK_SIZE]; BLOCKS_PER_READ]; NUM_READS],
        WriteData = (),
    >,
{
    const WIDTH: usize = A::WIDTH;
    type ReadData = [[u8; TOTAL_READ_SIZE]; NUM_READS];
    type WriteData = ();
    type RecordMut<'a>
        = A::RecordMut<'a>
    where
        Self: 'a;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        A::start(pc, memory, record);
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let data_inner = <A as AdapterTraceExecutor<F>>::read(&self.0, memory, instruction, record);

        core::array::from_fn(|i| {
            let mut out = [0u8; TOTAL_READ_SIZE];
            for (block_idx, block) in data_inner[i].iter().enumerate() {
                let start = block_idx * BLOCK_SIZE;
                out[start..start + BLOCK_SIZE].copy_from_slice(&block[..]);
            }
            out
        })
    }

    #[inline(always)]
    fn write(
        &self,
        _memory: &mut TracingMemory,
        _instruction: &Instruction<F>,
        _data: Self::WriteData,
        _record: &mut Self::RecordMut<'_>,
    ) {
        // Branch adapters don't write anything
    }
}
