#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]
use openvm_circuit::{
    self,
    arch::{
        AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller, ExecutionBridge,
        ImmInstruction, InitFileGenerator, MinimalInstruction, SystemConfig, VmAdapterAir,
        VmAdapterInterface, VmAirWrapper, VmChipWrapper,
    },
    system::{
        memory::{
            offline_checker::MemoryBridge,
            online::TracingMemory,
            MemoryAuxColsFactory,
        },
        SystemExecutor,
    },
};
use openvm_circuit_derive::{PreflightExecutor, VmConfig};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_rv32_adapters::{
    Rv32HeapBranchAdapterAirGeneric, Rv32HeapBranchAdapterExecutorGeneric,
    Rv32HeapBranchAdapterFillerGeneric, Rv32VecHeapAdapterAir, Rv32VecHeapAdapterExecutor,
    Rv32VecHeapAdapterFiller,
};
use openvm_rv32im_circuit::{
    adapters::{INT256_NUM_LIMBS, RV32_CELL_BITS},
    BaseAluCoreAir, BaseAluExecutor, BaseAluFiller, BranchEqualCoreAir, BranchEqualExecutor,
    BranchEqualFiller, BranchLessThanCoreAir, BranchLessThanExecutor, BranchLessThanFiller,
    LessThanCoreAir, LessThanExecutor, LessThanFiller, MultiplicationCoreAir,
    MultiplicationExecutor, MultiplicationFiller, Rv32I, Rv32IExecutor, Rv32Io, Rv32IoExecutor,
    Rv32M, Rv32MExecutor, ShiftCoreAir, ShiftExecutor, ShiftFiller,
};
use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeField32},
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

mod extension;
pub use extension::*;

mod base_alu;
mod branch_eq;
mod branch_lt;
pub(crate) mod common;
mod less_than;
mod mult;
mod shift;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

use crate::common::{INT256_BLOCKS_PER_ACCESS, INT256_CHUNK_BYTES};

type BigintHeapAdapterInner =
    Rv32VecHeapAdapterExecutor<2, INT256_BLOCKS_PER_ACCESS, INT256_BLOCKS_PER_ACCESS, INT256_CHUNK_BYTES, INT256_CHUNK_BYTES>;
type BigintHeapAdapterFillerInner =
    Rv32VecHeapAdapterFiller<2, INT256_BLOCKS_PER_ACCESS, INT256_BLOCKS_PER_ACCESS, INT256_CHUNK_BYTES, INT256_CHUNK_BYTES>;

type BigintBranchAdapterInner =
    Rv32HeapBranchAdapterExecutorGeneric<2, INT256_BLOCKS_PER_ACCESS, INT256_CHUNK_BYTES>;
type BigintBranchAdapterFillerInner =
    Rv32HeapBranchAdapterFillerGeneric<2, INT256_BLOCKS_PER_ACCESS, INT256_CHUNK_BYTES>;

fn chunk_expr<T>(
    word: [T; INT256_NUM_LIMBS],
) -> [[T; INT256_CHUNK_BYTES]; INT256_BLOCKS_PER_ACCESS] {
    let mut iter = word.into_iter();
    std::array::from_fn(|_| std::array::from_fn(|_| iter.next().expect("chunk size")))
}

#[inline(always)]
fn flatten_int256(
    chunks: [[u8; INT256_CHUNK_BYTES]; INT256_BLOCKS_PER_ACCESS],
) -> [u8; INT256_NUM_LIMBS] {
    let mut word = [0u8; INT256_NUM_LIMBS];
    for (block_idx, block) in chunks.into_iter().enumerate() {
        let start = block_idx * INT256_CHUNK_BYTES;
        word[start..start + INT256_CHUNK_BYTES].copy_from_slice(&block);
    }
    word
}

#[inline(always)]
fn chunk_int256(
    word: [u8; INT256_NUM_LIMBS],
) -> [[u8; INT256_CHUNK_BYTES]; INT256_BLOCKS_PER_ACCESS] {
    let mut chunks = [[0u8; INT256_CHUNK_BYTES]; INT256_BLOCKS_PER_ACCESS];
    for (block_idx, chunk) in chunks.iter_mut().enumerate() {
        let start = block_idx * INT256_CHUNK_BYTES;
        chunk.copy_from_slice(&word[start..start + INT256_CHUNK_BYTES]);
    }
    chunks
}

#[derive(Clone)]
pub struct BigintAccessReads<T>(
    pub [[[T; INT256_CHUNK_BYTES]; INT256_BLOCKS_PER_ACCESS]; 2],
);

#[derive(Clone)]
pub struct BigintAccessWrites<T>(pub [[T; INT256_CHUNK_BYTES]; INT256_BLOCKS_PER_ACCESS]);

impl<T> BigintAccessReads<T> {
    fn into_inner(self) -> [[[T; INT256_CHUNK_BYTES]; INT256_BLOCKS_PER_ACCESS]; 2] {
        self.0
    }
}

impl<T> BigintAccessWrites<T> {
    fn into_inner(self) -> [[T; INT256_CHUNK_BYTES]; INT256_BLOCKS_PER_ACCESS] {
        self.0
    }
}

impl<T> From<[[T; INT256_NUM_LIMBS]; 2]> for BigintAccessReads<T> {
    fn from(value: [[T; INT256_NUM_LIMBS]; 2]) -> Self {
        let [first, second] = value;
        Self([chunk_expr(first), chunk_expr(second)])
    }
}

impl<T> From<[[T; INT256_NUM_LIMBS]; 1]> for BigintAccessWrites<T> {
    fn from(value: [[T; INT256_NUM_LIMBS]; 1]) -> Self {
        let [word] = value;
        Self(chunk_expr(word))
    }
}

pub struct BigintHeapAdapterInterface<T>(PhantomData<T>);

impl<T> VmAdapterInterface<T> for BigintHeapAdapterInterface<T> {
    type Reads = BigintAccessReads<T>;
    type Writes = BigintAccessWrites<T>;
    type ProcessedInstruction = MinimalInstruction<T>;
}

pub struct BigintBranchAdapterInterface<T>(PhantomData<T>);

impl<T> VmAdapterInterface<T> for BigintBranchAdapterInterface<T> {
    type Reads = BigintAccessReads<T>;
    type Writes = ();
    type ProcessedInstruction = ImmInstruction<T>;
}

#[derive(Clone, Copy)]
pub struct BigintHeapAdapterAir {
    inner: Rv32VecHeapAdapterAir<
        2,
        INT256_BLOCKS_PER_ACCESS,
        INT256_BLOCKS_PER_ACCESS,
        INT256_CHUNK_BYTES,
        INT256_CHUNK_BYTES,
    >,
}

impl BigintHeapAdapterAir {
    pub fn new(
        execution_bridge: ExecutionBridge,
        memory_bridge: MemoryBridge,
        bus: BitwiseOperationLookupBus,
        address_bits: usize,
    ) -> Self {
        Self {
            inner: Rv32VecHeapAdapterAir::new(
                execution_bridge,
                memory_bridge,
                bus,
                address_bits,
            ),
        }
    }
}

impl<F: Field> BaseAir<F> for BigintHeapAdapterAir {
    fn width(&self) -> usize {
        <Rv32VecHeapAdapterAir<
            2,
            INT256_BLOCKS_PER_ACCESS,
            INT256_BLOCKS_PER_ACCESS,
            INT256_CHUNK_BYTES,
            INT256_CHUNK_BYTES,
        > as BaseAir<F>>::width(&self.inner)
    }
}

impl<AB> VmAdapterAir<AB> for BigintHeapAdapterAir
where
    AB: InteractionBuilder,
{
    type Interface = BigintHeapAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let inner_ctx = AdapterAirContext {
            to_pc: ctx.to_pc,
            reads: ctx.reads.into_inner(),
            writes: ctx.writes.into_inner(),
            instruction: ctx.instruction,
        };
        self.inner.eval(builder, local, inner_ctx);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        <Rv32VecHeapAdapterAir<
            2,
            INT256_BLOCKS_PER_ACCESS,
            INT256_BLOCKS_PER_ACCESS,
            INT256_CHUNK_BYTES,
            INT256_CHUNK_BYTES,
        > as VmAdapterAir<AB>>::get_from_pc(&self.inner, local)
    }
}

#[derive(Clone, Copy)]
pub struct BigintBranchAdapterAir {
    inner:
        Rv32HeapBranchAdapterAirGeneric<2, INT256_BLOCKS_PER_ACCESS, INT256_CHUNK_BYTES>,
}

impl BigintBranchAdapterAir {
    pub fn new(
        execution_bridge: ExecutionBridge,
        memory_bridge: MemoryBridge,
        bus: BitwiseOperationLookupBus,
        address_bits: usize,
    ) -> Self {
        Self {
            inner: Rv32HeapBranchAdapterAirGeneric::new(
                execution_bridge,
                memory_bridge,
                bus,
                address_bits,
            ),
        }
    }
}

impl<F: Field> BaseAir<F> for BigintBranchAdapterAir {
    fn width(&self) -> usize {
        <Rv32HeapBranchAdapterAirGeneric<
            2,
            INT256_BLOCKS_PER_ACCESS,
            INT256_CHUNK_BYTES,
        > as BaseAir<F>>::width(&self.inner)
    }
}

impl<AB> VmAdapterAir<AB> for BigintBranchAdapterAir
where
    AB: InteractionBuilder,
{
    type Interface = BigintBranchAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let inner_ctx = AdapterAirContext {
            to_pc: ctx.to_pc,
            reads: ctx.reads.into_inner(),
            writes: [],
            instruction: ctx.instruction,
        };
        self.inner.eval(builder, local, inner_ctx);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        <Rv32HeapBranchAdapterAirGeneric<
            2,
            INT256_BLOCKS_PER_ACCESS,
            INT256_CHUNK_BYTES,
        > as VmAdapterAir<AB>>::get_from_pc(&self.inner, local)
    }
}

#[derive(Clone, Copy)]
pub struct BigintHeapAdapterExecutor {
    inner: BigintHeapAdapterInner,
}

impl BigintHeapAdapterExecutor {
    pub fn new(pointer_max_bits: usize) -> Self {
        Self {
            inner: BigintHeapAdapterInner::new(pointer_max_bits),
        }
    }
}

impl<F: PrimeField32> AdapterTraceExecutor<F> for BigintHeapAdapterExecutor {
    const WIDTH: usize = <BigintHeapAdapterInner as AdapterTraceExecutor<F>>::WIDTH;
    type ReadData = [[u8; INT256_NUM_LIMBS]; 2];
    type WriteData = [[u8; INT256_NUM_LIMBS]; 1];
    type RecordMut<'a> = <BigintHeapAdapterInner as AdapterTraceExecutor<F>>::RecordMut<'a>;

    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        <BigintHeapAdapterInner as AdapterTraceExecutor<F>>::start(pc, memory, record);
    }

    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let chunked = <BigintHeapAdapterInner as AdapterTraceExecutor<F>>::read(
            &self.inner,
            memory,
            instruction,
            record,
        );
        chunked.map(flatten_int256)
    }

    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let [word] = data;
        let chunked = chunk_int256(word);
        <BigintHeapAdapterInner as AdapterTraceExecutor<F>>::write(
            &self.inner,
            memory,
            instruction,
            chunked,
            record,
        );
    }
}

#[derive(Clone)]
pub struct BigintHeapAdapterFiller {
    inner: BigintHeapAdapterFillerInner,
}

impl BigintHeapAdapterFiller {
    pub fn new(
        pointer_max_bits: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> Self {
        Self {
            inner: BigintHeapAdapterFillerInner::new(pointer_max_bits, bitwise_lookup_chip),
        }
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for BigintHeapAdapterFiller {
    const WIDTH: usize = <BigintHeapAdapterFillerInner as AdapterTraceFiller<F>>::WIDTH;

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, adapter_row: &mut [F]) {
        self.inner.fill_trace_row(mem_helper, adapter_row);
    }
}

#[derive(Clone, Copy)]
pub struct BigintBranchAdapterExecutor {
    inner: BigintBranchAdapterInner,
}

impl BigintBranchAdapterExecutor {
    pub fn new(pointer_max_bits: usize) -> Self {
        Self {
            inner: BigintBranchAdapterInner::new(pointer_max_bits),
        }
    }
}

impl<F: PrimeField32> AdapterTraceExecutor<F> for BigintBranchAdapterExecutor {
    const WIDTH: usize = <BigintBranchAdapterInner as AdapterTraceExecutor<F>>::WIDTH;
    type ReadData = [[u8; INT256_NUM_LIMBS]; 2];
    type WriteData = ();
    type RecordMut<'a> = <BigintBranchAdapterInner as AdapterTraceExecutor<F>>::RecordMut<'a>;

    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        <BigintBranchAdapterInner as AdapterTraceExecutor<F>>::start(pc, memory, record);
    }

    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let chunked = <BigintBranchAdapterInner as AdapterTraceExecutor<F>>::read(
            &self.inner,
            memory,
            instruction,
            record,
        );
        chunked.map(flatten_int256)
    }

    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        <BigintBranchAdapterInner as AdapterTraceExecutor<F>>::write(
            &self.inner,
            memory,
            instruction,
            data,
            record,
        );
    }
}

#[derive(Clone)]
pub struct BigintBranchAdapterFiller {
    inner: BigintBranchAdapterFillerInner,
}

impl BigintBranchAdapterFiller {
    pub fn new(
        pointer_max_bits: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> Self {
        Self {
            inner: BigintBranchAdapterFillerInner::new(pointer_max_bits, bitwise_lookup_chip),
        }
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for BigintBranchAdapterFiller {
    const WIDTH: usize = <BigintBranchAdapterFillerInner as AdapterTraceFiller<F>>::WIDTH;

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, adapter_row: &mut [F]) {
        self.inner.fill_trace_row(mem_helper, adapter_row);
    }
}

/// BaseAlu256
pub type Rv32BaseAlu256Air = VmAirWrapper<
    BigintHeapAdapterAir,
    BaseAluCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32BaseAlu256Executor(
    BaseAluExecutor<
        BigintHeapAdapterExecutor,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);
pub type Rv32BaseAlu256Chip<F> = VmChipWrapper<
    F,
    BaseAluFiller<
        BigintHeapAdapterFiller,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// LessThan256
pub type Rv32LessThan256Air = VmAirWrapper<
    BigintHeapAdapterAir,
    LessThanCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32LessThan256Executor(
    LessThanExecutor<
        BigintHeapAdapterExecutor,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);
pub type Rv32LessThan256Chip<F> = VmChipWrapper<
    F,
    LessThanFiller<
        BigintHeapAdapterFiller,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// Multiplication256
pub type Rv32Multiplication256Air = VmAirWrapper<
    BigintHeapAdapterAir,
    MultiplicationCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32Multiplication256Executor(
    MultiplicationExecutor<
        BigintHeapAdapterExecutor,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);
pub type Rv32Multiplication256Chip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<
        BigintHeapAdapterFiller,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// Shift256
pub type Rv32Shift256Air = VmAirWrapper<
    BigintHeapAdapterAir,
    ShiftCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32Shift256Executor(
    ShiftExecutor<
        BigintHeapAdapterExecutor,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);
pub type Rv32Shift256Chip<F> = VmChipWrapper<
    F,
    ShiftFiller<
        BigintHeapAdapterFiller,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// BranchEqual256
pub type Rv32BranchEqual256Air = VmAirWrapper<
    BigintBranchAdapterAir,
    BranchEqualCoreAir<INT256_NUM_LIMBS>,
>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32BranchEqual256Executor(
    BranchEqualExecutor<BigintBranchAdapterExecutor, INT256_NUM_LIMBS>,
);
pub type Rv32BranchEqual256Chip<F> = VmChipWrapper<
    F,
    BranchEqualFiller<BigintBranchAdapterFiller, INT256_NUM_LIMBS>,
>;

/// BranchLessThan256
pub type Rv32BranchLessThan256Air = VmAirWrapper<
    BigintBranchAdapterAir,
    BranchLessThanCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32BranchLessThan256Executor(
    BranchLessThanExecutor<
        BigintBranchAdapterExecutor,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);
pub type Rv32BranchLessThan256Chip<F> = VmChipWrapper<
    F,
    BranchLessThanFiller<
        BigintBranchAdapterFiller,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Int256Rv32Config {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub bigint: Int256,
}

// Default implementation uses no init file
impl InitFileGenerator for Int256Rv32Config {}

impl Default for Int256Rv32Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            rv32i: Rv32I,
            rv32m: Rv32M::default(),
            io: Rv32Io,
            bigint: Int256::default(),
        }
    }
}
