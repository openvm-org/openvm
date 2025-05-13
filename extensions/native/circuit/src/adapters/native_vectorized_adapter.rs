use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, BasicAdapterInterface,
        ExecutionBridge, ExecutionState, MinimalInstruction, VmAdapterAir,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        online::{GuestMemory, TracingMemory},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupChip;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_native_compiler::conversion::AS;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

use super::{memory_read, memory_write, tracing_read, tracing_write};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeVectorizedAdapterCols<T, const N: usize> {
    pub from_state: ExecutionState<T>,
    pub a_pointer: T,
    pub b_pointer: T,
    pub c_pointer: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
    pub writes_aux: [MemoryWriteAuxCols<T, N>; 1],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct NativeVectorizedAdapterAir<const N: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field, const N: usize> BaseAir<F> for NativeVectorizedAdapterAir<N> {
    fn width(&self) -> usize {
        NativeVectorizedAdapterCols::<F, N>::width()
    }
}

impl<AB: InteractionBuilder, const N: usize> VmAdapterAir<AB> for NativeVectorizedAdapterAir<N> {
    type Interface = BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, 2, 1, N, N>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &NativeVectorizedAdapterCols<_, N> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        let native_as = AB::Expr::from_canonical_u32(AS::Native as u32);

        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), cols.b_pointer),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &cols.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), cols.c_pointer),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &cols.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(native_as.clone(), cols.a_pointer),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &cols.writes_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.a_pointer.into(),
                    cols.b_pointer.into(),
                    cols.c_pointer.into(),
                    native_as.clone(),
                    native_as.clone(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &NativeVectorizedAdapterCols<_, N> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(derive_new::new)]
pub struct NativeVectorizedAdapterChip<const N: usize>;

impl<F, CTX, const N: usize> AdapterTraceStep<F, CTX> for NativeVectorizedAdapterChip<N>
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<NativeVectorizedAdapterCols<u8, N>>();
    type ReadData = [[F; N]; 2];
    type WriteData = [F; N];
    type TraceContext<'a> = &'a BitwiseOperationLookupChip<64>;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory<F>, adapter_row: &mut [F]) {
        let adapter_row: &mut NativeVectorizedAdapterCols<F, N> = adapter_row.borrow_mut();

        adapter_row.from_state.pc = F::from_canonical_u32(pc);
        adapter_row.from_state.timestamp = F::from_canonical_u32(memory.timestamp());
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData {
        let Instruction { a, b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), AS::Native);
        debug_assert_eq!(e.as_canonical_u32(), AS::Native);

        let adapter_row: &mut NativeVectorizedAdapterCols<F, N> = adapter_row.borrow_mut();

        let y_val = tracing_read(
            memory,
            b.as_canonical_u32(),
            (&mut adapter_row.b_pointer, &mut adapter_row.reads_aux[0]),
        );
        let z_val = tracing_read(
            memory,
            c.as_canonical_u32(),
            (&mut adapter_row.c_pointer, &mut adapter_row.reads_aux[1]),
        );

        [y_val, z_val]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        let Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), AS::Native);

        let adapter_row: &mut NativeVectorizedAdapterCols<F, N> = adapter_row.borrow_mut();

        tracing_write(
            memory,
            a.as_canonical_u32(),
            data,
            (&mut adapter_row.c_pointer, &mut adapter_row.writes_aux[0]),
        );
    }

    #[inline(always)]
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        bitwise_lookup_chip: Self::TraceContext<'_>,
        adapter_row: &mut [F],
    ) {
        let adapter_row: &mut NativeVectorizedAdapterCols<F, N> = adapter_row.borrow_mut();

        let mut timestamp = adapter_row.from_state.timestamp.as_canonical_u32();

        mem_helper.fill_read_aux(timestamp, &mut adapter_row.reads_aux[0]);
        timestamp += 1;

        mem_helper.fill_read_aux(timestamp, &mut adapter_row.reads_aux[1]);
        timestamp += 1;

        mem_helper.fill_write_aux(timestamp, &mut adapter_row.writes_aux[0]);
    }
}

impl<F, const N: usize> AdapterExecutorE1<F> for NativeVectorizedAdapterChip<N>
where
    F: PrimeField32,
{
    type ReadData = [[F; N]; 2];
    type WriteData = [F; N];

    #[inline(always)]
    fn read<Mem>(&self, memory: &mut Mem, instruction: &Instruction<F>) -> Self::ReadData
    where
        Mem: GuestMemory,
    {
        let Instruction { b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), AS::Native);
        debug_assert_eq!(e.as_canonical_u32(), AS::Native);

        let y_val: [F; N] = memory_read(memory, b.as_canonical_u32());
        let z_val: [F; N] = memory_read(memory, c.as_canonical_u32());

        [y_val, z_val]
    }

    #[inline(always)]
    fn write<Mem>(&self, memory: &mut Mem, instruction: &Instruction<F>, data: &Self::WriteData)
    where
        Mem: GuestMemory,
    {
        let Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), AS::Native);

        memory_write(memory, a.as_canonical_u32(), data);
    }
}
