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
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_native_compiler::conversion::AS;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

use crate::adapters::{memory_write, tracing_write};

use super::{memory_read, tracing_read};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ConvertAdapterCols<T, const READ_SIZE: usize, const WRITE_SIZE: usize> {
    pub from_state: ExecutionState<T>,
    pub a_pointer: T,
    pub b_pointer: T,
    pub writes_aux: [MemoryWriteAuxCols<T, WRITE_SIZE>; 1],
    pub reads_aux: [MemoryReadAuxCols<T>; 1],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct ConvertAdapterAir<const READ_SIZE: usize, const WRITE_SIZE: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field, const READ_SIZE: usize, const WRITE_SIZE: usize> BaseAir<F>
    for ConvertAdapterAir<READ_SIZE, WRITE_SIZE>
{
    fn width(&self) -> usize {
        ConvertAdapterCols::<F, READ_SIZE, WRITE_SIZE>::width()
    }
}

impl<AB: InteractionBuilder, const READ_SIZE: usize, const WRITE_SIZE: usize> VmAdapterAir<AB>
    for ConvertAdapterAir<READ_SIZE, WRITE_SIZE>
{
    type Interface =
        BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, 1, 1, READ_SIZE, WRITE_SIZE>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &ConvertAdapterCols<_, READ_SIZE, WRITE_SIZE> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        let d = AB::Expr::TWO;
        let e = AB::Expr::from_canonical_u32(AS::Native as u32);

        self.memory_bridge
            .read(
                MemoryAddress::new(e.clone(), cols.b_pointer),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &cols.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(d.clone(), cols.a_pointer),
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
                    AB::Expr::ZERO,
                    d,
                    e,
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &ConvertAdapterCols<_, READ_SIZE, WRITE_SIZE> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(derive_new::new)]
pub struct ConvertAdapterStep<const READ_SIZE: usize, const WRITE_SIZE: usize>;

impl<F, CTX, const READ_SIZE: usize, const WRITE_SIZE: usize> AdapterTraceStep<F, CTX>
    for ConvertAdapterStep<READ_SIZE, WRITE_SIZE>
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<ConvertAdapterCols<u8, READ_SIZE, WRITE_SIZE>>();
    type ReadData = [F; READ_SIZE];
    type WriteData = [F; WRITE_SIZE];
    type TraceContext<'a> = ();

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory<F>, adapter_row: &mut [F]) {
        let adapter_row: &mut ConvertAdapterCols<F, READ_SIZE, WRITE_SIZE> =
            adapter_row.borrow_mut();

        adapter_row.from_state.pc = F::from_canonical_u32(pc);
        adapter_row.from_state.timestamp = F::from_canonical_u32(memory.timestamp);
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData {
        let Instruction { b, e, .. } = instruction;

        debug_assert_eq!(e.as_canonical_u32(), AS::Native);

        let adapter_row: &mut ConvertAdapterCols<F, READ_SIZE, WRITE_SIZE> =
            adapter_row.borrow_mut();

        let read = tracing_read(
            memory,
            b.as_canonical_u32(),
            (&mut adapter_row.b_pointer, &mut adapter_row.reads_aux[0]),
        );
        read
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

        let adapter_row: &mut ConvertAdapterCols<F, READ_SIZE, WRITE_SIZE> =
            adapter_row.borrow_mut();

        tracing_write(
            memory,
            a.as_canonical_u32(),
            data,
            (&mut adapter_row.a_pointer, &mut adapter_row.writes_aux[0]),
        );
    }

    #[inline(always)]
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        _trace_ctx: Self::TraceContext<'_>,
        adapter_row: &mut [F],
    ) {
        let adapter_row: &mut ConvertAdapterCols<F, READ_SIZE, WRITE_SIZE> =
            adapter_row.borrow_mut();

        let mut timestamp = adapter_row.from_state.timestamp.as_canonical_u32();

        mem_helper.fill_from_prev(timestamp, adapter_row.reads_aux[0].as_mut());
        timestamp += 1;

        mem_helper.fill_from_prev(timestamp, adapter_row.writes_aux[0].as_mut());
    }
}

impl<F, const READ_SIZE: usize, const WRITE_SIZE: usize> AdapterExecutorE1<F>
    for ConvertAdapterStep<READ_SIZE, WRITE_SIZE>
where
    F: PrimeField32,
{
    type ReadData = [F; READ_SIZE];
    type WriteData = [F; WRITE_SIZE];

    #[inline(always)]
    fn read(&self, memory: &mut GuestMemory, instruction: &Instruction<F>) -> Self::ReadData {
        let Instruction { b, e, .. } = instruction;

        debug_assert_eq!(e.as_canonical_u32(), AS::Native);

        memory_read(memory, b.as_canonical_u32())
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut GuestMemory,
        instruction: &Instruction<F>,
        data: &Self::WriteData,
    ) {
        let Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), AS::Native);

        memory_write(memory, a.as_canonical_u32(), data);
    }
}
