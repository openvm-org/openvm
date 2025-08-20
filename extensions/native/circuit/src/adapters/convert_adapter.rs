use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, MinimalInstruction, VmAdapterAir,
    },
    system::{
        memory::{
            offline_checker::{
                MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
                MemoryWriteBytesAuxRecord,
            },
            online::TracingMemory,
            MemoryAddress, MemoryAuxColsFactory,
        },
        native_adapter::util::tracing_read_native,
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_MEMORY_AS,
};
use openvm_native_compiler::conversion::AS;
use openvm_rv32im_circuit::adapters::tracing_write;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

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

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct ConvertAdapterRecord<F, const READ_SIZE: usize, const WRITE_SIZE: usize> {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub a_ptr: F,
    pub b_ptr: F,

    pub read_aux: MemoryReadAuxRecord,
    pub write_aux: MemoryWriteBytesAuxRecord<WRITE_SIZE>,
}

#[derive(derive_new::new, Clone, Copy)]
pub struct ConvertAdapterExecutor<const READ_SIZE: usize, const WRITE_SIZE: usize>;

#[derive(derive_new::new)]
pub struct ConvertAdapterFiller<const READ_SIZE: usize, const WRITE_SIZE: usize>;

impl<F: PrimeField32, const READ_SIZE: usize, const WRITE_SIZE: usize> AdapterTraceExecutor<F>
    for ConvertAdapterExecutor<READ_SIZE, WRITE_SIZE>
{
    const WIDTH: usize = size_of::<ConvertAdapterCols<u8, READ_SIZE, WRITE_SIZE>>();
    type ReadData = [F; READ_SIZE];
    type WriteData = [u8; WRITE_SIZE];
    type RecordMut<'a> = &'a mut ConvertAdapterRecord<F, READ_SIZE, WRITE_SIZE>;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let &Instruction { b, e, .. } = instruction;
        debug_assert_eq!(e.as_canonical_u32(), AS::Native as u32);

        record.b_ptr = b;

        tracing_read_native(
            memory,
            b.as_canonical_u32(),
            &mut record.read_aux.prev_timestamp,
        )
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let &Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_MEMORY_AS);

        record.a_ptr = a;
        tracing_write(
            memory,
            RV32_MEMORY_AS,
            a.as_canonical_u32(),
            data,
            &mut record.write_aux.prev_timestamp,
            &mut record.write_aux.prev_data,
        );
    }
}

impl<F: PrimeField32, const READ_SIZE: usize, const WRITE_SIZE: usize> AdapterTraceFiller<F>
    for ConvertAdapterFiller<READ_SIZE, WRITE_SIZE>
{
    const WIDTH: usize = size_of::<ConvertAdapterCols<u8, READ_SIZE, WRITE_SIZE>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut row_slice: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &ConvertAdapterRecord<F, READ_SIZE, WRITE_SIZE> =
            unsafe { get_record_from_slice(&mut row_slice, ()) };
        let adapter_row: &mut ConvertAdapterCols<F, READ_SIZE, WRITE_SIZE> = row_slice.borrow_mut();

        // Writing in reverse order to avoid overwriting the `record`
        mem_helper.fill(
            record.read_aux.prev_timestamp,
            record.from_timestamp,
            adapter_row.reads_aux[0].as_mut(),
        );

        adapter_row.writes_aux[0]
            .set_prev_data(record.write_aux.prev_data.map(F::from_canonical_u8));
        mem_helper.fill(
            record.write_aux.prev_timestamp,
            record.from_timestamp + 1,
            adapter_row.writes_aux[0].as_mut(),
        );

        adapter_row.b_pointer = record.b_ptr;
        adapter_row.a_pointer = record.a_ptr;

        adapter_row.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
