use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        ExecutionBridge, ExecutionState, VmAdapterAir, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{
                MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
                MemoryWriteAuxRecord,
            },
            online::TracingMemory,
            MemoryAddress, MemoryAuxColsFactory,
        },
        native_adapter::util::{tracing_read_native, tracing_write_native},
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::{
    conversion::AS,
    NativeLoadStoreOpcode::{self, *},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

pub struct NativeLoadStoreInstruction<T> {
    pub is_valid: T,
    // Absolute opcode number
    pub opcode: T,
    pub is_loadw: T,
    pub is_storew: T,
    pub is_hint_storew: T,
}

pub struct NativeLoadStoreAdapterInterface<T, const NUM_CELLS: usize>(PhantomData<T>);

impl<T, const NUM_CELLS: usize> VmAdapterInterface<T>
    for NativeLoadStoreAdapterInterface<T, NUM_CELLS>
{
    type Reads = (T, [T; NUM_CELLS]);
    type Writes = [T; NUM_CELLS];
    type ProcessedInstruction = NativeLoadStoreInstruction<T>;
}

#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct NativeLoadStoreAdapterCols<T, const NUM_CELLS: usize> {
    pub from_state: ExecutionState<T>,
    pub a: T,
    pub b: T,
    pub c: T,

    pub data_write_pointer: T,

    pub pointer_read_aux_cols: MemoryReadAuxCols<T>,
    pub data_read_aux_cols: MemoryReadAuxCols<T>,
    pub data_write_aux_cols: MemoryWriteAuxCols<T, NUM_CELLS>,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct NativeLoadStoreAdapterAir<const NUM_CELLS: usize> {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
}

impl<F: Field, const NUM_CELLS: usize> BaseAir<F> for NativeLoadStoreAdapterAir<NUM_CELLS> {
    fn width(&self) -> usize {
        NativeLoadStoreAdapterCols::<F, NUM_CELLS>::width()
    }
}

impl<AB: InteractionBuilder, const NUM_CELLS: usize> VmAdapterAir<AB>
    for NativeLoadStoreAdapterAir<NUM_CELLS>
{
    type Interface = NativeLoadStoreAdapterInterface<AB::Expr, NUM_CELLS>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &NativeLoadStoreAdapterCols<_, NUM_CELLS> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = AB::Expr::from_canonical_usize(0);

        let is_valid = ctx.instruction.is_valid;
        let is_loadw = ctx.instruction.is_loadw;
        let is_storew = ctx.instruction.is_storew;
        let is_hint_storew = ctx.instruction.is_hint_storew;

        let native_as = AB::Expr::from_canonical_u32(AS::Native as u32);

        let ptr = ctx.reads.0;
        // Here we ignore ctx.reads.1 and we use `ctx.writes` as the data for both the write and the
        // second read (in the case of load/store when it exists).
        let data = ctx.writes;

        // first pointer read is always [c]_d
        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), cols.c),
                [ptr.clone()],
                timestamp + timestamp_delta.clone(),
                &cols.pointer_read_aux_cols,
            )
            .eval(builder, is_valid.clone());
        timestamp_delta += is_valid.clone();

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    native_as.clone(),
                    is_storew.clone() * cols.a + is_loadw.clone() * (ptr.clone() + cols.b),
                ),
                data.clone(),
                timestamp + timestamp_delta.clone(),
                &cols.data_read_aux_cols,
            )
            .eval(builder, is_valid.clone() - is_hint_storew.clone());
        timestamp_delta += is_valid.clone() - is_hint_storew.clone();

        builder.assert_eq(
            is_valid.clone() * cols.data_write_pointer,
            is_loadw.clone() * cols.a
                + (is_storew.clone() + is_hint_storew.clone()) * (ptr.clone() + cols.b),
        );

        self.memory_bridge
            .write(
                MemoryAddress::new(native_as.clone(), cols.data_write_pointer),
                data.clone(),
                timestamp + timestamp_delta.clone(),
                &cols.data_write_aux_cols,
            )
            .eval(builder, is_valid.clone());
        timestamp_delta += is_valid.clone();

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.a.into(),
                    cols.b.into(),
                    cols.c.into(),
                    native_as.clone(),
                    native_as.clone(),
                ],
                cols.from_state,
                timestamp_delta.clone(),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let local_cols: &NativeLoadStoreAdapterCols<_, NUM_CELLS> = local.borrow();
        local_cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct NativeLoadStoreAdapterRecord<F, const NUM_CELLS: usize> {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub a: F,
    pub b: F,
    pub c: F,
    pub write_ptr: F,

    pub ptr_read: MemoryReadAuxRecord,
    // Will set `prev_timestamp` to u32::MAX if `HINT_STOREW`
    pub data_read: MemoryReadAuxRecord,
    pub data_write: MemoryWriteAuxRecord<F, NUM_CELLS>,
}

#[derive(derive_new::new, Clone, Copy)]
pub struct NativeLoadStoreAdapterExecutor<const NUM_CELLS: usize> {
    offset: usize,
}

#[derive(derive_new::new)]
pub struct NativeLoadStoreAdapterFiller<const NUM_CELLS: usize>;

impl<F: PrimeField32, const NUM_CELLS: usize> AdapterTraceExecutor<F>
    for NativeLoadStoreAdapterExecutor<NUM_CELLS>
{
    const WIDTH: usize = std::mem::size_of::<NativeLoadStoreAdapterCols<u8, NUM_CELLS>>();
    type ReadData = (F, [F; NUM_CELLS]);
    type WriteData = [F; NUM_CELLS];
    type RecordMut<'a> = &'a mut NativeLoadStoreAdapterRecord<F, NUM_CELLS>;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp();
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), AS::Native as u32);
        debug_assert_eq!(e.as_canonical_u32(), AS::Native as u32);

        let local_opcode = NativeLoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        record.a = a;
        record.b = b;
        record.c = c;

        // Read the pointer value from memory
        let [read_cell] = tracing_read_native::<F, 1>(
            memory,
            c.as_canonical_u32(),
            &mut record.ptr_read.prev_timestamp,
        );

        let data_read_ptr = match local_opcode {
            LOADW => read_cell + record.b,
            STOREW | HINT_STOREW => record.a,
        }
        .as_canonical_u32();

        // It's easier to do this here than in `write`
        match local_opcode {
            LOADW => record.write_ptr = record.a,
            STOREW | HINT_STOREW => record.write_ptr = read_cell + record.b,
        }

        // Read data based on opcode
        let data_read: [F; NUM_CELLS] = match local_opcode {
            HINT_STOREW => {
                record.data_read.prev_timestamp = u32::MAX;
                [F::ZERO; NUM_CELLS]
            }
            LOADW | STOREW => {
                tracing_read_native(memory, data_read_ptr, &mut record.data_read.prev_timestamp)
            }
        };

        (read_cell, data_read)
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        _instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        // Write data to memory
        tracing_write_native(
            memory,
            record.write_ptr.as_canonical_u32(),
            data,
            &mut record.data_write.prev_timestamp,
            &mut record.data_write.prev_data,
        );
    }
}

impl<F: PrimeField32, const NUM_CELLS: usize> AdapterTraceFiller<F>
    for NativeLoadStoreAdapterFiller<NUM_CELLS>
{
    const WIDTH: usize = size_of::<NativeLoadStoreAdapterCols<u8, NUM_CELLS>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &NativeLoadStoreAdapterRecord<F, NUM_CELLS> =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut NativeLoadStoreAdapterCols<F, NUM_CELLS> = adapter_row.borrow_mut();

        // Writing in reverse order to avoid overwriting the `record`

        let is_hint_storew = record.data_read.prev_timestamp == u32::MAX;

        adapter_row
            .data_write_aux_cols
            .set_prev_data(record.data_write.prev_data);
        // Note, if `HINT_STOREW` we didn't do a data read and we didn't update the timestamp
        mem_helper.fill(
            record.data_write.prev_timestamp,
            record.from_timestamp + 2 - is_hint_storew as u32,
            adapter_row.data_write_aux_cols.as_mut(),
        );

        if !is_hint_storew {
            mem_helper.fill(
                record.data_read.prev_timestamp,
                record.from_timestamp + 1,
                adapter_row.data_read_aux_cols.as_mut(),
            );
        } else {
            mem_helper.fill_zero(adapter_row.data_read_aux_cols.as_mut());
        }

        mem_helper.fill(
            record.ptr_read.prev_timestamp,
            record.from_timestamp,
            adapter_row.pointer_read_aux_cols.as_mut(),
        );

        adapter_row.data_write_pointer = record.write_ptr;
        adapter_row.c = record.c;
        adapter_row.b = record.b;
        adapter_row.a = record.a;

        adapter_row.from_state.pc = F::from_canonical_u32(record.from_pc);
        adapter_row.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
    }
}
