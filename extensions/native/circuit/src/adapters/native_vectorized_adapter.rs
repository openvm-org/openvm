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
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_native_compiler::conversion::AS;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

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

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct NativeVectorizedAdapterRecord<F, const N: usize> {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub a_ptr: F,
    pub b_ptr: F,
    pub c_ptr: F,
    pub reads_aux: [MemoryReadAuxRecord; 2],
    pub write_aux: MemoryWriteAuxRecord<F, N>,
}

#[derive(derive_new::new, Clone, Copy)]
pub struct NativeVectorizedAdapterExecutor<const N: usize>;

#[derive(derive_new::new)]
pub struct NativeVectorizedAdapterFiller<const N: usize>;

impl<F: PrimeField32, const N: usize> AdapterTraceExecutor<F>
    for NativeVectorizedAdapterExecutor<N>
{
    const WIDTH: usize = size_of::<NativeVectorizedAdapterCols<u8, N>>();
    type ReadData = [[F; N]; 2];
    type WriteData = [F; N];
    type RecordMut<'a> = &'a mut NativeVectorizedAdapterRecord<F, N>;

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
        let &Instruction { b, c, d, e, .. } = instruction;
        debug_assert_eq!(d.as_canonical_u32(), AS::Native as u32);
        debug_assert_eq!(e.as_canonical_u32(), AS::Native as u32);

        record.b_ptr = b;
        let b_val = tracing_read_native(
            memory,
            b.as_canonical_u32(),
            &mut record.reads_aux[0].prev_timestamp,
        );
        record.c_ptr = c;
        let c_val = tracing_read_native(
            memory,
            c.as_canonical_u32(),
            &mut record.reads_aux[1].prev_timestamp,
        );

        [b_val, c_val]
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

        debug_assert_eq!(d.as_canonical_u32(), AS::Native as u32);

        record.a_ptr = a;
        tracing_write_native(
            memory,
            a.as_canonical_u32(),
            data,
            &mut record.write_aux.prev_timestamp,
            &mut record.write_aux.prev_data,
        );
    }
}

impl<F: PrimeField32, const N: usize> AdapterTraceFiller<F> for NativeVectorizedAdapterFiller<N> {
    const WIDTH: usize = size_of::<NativeVectorizedAdapterCols<u8, N>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &NativeVectorizedAdapterRecord<F, N> =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut NativeVectorizedAdapterCols<F, N> = adapter_row.borrow_mut();

        // Writing in reverse order to avoid overwriting the `record`
        adapter_row.writes_aux[0].set_prev_data(record.write_aux.prev_data);
        mem_helper.fill(
            record.write_aux.prev_timestamp,
            record.from_timestamp + 2,
            adapter_row.writes_aux[0].as_mut(),
        );

        adapter_row
            .reads_aux
            .iter_mut()
            .enumerate()
            .zip(record.reads_aux.iter())
            .rev()
            .for_each(|((i, read_cols), read_record)| {
                mem_helper.fill(
                    read_record.prev_timestamp,
                    record.from_timestamp + i as u32,
                    read_cols.as_mut(),
                );
            });

        adapter_row.c_pointer = record.c_ptr;
        adapter_row.b_pointer = record.b_ptr;
        adapter_row.a_pointer = record.a_ptr;

        adapter_row.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
