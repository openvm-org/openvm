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
                MemoryBridge, MemoryReadAuxRecord, MemoryReadOrImmediateAuxCols,
                MemoryWriteAuxCols, MemoryWriteAuxRecord,
            },
            online::TracingMemory,
            MemoryAddress, MemoryAuxColsFactory,
        },
        native_adapter::util::{tracing_read_or_imm_native, tracing_write_native},
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
pub struct AluNativeAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub a_pointer: T,
    pub b_pointer: T,
    pub c_pointer: T,
    pub e_as: T,
    pub f_as: T,
    pub reads_aux: [MemoryReadOrImmediateAuxCols<T>; 2],
    pub write_aux: MemoryWriteAuxCols<T, 1>,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct AluNativeAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for AluNativeAdapterAir {
    fn width(&self) -> usize {
        AluNativeAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for AluNativeAdapterAir {
    type Interface = BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, 2, 1, 1, 1>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &AluNativeAdapterCols<_> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        let native_as = AB::Expr::from_canonical_u32(AS::Native as u32);

        // TODO: we assume address space is either 0 or 4, should we add a
        //       constraint for that?
        self.memory_bridge
            .read_or_immediate(
                MemoryAddress::new(cols.e_as, cols.b_pointer),
                ctx.reads[0][0].clone(),
                timestamp_pp(),
                &cols.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read_or_immediate(
                MemoryAddress::new(cols.f_as, cols.c_pointer),
                ctx.reads[1][0].clone(),
                timestamp_pp(),
                &cols.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(native_as.clone(), cols.a_pointer),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &cols.write_aux,
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
                    cols.e_as.into(),
                    cols.f_as.into(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &AluNativeAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct AluNativeAdapterRecord<F> {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub a_ptr: F,
    pub b: F,
    pub c: F,

    // Will set prev_timestamp to `u32::MAX` if the read is an immediate
    pub reads_aux: [MemoryReadAuxRecord; 2],
    pub write_aux: MemoryWriteAuxRecord<F, 1>,
}

#[derive(derive_new::new, Clone, Copy)]
pub struct AluNativeAdapterExecutor;

#[derive(derive_new::new)]
pub struct AluNativeAdapterFiller;

impl<F: PrimeField32> AdapterTraceExecutor<F> for AluNativeAdapterExecutor {
    const WIDTH: usize = size_of::<AluNativeAdapterCols<u8>>();
    type ReadData = [F; 2];
    type WriteData = [F; 1];
    type RecordMut<'a> = &'a mut AluNativeAdapterRecord<F>;

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
        let &Instruction { b, c, e, f, .. } = instruction;

        record.b = b;
        let rs1 = tracing_read_or_imm_native(memory, e, b, &mut record.reads_aux[0].prev_timestamp);
        record.c = c;
        let rs2 = tracing_read_or_imm_native(memory, f, c, &mut record.reads_aux[1].prev_timestamp);
        [rs1, rs2]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let &Instruction { a, .. } = instruction;

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

impl<F: PrimeField32> AdapterTraceFiller<F> for AluNativeAdapterFiller {
    const WIDTH: usize = size_of::<AluNativeAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &AluNativeAdapterRecord<F> =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut AluNativeAdapterCols<F> = adapter_row.borrow_mut();

        // Writing in reverse order to avoid overwriting the `record`
        adapter_row
            .write_aux
            .set_prev_data(record.write_aux.prev_data);
        mem_helper.fill(
            record.write_aux.prev_timestamp,
            record.from_timestamp + 2,
            adapter_row.write_aux.as_mut(),
        );

        let native_as = F::from_canonical_u32(AS::Native as u32);
        for ((i, read_record), read_cols) in record
            .reads_aux
            .iter()
            .enumerate()
            .zip(adapter_row.reads_aux.iter_mut())
            .rev()
        {
            let as_col = if i == 0 {
                &mut adapter_row.e_as
            } else {
                &mut adapter_row.f_as
            };
            // previous timestamp is u32::MAX if the read is an immediate
            if read_record.prev_timestamp == u32::MAX {
                read_cols.is_zero_aux = F::ZERO;
                read_cols.is_immediate = F::ONE;
                mem_helper.fill(0, record.from_timestamp + i as u32, read_cols.as_mut());
                *as_col = F::ZERO;
            } else {
                read_cols.is_zero_aux = native_as.inverse();
                read_cols.is_immediate = F::ZERO;
                mem_helper.fill(
                    read_record.prev_timestamp,
                    record.from_timestamp + i as u32,
                    read_cols.as_mut(),
                );
                *as_col = native_as;
            }
        }

        adapter_row.c_pointer = record.c;
        adapter_row.b_pointer = record.b;
        adapter_row.a_pointer = record.a_ptr;

        adapter_row.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
