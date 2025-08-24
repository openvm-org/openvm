use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxRecord, MemoryReadOrImmediateAuxCols},
            online::TracingMemory,
            MemoryAddress, MemoryAuxColsFactory,
        },
        native_adapter::util::tracing_read_or_imm_native,
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
#[derive(AlignedBorrow, Debug)]
pub struct BranchNativeAdapterReadCols<T> {
    pub address: MemoryAddress<T, T>,
    pub read_aux: MemoryReadOrImmediateAuxCols<T>,
}

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct BranchNativeAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub reads_aux: [BranchNativeAdapterReadCols<T>; 2],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct BranchNativeAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for BranchNativeAdapterAir {
    fn width(&self) -> usize {
        BranchNativeAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for BranchNativeAdapterAir {
    type Interface = BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, 2, 0, 1, 1>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &BranchNativeAdapterCols<_> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // check that d and e are in {0, 4}
        let d = cols.reads_aux[0].address.address_space;
        let e = cols.reads_aux[1].address.address_space;
        builder.assert_eq(
            d * (d - AB::F::from_canonical_u32(AS::Native as u32)),
            AB::F::ZERO,
        );
        builder.assert_eq(
            e * (e - AB::F::from_canonical_u32(AS::Native as u32)),
            AB::F::ZERO,
        );

        self.memory_bridge
            .read_or_immediate(
                cols.reads_aux[0].address,
                ctx.reads[0][0].clone(),
                timestamp_pp(),
                &cols.reads_aux[0].read_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read_or_immediate(
                cols.reads_aux[1].address,
                ctx.reads[1][0].clone(),
                timestamp_pp(),
                &cols.reads_aux[1].read_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.reads_aux[0].address.pointer.into(),
                    cols.reads_aux[1].address.pointer.into(),
                    ctx.instruction.immediate,
                    cols.reads_aux[0].address.address_space.into(),
                    cols.reads_aux[1].address.address_space.into(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &BranchNativeAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BranchNativeAdapterRecord<F> {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub ptrs: [F; 2],
    // Will set prev_timestamp to `u32::MAX` if the read is an immediate
    pub reads_aux: [MemoryReadAuxRecord; 2],
}

#[derive(derive_new::new, Clone, Copy)]
pub struct BranchNativeAdapterExecutor;

#[derive(derive_new::new)]
pub struct BranchNativeAdapterFiller;

impl<F> AdapterTraceExecutor<F> for BranchNativeAdapterExecutor
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<BranchNativeAdapterCols<u8>>();
    type ReadData = [F; 2];
    type WriteData = ();
    type RecordMut<'a> = &'a mut BranchNativeAdapterRecord<F>;

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
        let &Instruction { a, b, d, e, .. } = instruction;

        record.ptrs[0] = a;
        let rs1 = tracing_read_or_imm_native(memory, d, a, &mut record.reads_aux[0].prev_timestamp);
        record.ptrs[1] = b;
        let rs2 = tracing_read_or_imm_native(memory, e, b, &mut record.reads_aux[1].prev_timestamp);
        [rs1, rs2]
    }

    #[inline(always)]
    fn write(
        &self,
        _memory: &mut TracingMemory,
        _instruction: &Instruction<F>,
        _data: Self::WriteData,
        _record: &mut Self::RecordMut<'_>,
    ) {
        // This adapter doesn't write anything
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for BranchNativeAdapterFiller {
    const WIDTH: usize = size_of::<BranchNativeAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &BranchNativeAdapterRecord<F> =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut BranchNativeAdapterCols<F> = adapter_row.borrow_mut();

        // Writing in reverse order to avoid overwriting the `record`

        let native_as = F::from_canonical_u32(AS::Native as u32);
        for ((i, read_record), read_cols) in record
            .reads_aux
            .iter()
            .enumerate()
            .zip(adapter_row.reads_aux.iter_mut())
            .rev()
        {
            // previous timestamp is u32::MAX if the read is an immediate
            if read_record.prev_timestamp == u32::MAX {
                read_cols.read_aux.is_zero_aux = F::ZERO;
                read_cols.read_aux.is_immediate = F::ONE;
                mem_helper.fill(
                    0,
                    record.from_timestamp + i as u32,
                    read_cols.read_aux.as_mut(),
                );
                read_cols.address.pointer = record.ptrs[i];
                read_cols.address.address_space = F::ZERO;
            } else {
                read_cols.read_aux.is_zero_aux = native_as.inverse();
                read_cols.read_aux.is_immediate = F::ZERO;
                mem_helper.fill(
                    read_record.prev_timestamp,
                    record.from_timestamp + i as u32,
                    read_cols.read_aux.as_mut(),
                );
                read_cols.address.pointer = record.ptrs[i];
                read_cols.address.address_space = native_as;
            }
        }

        adapter_row.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
