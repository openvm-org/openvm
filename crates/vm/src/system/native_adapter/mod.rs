pub mod util;

use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionState,
        MinimalInstruction, VmAdapterAir,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_IMM_AS, NATIVE_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};
use util::{tracing_read_or_imm_native, tracing_write_native};

use super::memory::{online::TracingMemory, MemoryAuxColsFactory};
use crate::{
    arch::{get_record_from_slice, AdapterTraceExecutor, AdapterTraceFiller},
    system::memory::offline_checker::{MemoryReadAuxRecord, MemoryWriteAuxRecord},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeAdapterReadCols<T> {
    pub address: MemoryAddress<T, T>,
    pub read_aux: MemoryReadOrImmediateAuxCols<T>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeAdapterWriteCols<T> {
    pub address: MemoryAddress<T, T>,
    pub write_aux: MemoryWriteAuxCols<T, 1>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeAdapterCols<T, const R: usize, const W: usize> {
    pub from_state: ExecutionState<T>,
    pub reads_aux: [NativeAdapterReadCols<T>; R],
    pub writes_aux: [NativeAdapterWriteCols<T>; W],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct NativeAdapterAir<const R: usize, const W: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field, const R: usize, const W: usize> BaseAir<F> for NativeAdapterAir<R, W> {
    fn width(&self) -> usize {
        NativeAdapterCols::<F, R, W>::width()
    }
}

impl<AB: InteractionBuilder, const R: usize, const W: usize> VmAdapterAir<AB>
    for NativeAdapterAir<R, W>
{
    type Interface = BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, R, W, 1, 1>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &NativeAdapterCols<_, R, W> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        for (i, r_cols) in cols.reads_aux.iter().enumerate() {
            self.memory_bridge
                .read_or_immediate(
                    r_cols.address,
                    ctx.reads[i][0].clone(),
                    timestamp_pp(),
                    &r_cols.read_aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }
        for (i, w_cols) in cols.writes_aux.iter().enumerate() {
            self.memory_bridge
                .write(
                    w_cols.address,
                    ctx.writes[i].clone(),
                    timestamp_pp(),
                    &w_cols.write_aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let zero_address =
            || MemoryAddress::new(AB::Expr::from(AB::F::ZERO), AB::Expr::from(AB::F::ZERO));
        let f = |var_addr: MemoryAddress<AB::Var, AB::Var>| -> MemoryAddress<AB::Expr, AB::Expr> {
            MemoryAddress::new(var_addr.address_space.into(), var_addr.pointer.into())
        };

        let addr_a = if W >= 1 {
            f(cols.writes_aux[0].address)
        } else {
            zero_address()
        };
        let addr_b = if R >= 1 {
            f(cols.reads_aux[0].address)
        } else {
            zero_address()
        };
        let addr_c = if R >= 2 {
            f(cols.reads_aux[1].address)
        } else {
            zero_address()
        };
        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    addr_a.pointer,
                    addr_b.pointer,
                    addr_c.pointer,
                    addr_a.address_space,
                    addr_b.address_space,
                    addr_c.address_space,
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &NativeAdapterCols<_, R, W> = local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct NativeAdapterRecord<F, const R: usize, const W: usize> {
    pub from_pc: u32,
    pub from_timestamp: u32,

    // These are either a pointer to native memory or an immediate value
    pub read_ptr_or_imm: [F; R],
    // Will set prev_timestamp to `u32::MAX` if the read is from RV32_IMM_AS
    pub reads_aux: [MemoryReadAuxRecord; R],
    pub write_ptr: [F; W],
    pub writes_aux: [MemoryWriteAuxRecord<F, 1>; W],
}

/// R reads(R<=2), W writes(W<=1).
/// Operands: b for the first read, c for the second read, a for the first write.
/// If an operand is not used, its address space and pointer should be all 0.
#[derive(Clone, Debug)]
pub struct NativeAdapterExecutor<F, const R: usize, const W: usize> {
    _phantom: PhantomData<F>,
}

impl<F, const R: usize, const W: usize> Default for NativeAdapterExecutor<F, R, W> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F, const R: usize, const W: usize> AdapterTraceExecutor<F> for NativeAdapterExecutor<F, R, W>
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<NativeAdapterCols<u8, R, W>>();
    type ReadData = [[F; 1]; R];
    type WriteData = [[F; 1]; W];
    type RecordMut<'a> = &'a mut NativeAdapterRecord<F, R, W>;

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
        debug_assert!(R <= 2);
        let &Instruction { b, c, e, f, .. } = instruction;

        let mut reads = [[F::ZERO; 1]; R];
        record
            .read_ptr_or_imm
            .iter_mut()
            .enumerate()
            .zip(record.reads_aux.iter_mut())
            .for_each(|((i, ptr_or_imm), read_aux)| {
                *ptr_or_imm = if i == 0 { b } else { c };
                let addr_space = if i == 0 { e } else { f };
                reads[i][0] = tracing_read_or_imm_native(
                    memory,
                    addr_space,
                    *ptr_or_imm,
                    &mut read_aux.prev_timestamp,
                );
            });
        reads
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
        debug_assert!(W <= 1);
        debug_assert_eq!(d.as_canonical_u32(), NATIVE_AS);

        if W >= 1 {
            record.write_ptr[0] = a;
            tracing_write_native(
                memory,
                a.as_canonical_u32(),
                data[0],
                &mut record.writes_aux[0].prev_timestamp,
                &mut record.writes_aux[0].prev_data,
            );
        }
    }
}

impl<F: PrimeField32, const R: usize, const W: usize> AdapterTraceFiller<F>
    for NativeAdapterExecutor<F, R, W>
{
    const WIDTH: usize = size_of::<NativeAdapterCols<u8, R, W>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &NativeAdapterRecord<F, R, W> =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut NativeAdapterCols<_, R, W> = adapter_row.borrow_mut();
        // Writing in reverse order to avoid overwriting the `record`
        if W >= 1 {
            adapter_row.writes_aux[0]
                .write_aux
                .set_prev_data(record.writes_aux[0].prev_data);
            mem_helper.fill(
                record.writes_aux[0].prev_timestamp,
                record.from_timestamp + R as u32,
                adapter_row.writes_aux[0].write_aux.as_mut(),
            );
            adapter_row.writes_aux[0].address.pointer = record.write_ptr[0];
            adapter_row.writes_aux[0].address.address_space = F::from_canonical_u32(NATIVE_AS);
        }

        adapter_row
            .reads_aux
            .iter_mut()
            .enumerate()
            .zip(record.reads_aux.iter().zip(record.read_ptr_or_imm.iter()))
            .rev()
            .for_each(|((i, read_cols), (read_record, ptr_or_imm))| {
                if read_record.prev_timestamp == u32::MAX {
                    read_cols.read_aux.is_zero_aux = F::ZERO;
                    read_cols.read_aux.is_immediate = F::ONE;
                    mem_helper.fill(
                        0,
                        record.from_timestamp + i as u32,
                        read_cols.read_aux.as_mut(),
                    );
                    read_cols.address.pointer = *ptr_or_imm;
                    read_cols.address.address_space = F::from_canonical_u32(RV32_IMM_AS);
                } else {
                    read_cols.read_aux.is_zero_aux = F::from_canonical_u32(NATIVE_AS).inverse();
                    read_cols.read_aux.is_immediate = F::ZERO;
                    mem_helper.fill(
                        read_record.prev_timestamp,
                        record.from_timestamp + i as u32,
                        read_cols.read_aux.as_mut(),
                    );
                    read_cols.address.pointer = *ptr_or_imm;
                    read_cols.address.address_space = F::from_canonical_u32(NATIVE_AS);
                }
            });

        adapter_row.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
