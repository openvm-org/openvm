use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::{
        execution_mode::E1E2ExecutionCtx, AdapterAirContext, AdapterExecutorE1, AdapterTraceStep,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, MinimalInstruction, VmAdapterAir,
        VmStateMut,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
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

use crate::adapters::{
    memory_read_or_imm_native_from_state, memory_write_native_from_state,
    tracing_read_or_imm_native,
};

use super::tracing_write_native;

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

#[derive(derive_new::new)]
pub struct AluNativeAdapterStep;

impl<F, CTX> AdapterTraceStep<F, CTX> for AluNativeAdapterStep
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<AluNativeAdapterCols<u8>>();
    type ReadData = [F; 2];
    type WriteData = [F; 1];
    type TraceContext<'a> = ();

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory<F>, adapter_row: &mut [F]) {
        let adapter_row: &mut AluNativeAdapterCols<F> = adapter_row.borrow_mut();

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
        let &Instruction { b, c, e, f, .. } = instruction;

        let adapter_row: &mut AluNativeAdapterCols<F> = adapter_row.borrow_mut();

        adapter_row.b_pointer = b;
        let rs1 = tracing_read_or_imm_native(
            memory,
            e.as_canonical_u32(),
            b,
            &mut adapter_row.e_as,
            &mut adapter_row.reads_aux[0],
        );
        adapter_row.c_pointer = c;
        let rs2 = tracing_read_or_imm_native(
            memory,
            f.as_canonical_u32(),
            c,
            &mut adapter_row.f_as,
            &mut adapter_row.reads_aux[1],
        );
        [rs1, rs2]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        let &Instruction { a, .. } = instruction;

        let adapter_row: &mut AluNativeAdapterCols<F> = adapter_row.borrow_mut();
        adapter_row.a_pointer = a;
        tracing_write_native(
            memory,
            a.as_canonical_u32(),
            data,
            &mut adapter_row.write_aux,
        );
    }

    #[inline(always)]
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        _ctx: (),
        adapter_row: &mut [F],
    ) {
        let adapter_row: &mut AluNativeAdapterCols<F> = adapter_row.borrow_mut();

        let mut timestamp = adapter_row.from_state.timestamp.as_canonical_u32();

        mem_helper.fill_from_prev(timestamp, &mut adapter_row.reads_aux[0].base);
        timestamp += 1;

        mem_helper.fill_from_prev(timestamp, &mut adapter_row.reads_aux[1].base);
        timestamp += 1;

        mem_helper.fill_from_prev(timestamp, adapter_row.write_aux.as_mut());

        if adapter_row.e_as.is_zero() {
            adapter_row.reads_aux[0].is_immediate = F::ONE;
            adapter_row.reads_aux[0].is_zero_aux = F::ZERO;
        } else {
            adapter_row.reads_aux[0].is_immediate = F::ZERO;
            adapter_row.reads_aux[0].is_zero_aux = adapter_row.e_as.inverse();
        }

        if adapter_row.f_as.is_zero() {
            adapter_row.reads_aux[1].is_immediate = F::ONE;
            adapter_row.reads_aux[1].is_zero_aux = F::ZERO;
        } else {
            adapter_row.reads_aux[1].is_immediate = F::ZERO;
            adapter_row.reads_aux[1].is_zero_aux = adapter_row.f_as.inverse();
        }
    }
}

impl<F> AdapterExecutorE1<F> for AluNativeAdapterStep
where
    F: PrimeField32,
{
    type ReadData = [F; 2];
    type WriteData = [F; 1];

    #[inline(always)]
    fn read<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Self::ReadData
    where
        Ctx: E1E2ExecutionCtx,
    {
        let &Instruction { b, c, e, f, .. } = instruction;

        let rs1 = memory_read_or_imm_native_from_state(state, e.as_canonical_u32(), b);
        let rs2 = memory_read_or_imm_native_from_state(state, f.as_canonical_u32(), c);

        [rs1, rs2]
    }

    #[inline(always)]
    fn write<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
        data: &Self::WriteData,
    ) where
        Ctx: E1E2ExecutionCtx,
    {
        let &Instruction { a, .. } = instruction;

        memory_write_native_from_state(state, a.as_canonical_u32(), data);
    }
}
