use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::{
        execution_mode::E1E2ExecutionCtx, AdapterAirContext, AdapterExecutorE1, AdapterTraceStep,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir,
        VmStateMut,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadOrImmediateAuxCols},
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

use crate::adapters::{memory_read_or_imm_native_from_state, tracing_read_or_imm_native};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct BranchNativeAdapterReadCols<T> {
    pub address: MemoryAddress<T, T>,
    pub read_aux: MemoryReadOrImmediateAuxCols<T>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
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

#[derive(derive_new::new)]
pub struct BranchNativeAdapterStep;

impl<F, CTX> AdapterTraceStep<F, CTX> for BranchNativeAdapterStep
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<BranchNativeAdapterCols<u8>>();
    type ReadData = [F; 2];
    type WriteData = ();
    type TraceContext<'a> = ();

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory<F>, adapter_row: &mut [F]) {
        let adapter_row: &mut BranchNativeAdapterCols<F> = adapter_row.borrow_mut();

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
        let &Instruction { a, b, d, e, .. } = instruction;
        let adapter_row: &mut BranchNativeAdapterCols<F> = adapter_row.borrow_mut();

        adapter_row.reads_aux[0].address.pointer = a;
        let rs1 = tracing_read_or_imm_native(
            memory,
            d.as_canonical_u32(),
            a,
            &mut adapter_row.reads_aux[0].address.address_space,
            &mut adapter_row.reads_aux[0].read_aux,
        );
        adapter_row.reads_aux[1].address.pointer = b;
        let rs2 = tracing_read_or_imm_native(
            memory,
            e.as_canonical_u32(),
            b,
            &mut adapter_row.reads_aux[1].address.address_space,
            &mut adapter_row.reads_aux[1].read_aux,
        );
        [rs1, rs2]
    }

    #[inline(always)]
    fn write(
        &self,
        _memory: &mut TracingMemory<F>,
        _instruction: &Instruction<F>,
        _adapter_row: &mut [F],
        _data: &Self::WriteData,
    ) {
    }

    #[inline(always)]
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        _trace_ctx: Self::TraceContext<'_>,
        adapter_row: &mut [F],
    ) {
        let adapter_row: &mut BranchNativeAdapterCols<F> = adapter_row.borrow_mut();

        let mut timestamp = adapter_row.from_state.timestamp.as_canonical_u32();

        mem_helper.fill_from_prev(timestamp, &mut adapter_row.reads_aux[0].read_aux.base);
        timestamp += 1;

        mem_helper.fill_from_prev(timestamp, &mut adapter_row.reads_aux[1].read_aux.base);

        let read_aux0 = &mut adapter_row.reads_aux[0];
        if read_aux0.address.address_space.is_zero() {
            read_aux0.read_aux.is_immediate = F::ONE;
            read_aux0.read_aux.is_zero_aux = F::ZERO;
        } else {
            read_aux0.read_aux.is_immediate = F::ZERO;
            read_aux0.read_aux.is_zero_aux = read_aux0.address.address_space.inverse();
        }

        let read_aux1 = &mut adapter_row.reads_aux[1];
        if read_aux1.address.address_space.is_zero() {
            read_aux1.read_aux.is_immediate = F::ONE;
            read_aux1.read_aux.is_zero_aux = F::ZERO;
        } else {
            read_aux1.read_aux.is_immediate = F::ZERO;
            read_aux1.read_aux.is_zero_aux = read_aux1.address.address_space.inverse();
        }
    }
}

impl<F> AdapterExecutorE1<F> for BranchNativeAdapterStep
where
    F: PrimeField32,
{
    type ReadData = [F; 2];
    type WriteData = ();

    #[inline(always)]
    fn read<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Self::ReadData
    where
        Ctx: E1E2ExecutionCtx,
    {
        let &Instruction { a, b, d, e, .. } = instruction;

        let rs1 = memory_read_or_imm_native_from_state(state, d.as_canonical_u32(), a);
        let rs2 = memory_read_or_imm_native_from_state(state, e.as_canonical_u32(), b);

        [rs1, rs2]
    }

    #[inline(always)]
    fn write<Ctx>(
        &self,
        _state: &mut VmStateMut<F, GuestMemory, Ctx>,
        _instruction: &Instruction<F>,
        _data: &Self::WriteData,
    ) {
    }
}
