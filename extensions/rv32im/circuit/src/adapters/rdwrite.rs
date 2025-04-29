use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, BasicAdapterInterface,
        ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryWriteAuxCols},
        online::{GuestMemory, TracingMemory},
        MemoryAddress, MemoryAuxColsFactory, RecordId,
    },
};
use openvm_circuit_primitives::utils::not;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
};
use serde::{Deserialize, Serialize};

use crate::adapters::tracing_write;

use super::RV32_REGISTER_NUM_LIMBS;

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rv32RdWriteWriteRecord {
    pub from_state: ExecutionState<u32>,
    pub rd_id: Option<RecordId>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32RdWriteAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rd_aux_cols: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32CondRdWriteAdapterCols<T> {
    pub inner: Rv32RdWriteAdapterCols<T>,
    pub needs_write: T,
}

/// This adapter doesn't read anything, and writes to \[a:4\]_d, where d == 1
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32RdWriteAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
}

/// This adapter doesn't read anything, and **maybe** writes to \[a:4\]_d, where d == 1
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32CondRdWriteAdapterAir {
    inner: Rv32RdWriteAdapterAir,
}

impl<F: Field> BaseAir<F> for Rv32RdWriteAdapterAir {
    fn width(&self) -> usize {
        Rv32RdWriteAdapterCols::<F>::width()
    }
}

impl<F: Field> BaseAir<F> for Rv32CondRdWriteAdapterAir {
    fn width(&self) -> usize {
        Rv32CondRdWriteAdapterCols::<F>::width()
    }
}

impl Rv32RdWriteAdapterAir {
    /// If `needs_write` is provided:
    /// - Only writes if `needs_write`.
    /// - Sets operand `f = needs_write` in the instruction.
    /// - Does not put any other constraints on `needs_write`
    ///
    /// Otherwise:
    /// - Writes if `ctx.instruction.is_valid`.
    /// - Sets operand `f` to default value of `0` in the instruction.
    #[allow(clippy::type_complexity)]
    fn conditional_eval<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local_cols: &Rv32RdWriteAdapterCols<AB::Var>,
        ctx: AdapterAirContext<
            AB::Expr,
            BasicAdapterInterface<
                AB::Expr,
                ImmInstruction<AB::Expr>,
                0,
                1,
                0,
                RV32_REGISTER_NUM_LIMBS,
            >,
        >,
        needs_write: Option<AB::Expr>,
    ) {
        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let timestamp_delta = 1;
        let (write_count, f) = if let Some(needs_write) = needs_write {
            (needs_write.clone(), needs_write)
        } else {
            (ctx.instruction.is_valid.clone(), AB::Expr::ZERO)
        };
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.rd_ptr,
                ),
                ctx.writes[0].clone(),
                timestamp,
                &local_cols.rd_aux_cols,
            )
            .eval(builder, write_count);

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP));
        // regardless of `needs_write`, must always execute instruction when `is_valid`.
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    local_cols.rd_ptr.into(),
                    AB::Expr::ZERO,
                    ctx.instruction.immediate,
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::ZERO,
                    f,
                ],
                local_cols.from_state,
                ExecutionState {
                    pc: to_pc,
                    timestamp: timestamp + AB::F::from_canonical_usize(timestamp_delta),
                },
            )
            .eval(builder, ctx.instruction.is_valid);
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32RdWriteAdapterAir {
    type Interface =
        BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, 0, 1, 0, RV32_REGISTER_NUM_LIMBS>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv32RdWriteAdapterCols<AB::Var> = (*local).borrow();
        self.conditional_eval(builder, local_cols, ctx, None);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32RdWriteAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32CondRdWriteAdapterAir {
    type Interface =
        BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, 0, 1, 0, RV32_REGISTER_NUM_LIMBS>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv32CondRdWriteAdapterCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local_cols.needs_write);
        builder
            .when::<AB::Expr>(not(ctx.instruction.is_valid.clone()))
            .assert_zero(local_cols.needs_write);

        self.inner.conditional_eval(
            builder,
            &local_cols.inner,
            ctx,
            Some(local_cols.needs_write.into()),
        );
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32CondRdWriteAdapterCols<_> = local.borrow();
        cols.inner.from_state.pc
    }
}

/// This adapter doesn't read anything, and writes to \[a:4\]_d, where d == 1
#[derive(derive_new::new)]
pub struct Rv32RdWriteAdapterStep;

impl<F, CTX> AdapterTraceStep<F, CTX> for Rv32RdWriteAdapterStep
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32RdWriteAdapterCols<u8>>();

    type ReadData = ();
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];
    type TraceContext<'a> = ();

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, adapter_row: &mut [F]) {
        let adapter_row: &mut Rv32RdWriteAdapterCols<F> = adapter_row.borrow_mut();
        adapter_row.from_state.pc = F::from_canonical_u32(pc);
        adapter_row.from_state.timestamp = F::from_canonical_u32(memory.timestamp);
    }

    #[inline(always)]
    fn read(
        &self,
        _memory: &mut TracingMemory,
        _instruction: &Instruction<F>,
        _adapter_row: &mut [F],
    ) -> Self::ReadData {
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        let &Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        let adapter_row: &mut Rv32RdWriteAdapterCols<F> = adapter_row.borrow_mut();

        adapter_row.rd_ptr = a;
        tracing_write(
            memory,
            d.as_canonical_u32(),
            a.as_canonical_u32(),
            data,
            &mut adapter_row.rd_aux_cols,
        );
    }

    #[inline(always)]
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        _trace_ctx: Self::TraceContext<'_>,
        adapter_row: &mut [F],
    ) {
        let adapter_row: &mut Rv32RdWriteAdapterCols<F> = adapter_row.borrow_mut();

        let timestamp = adapter_row.from_state.timestamp.as_canonical_u32();

        mem_helper.fill_from_prev(timestamp, adapter_row.rd_aux_cols.as_mut());
    }
}

impl<F> AdapterExecutorE1<F> for Rv32RdWriteAdapterStep
where
    F: PrimeField32,
{
    type ReadData = ();
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];

    #[inline(always)]
    fn read<Mem>(&self, _memory: &mut Mem, _instruction: &Instruction<F>) -> Self::ReadData
    where
        Mem: GuestMemory,
    {
    }

    #[inline(always)]
    fn write<Mem>(&self, memory: &mut Mem, instruction: &Instruction<F>, rd: &Self::WriteData)
    where
        Mem: GuestMemory,
    {
        let Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        unsafe { memory.write(d.as_canonical_u32(), a.as_canonical_u32(), rd) };
    }
}

/// This adapter doesn't read anything, and **maybe** writes to \[a:4\]_d, where d == 1
#[derive(derive_new::new)]
pub struct Rv32CondRdWriteAdapterStep {
    inner: Rv32RdWriteAdapterStep,
}

impl<F, CTX> AdapterTraceStep<F, CTX> for Rv32CondRdWriteAdapterStep
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32CondRdWriteAdapterCols<u8>>();
    type ReadData = ();
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];
    type TraceContext<'a> = ();

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, adapter_row: &mut [F]) {
        let adapter_row: &mut Rv32CondRdWriteAdapterCols<F> = adapter_row.borrow_mut();

        adapter_row.inner.from_state.pc = F::from_canonical_u32(pc);
        adapter_row.inner.from_state.timestamp = F::from_canonical_u32(memory.timestamp);
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData {
        <Rv32RdWriteAdapterStep as AdapterTraceStep<F, CTX>>::read(
            &self.inner,
            memory,
            instruction,
            adapter_row,
        )
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        let Instruction { f: enabled, .. } = instruction;

        if *enabled != F::ZERO {
            let (inner_row, needs_write) = unsafe {
                adapter_row.split_at_mut_unchecked(size_of::<Rv32RdWriteAdapterCols<u8>>())
            };

            needs_write[0] = F::ONE;
            <Rv32RdWriteAdapterStep as AdapterTraceStep<F, CTX>>::write(
                &self.inner,
                memory,
                instruction,
                inner_row,
                data,
            );
        } else {
            memory.increment_timestamp();
        }
    }

    #[inline(always)]
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace_ctx: Self::TraceContext<'_>,
        adapter_row: &mut [F],
    ) {
        let adapter_row_ref: &mut Rv32CondRdWriteAdapterCols<F> = adapter_row.borrow_mut();

        if adapter_row_ref.needs_write.is_one() {
            let (inner_row, _) = unsafe {
                adapter_row.split_at_mut_unchecked(size_of::<Rv32RdWriteAdapterCols<u8>>())
            };

            <Rv32RdWriteAdapterStep as AdapterTraceStep<F, CTX>>::fill_trace_row(
                &self.inner,
                mem_helper,
                trace_ctx,
                inner_row,
            )
        }
    }
}

impl<F> AdapterExecutorE1<F> for Rv32CondRdWriteAdapterStep
where
    F: PrimeField32,
{
    type ReadData = ();
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];

    #[inline(always)]
    fn read<Mem>(&self, memory: &mut Mem, instruction: &Instruction<F>) -> Self::ReadData
    where
        Mem: GuestMemory,
    {
        <Rv32RdWriteAdapterStep as AdapterExecutorE1<F>>::read(&self.inner, memory, instruction)
    }

    #[inline(always)]
    fn write<Mem>(&self, memory: &mut Mem, instruction: &Instruction<F>, rd: &Self::WriteData)
    where
        Mem: GuestMemory,
    {
        let Instruction { f: enabled, .. } = instruction;

        if *enabled != F::ZERO {
            <Rv32RdWriteAdapterStep as AdapterExecutorE1<F>>::write(
                &self.inner,
                memory,
                instruction,
                rd,
            )
        }
    }
}
